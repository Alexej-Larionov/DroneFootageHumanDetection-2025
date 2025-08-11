import os
import gc
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- Runtime tuning ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 20) // 2))
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


# ---------------- Config  ----------------
MODEL_PATH = "yolo11l.pt"
WINDOW_SIZE = 640
STRIDE = 560
MAX_OBJECTS = 30        
TARGET_CLASS = 0
MERGING_METHOD = "nms"   
IOU_THRESHOLD = 0.6
DEVICE = "cuda:0"
PROCESSING_MODE = "hue"  
TOLERANCE = 1
TILE_BATCH = 128       

KMEANS_PATH = "kmeans_model.pkl"
DOWNSAMPLE_FOR_FEATURES = 0.25
CLUSTER_STATS_PATH = "cluster_stats.json" 
# ---------------- Load KMeans ----------------
try:
    import joblib
except Exception:
    joblib = None

_kmeans = None
if joblib is not None and os.path.exists(KMEANS_PATH):
    try:
        _kmeans = joblib.load(KMEANS_PATH)
    except Exception:
        _kmeans = None

# ---------------- Feature helpers ----------------
def _circular_mean_hue(hue_deg: np.ndarray) -> float:
    rad = hue_deg.astype(np.float32) / 180.0 * (2 * np.pi)
    avg_x = np.cos(rad).mean()
    avg_y = np.sin(rad).mean()
    ang = np.arctan2(avg_y, avg_x) * 180.0 / np.pi
    if ang < 0: ang += 360.0
    if ang >= 180.0: ang -= 180.0
    return float(ang)

def _entropy_rgb_norm(img_bgr: np.ndarray) -> float:
    img = img_bgr.astype(np.uint8)
    ent = []
    for c in range(3):
        h = cv2.calcHist([img],[c],None,[256],[0,256]).ravel()
        p = h / max(h.sum(), 1.0)
        nz = p[p > 0]
        ent.append(float(-(nz * np.log2(nz)).sum()))
    return float(np.mean(ent) / 8.0)

def _features_for_kmeans(img_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
    if DOWNSAMPLE_FOR_FEATURES and DOWNSAMPLE_FOR_FEATURES != 1.0:
        img_small = cv2.resize(img_bgr, (0,0), fx=DOWNSAMPLE_FOR_FEATURES, fy=DOWNSAMPLE_FOR_FEATURES, interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    H = hsv[...,0].astype(np.float32)
    S = hsv[...,1].astype(np.float32)/255.0
    V = hsv[...,2].astype(np.float32)/255.0
    mean_hue = _circular_mean_hue(H)
    mean_sat = float(S.mean())
    mean_val = float(V.mean())
    ent = _entropy_rgb_norm(img_small)
    X = np.array([[ent, mean_hue, mean_sat, mean_val]], dtype=np.float32)
    d = {"entropy": ent, "mean_hue": mean_hue, "mean_saturation": mean_sat, "mean_value": mean_val}
    return X, d

# ---------------- Per-cluster preprocessing ----------------
def gray_world_balance(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    mb, mg, mr = float(img[...,0].mean()), float(img[...,1].mean()), float(img[...,2].mean())
    mgray = (mb + mg + mr) / 3.0 + 1e-6
    img[...,0] *= (mgray / (mb + 1e-6))
    img[...,1] *= (mgray / (mg + 1e-6))
    img[...,2] *= (mgray / (mr + 1e-6))
    return np.clip(img, 0, 255).astype(np.uint8)

def soft_denoise(img_bgr: np.ndarray, method: str = "bilateral", strength: float = 0.8) -> np.ndarray:
    s = float(max(0.0, min(1.5, strength)))
    if method == "nlmeans":
        h = int(5 * s)
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
    d = 7
    sigma = int(25 * s)
    if sigma <= 0:
        return img_bgr
    return cv2.bilateralFilter(img_bgr, d, sigma, sigma)

def clahe_L(img_bgr: np.ndarray, clip: float = 2.5, tiles=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tiles).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def adaptive_gamma(img_bgr: np.ndarray, target_mean: float = 0.5, clip_gamma=(0.7, 1.3)) -> np.ndarray:
    Y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[...,0].astype(np.float32) / 255.0
    mean_y = float(Y.mean() + 1e-6)
    gamma = float(np.clip(np.log(max(target_mean,1e-6)) / np.log(mean_y), clip_gamma[0], clip_gamma[1]))
    x = np.clip(img_bgr.astype(np.float32) / 255.0, 0, 1)
    return np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)

def unsharp_mask(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 0.6) -> np.ndarray:
    blur = cv2.GaussianBlur(img_bgr, (0,0), sigma)
    return cv2.addWeighted(img_bgr, 1 + float(amount), blur, -float(amount), 0)

def _params_from_stats_like(mv: float, ent: float, ms: float):
    p = dict(
        gray_world=True,
        denoise_method="bilateral",
        denoise_strength=0.8,
        clahe_clip=2.5, clahe_tiles=(8,8),
        gamma_target_mean=0.5, gamma_clip=(0.7, 1.3),
        unsharp_sigma=1.2, unsharp_amount=0.6,
    )
    if mv < 0.42:
        p['gamma_target_mean'] = 0.58
        p['clahe_clip'] = max(p['clahe_clip'], 3.0)
    elif mv > 0.65:
        p['gamma_target_mean'] = 0.47
    if ent > 0.65:
        p['denoise_strength'] = 1.0
    elif ent < 0.35:
        p['denoise_strength'] = 0.5
        p['unsharp_amount'] = 0.7
    if ms < 0.3:
        p['clahe_clip'] = max(p['clahe_clip'], 3.0)
    return p

_cluster_stats = None
if os.path.exists(CLUSTER_STATS_PATH):
    try:
        import json
        _cluster_stats = json.loads(Path(CLUSTER_STATS_PATH).read_text(encoding="utf-8"))
    except Exception:
        _cluster_stats = None

def _params_for_cluster(cid: int, feats: dict):
    if _cluster_stats and isinstance(_cluster_stats, dict):
        e = _cluster_stats.get(str(cid)) or _cluster_stats.get(int(cid))
        if isinstance(e, dict):
            mv = float(e.get("mean_value", feats["mean_value"]))
            en = float(e.get("entropy", feats["entropy"]))
            ms = float(e.get("mean_saturation", feats["mean_saturation"]))
            return _params_from_stats_like(mv, en, ms)
    return _params_from_stats_like(feats["mean_value"], feats["entropy"], feats["mean_saturation"])

def preprocess_pipeline(img_bgr: np.ndarray, cid: int, feats: dict) -> np.ndarray:
    p = _params_for_cluster(cid, feats)
    out = img_bgr
    if p.get("gray_world", True): out = gray_world_balance(out)
    ent = float(feats["entropy"]); mv = float(feats["mean_value"]); ms = float(feats["mean_saturation"])
    ds = float(p.get("denoise_strength", 0.8))
    if ds > 0 and ent > 0.55: out = soft_denoise(out, p.get("denoise_method","bilateral"), ds)
    cc = float(p.get("clahe_clip", 2.5))
    if cc > 0 and (mv < 0.45 or mv > 0.65 or ms < 0.3): out = clahe_L(out, cc, p.get("clahe_tiles",(8,8)))
    tgt = float(p.get("gamma_target_mean", 0.5))
    if abs(mv - tgt) > 0.05: out = adaptive_gamma(out, tgt, tuple(p.get("gamma_clip",(0.7,1.3))))
    ua = float(p.get("unsharp_amount", 0.6))
    if ua > 0 and ent < 0.7: out = unsharp_mask(out, float(p.get("unsharp_sigma",1.2)), ua)
    return out


# ---------------- Masking ----------------
def calc_avg_hsv(image):
    small = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    radians = hue / 180.0 * 2 * np.pi
    avg_x = np.cos(radians).mean()
    avg_y = np.sin(radians).mean()
    avg_hue = (np.arctan2(avg_y, avg_x) * 180 / np.pi) % 180
    avg_sat = sat.mean()
    avg_val = val.mean()
    return [avg_hue, avg_sat, avg_val]

def hue_mask(image, avg_hsv, tolerance):
    avg_h, avg_s, avg_v = avg_hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue_diff = np.abs(((hue - avg_h + 90) % 180) - 90)
    s_diff = np.abs(sat - avg_s)
    v_diff = np.abs(val - avg_v)
    mask = (hue_diff >= tolerance) & (s_diff >= tolerance) & (v_diff >= tolerance)
    return mask.astype(np.uint8) * 255


# ---------------- Tiling ----------------
def tile_and_mask(image, mask, size, stride=STRIDE, max_masked_ratio=0.95):
    h, w = image.shape[:2]
    pad_bottom = max(0, size - h)
    pad_right  = max(0, size - w)
    if pad_bottom > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(114,114,114))
        mask  = cv2.copyMakeBorder(mask,  0, pad_bottom, 0, pad_right,  cv2.BORDER_CONSTANT, value=0)
        h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2, y2 = min(x + size, w), min(y + size, h)
            tile_img  = image[y:y2, x:x2]
            tile_mask = mask[y:y2,  x:x2]
            w_real, h_real = tile_img.shape[1], tile_img.shape[0]
            if w_real < size or h_real < size:
                tile_img  = cv2.copyMakeBorder(tile_img,  0, size - h_real, 0, size - w_real, cv2.BORDER_CONSTANT, value=(114,114,114))
                tile_mask = cv2.copyMakeBorder(tile_mask, 0, size - h_real, 0, size - w_real, cv2.BORDER_CONSTANT, value=0)
            masked_ratio = 1.0 - (tile_mask.mean() / 255.0)
            if masked_ratio <= max_masked_ratio:
                tiles.append(((x, y, w_real, h_real), tile_img))
    return tiles


# ---------------- NMS ----------------

def _merge_nms_torch(all_boxes_xyxy: np.ndarray, all_scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if all_boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)
    try:
        from torchvision.ops import nms as tv_nms
        if torch.cuda.is_available():
            t_boxes = torch.as_tensor(all_boxes_xyxy, device='cuda', dtype=torch.float32)
            t_scores = torch.as_tensor(all_scores, device='cuda', dtype=torch.float32)
            keep = tv_nms(t_boxes, t_scores, float(iou_thr)).detach().cpu().numpy()
        else:
            t_boxes = torch.from_numpy(all_boxes_xyxy).float()
            t_scores = torch.from_numpy(all_scores).float()
            keep = tv_nms(t_boxes, t_scores, float(iou_thr)).cpu().numpy()
        return keep
    except Exception:
        idxs = np.argsort(-all_scores)
        keep = []
        while len(idxs):
            i = idxs[0]; keep.append(i)
            if len(idxs) == 1: break
            rest = idxs[1:]
            x1 = np.maximum(all_boxes_xyxy[i,0], all_boxes_xyxy[rest,0])
            y1 = np.maximum(all_boxes_xyxy[i,1], all_boxes_xyxy[rest,1])
            x2 = np.minimum(all_boxes_xyxy[i,2], all_boxes_xyxy[rest,2])
            y2 = np.minimum(all_boxes_xyxy[i,3], all_boxes_xyxy[rest,3])
            iw = np.maximum(0.0, x2 - x1); ih = np.maximum(0.0, y2 - y1)
            inter = iw * ih
            area_i = (all_boxes_xyxy[i,2]-all_boxes_xyxy[i,0])*(all_boxes_xyxy[i,3]-all_boxes_xyxy[i,1])
            area_r = (all_boxes_xyxy[rest,2]-all_boxes_xyxy[rest,0])*(all_boxes_xyxy[rest,3]-all_boxes_xyxy[rest,1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            idxs = rest[iou < iou_thr]
        return np.array(keep, dtype=np.int64)

def _fuse_by_iou_or_center(boxes_xyxy: np.ndarray,
                           scores: np.ndarray,
                           iou_thr: float = 0.55,
                           center_eps: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores
    order = np.argsort(-scores)
    boxes = boxes_xyxy[order].astype(np.float32, copy=False)
    confs = scores[order].astype(np.float32, copy=False)

    used = np.zeros(len(confs), dtype=bool)
    out_boxes, out_scores = [], []

    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    ww = (boxes[:, 2] - boxes[:, 0])
    hh = (boxes[:, 3] - boxes[:, 1])
    mins = np.maximum(1.0, np.minimum(ww, hh))  

    for i in range(len(confs)):
        if used[i]:
            continue
        cluster = [i]
        bi = boxes[i]
        ai = ww[i] * hh[i]

        for j in range(i + 1, len(confs)):
            if used[j]:
                continue
            bj = boxes[j]
            x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
            x2 = min(bi[2], bj[2]); y2 = min(bi[3], bj[3])
            iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
            inter = iw * ih
            aj = ww[j] * hh[j]
            iou = inter / (ai + aj - inter + 1e-6)
            dc = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            near = dc <= center_eps * min(mins[i], mins[j])
            if (iou >= iou_thr) or near:
                cluster.append(j)

        used[cluster] = True
        w = confs[cluster][:, None]
        b = boxes[cluster]
        fused = (b * w).sum(axis=0) / (w.sum(axis=0) + 1e-6)
        out_boxes.append(fused)
        out_scores.append(float(confs[cluster].max())) 

    out_boxes = np.vstack(out_boxes).astype(np.float32)
    out_scores = np.asarray(out_scores, dtype=np.float32)

    keep = []
    for i in range(len(out_boxes)):
        bi = out_boxes[i]
        area_i = (bi[2]-bi[0])*(bi[3]-bi[1])
        ok = True
        for k in keep:
            bj = out_boxes[k]
            x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
            x2 = min(bi[2], bj[2]); y2 = min(bi[3], bj[3])
            inter = max(0.0, x2-x1) * max(0.0, y2-y1)
            area_j = (bj[2]-bj[0])*(bj[3]-bj[1])
            frac = inter / (min(area_i, area_j) + 1e-6)
            if frac >= 0.85: 
                if out_scores[i] <= out_scores[k]:
                    ok = False
                else:
                    keep.remove(k)
                break
        if ok:
            keep.append(i)

    return out_boxes[keep], out_scores[keep]


def _wbf_post(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float = 0.70) -> tuple[np.ndarray, np.ndarray]:

    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores
    order = np.argsort(-scores)
    boxes = boxes_xyxy[order].astype(np.float32, copy=False)
    confs = scores[order].astype(np.float32, copy=False)

    used = np.zeros(len(confs), dtype=bool)
    out_boxes, out_scores = [], []

    for i in range(len(confs)):
        if used[i]:
            continue
        cluster_idxs = [i]
        bi = boxes[i]; ai = (bi[2]-bi[0])*(bi[3]-bi[1])
        for j in range(i+1, len(confs)):
            if used[j]:
                continue
            bj = boxes[j]
            x1 = max(bi[0], bj[0]); y1 = max(bi[1], bj[1])
            x2 = min(bi[2], bj[2]); y2 = min(bi[3], bj[3])
            iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
            inter = iw*ih
            aj = (bj[2]-bj[0])*(bj[3]-bj[1])
            iou = inter / (ai + aj - inter + 1e-6)
            if iou >= iou_thr:
                cluster_idxs.append(j)

        used[cluster_idxs] = True
        w = confs[cluster_idxs]
        b = boxes[cluster_idxs]
        wsum = np.sum(w) + 1e-6
        fused = (b * w[:, None]).sum(axis=0) / wsum
        out_boxes.append(fused)
        out_scores.append(np.max(w))  

    out_boxes = np.vstack(out_boxes).astype(np.float32)
    out_scores = np.asarray(out_scores, dtype=np.float32)
    return out_boxes, out_scores
def _load_model(path: str, device: str) -> YOLO:
    m = YOLO(path)
    try: m.model.to(device)
    except Exception: pass
    try: m.fuse()
    except Exception: pass
    try: m.model.half()
    except Exception: pass
    try: m.model.eval()
    except Exception: pass
    try:
        with torch.inference_mode():
            dev = device if device.startswith("cuda") else ("cuda:"+device if device.isdigit() else device)
            dummy = torch.zeros(1,3,WINDOW_SIZE,WINDOW_SIZE, device=dev, dtype=torch.float16)
            _ = m(dummy, imgsz=WINDOW_SIZE, device=device, verbose=False)
    except Exception:
        pass
    return m

_base_model = _load_model(MODEL_PATH, DEVICE)

_models_by_cluster: Dict[int, YOLO] = {}
root = Path(".")
for cid in range(5):
    p = root / f"c{cid}.pt"
    if p.exists():
        _models_by_cluster[cid] = _load_model(p.as_posix(), DEVICE)
if not _models_by_cluster:
    _models_by_cluster[0] = _base_model

def _select_model_for_image(img_bgr: np.ndarray) -> Tuple[int, YOLO, dict]:
    cid = 0
    feats = {"entropy":0.5, "mean_value":0.5, "mean_saturation":0.5, "mean_hue":90.0}
    if _kmeans is not None:
        try:
            X, feats = _features_for_kmeans(img_bgr)
            cid = int(_kmeans.predict(X)[0])
        except Exception:
            cid = 0
    model = _models_by_cluster.get(cid, _base_model)
    return cid, model, feats


# ---------------- Main predict----------------
def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[Dict]]:
    """
    images: list of RGB np.ndarray or single RGB np.ndarray
    returns: list[ list[ dict(label, xc, yc, w, h, score, w_img, h_img) ] ]
    """
    if isinstance(images, np.ndarray):
        images = [images]

    results_batch: List[List[Dict]] = []

    for image in images:
        h_img, w_img = image.shape[:2]
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cid, model_local, feats = _select_model_for_image(img_bgr)
        img_bgr = preprocess_pipeline(img_bgr, cid, feats)

        mask = hue_mask(img_bgr, calc_avg_hsv(img_bgr), TOLERANCE) if PROCESSING_MODE == "hue" else np.ones(img_bgr.shape[:2], np.uint8)*255

        tiles = tile_and_mask(img_bgr, mask, WINDOW_SIZE)

        tensors: List[torch.Tensor] = []
        metas: List[Tuple[int,int,int,int]] = []  
        for (x0, y0, w_real, h_real), tile in tiles:
            t = torch.from_numpy(tile).permute(2,0,1).contiguous().float().div_(255.0)
            tensors.append(t)
            metas.append((x0, y0, w_real, h_real))

        all_boxes = []
        all_scores = []

        with torch.inference_mode():
            if tensors:
                for s in range(0, len(tensors), TILE_BATCH):
                    e = min(s + TILE_BATCH, len(tensors))
                    batch = torch.stack(tensors[s:e], dim=0).to(DEVICE, non_blocking=True)
                    try:
                        if model_local.model.dtype == torch.float16:
                            batch = batch.half()
                    except Exception:
                        pass

                    preds = model_local(
                        batch,
                        imgsz=WINDOW_SIZE,
                        device=DEVICE,
                        verbose=False,
                        conf=0.6,
                        classes=[TARGET_CLASS],
                        max_det=MAX_OBJECTS
                    )

                    for bi, res in enumerate(preds):
                        x0, y0, w_real, h_real = metas[s + bi]
                        b = getattr(res, "boxes", None)
                        if b is None or len(b) == 0:
                            continue
                        xyxy = b.xyxy.detach().float().cpu().numpy()
                        conf = b.conf.detach().cpu().numpy().astype(np.float32)
                        cls  = b.cls.detach().cpu().numpy().astype(np.int32)

                        m = (cls == TARGET_CLASS)
                        xyxy = xyxy[m]; conf = conf[m]
                        if xyxy.size == 0:
                            continue

                        cx = 0.5 * (xyxy[:,0] + xyxy[:,2])
                        cy = 0.5 * (xyxy[:,1] + xyxy[:,3])
                        real = (cx < w_real) & (cy < h_real)
                        xyxy = xyxy[real]; conf = conf[real]
                        if xyxy.size == 0:
                            continue

                        xyxy[:,[0,2]] += x0
                        xyxy[:,[1,3]] += y0

                        xyxy[:,[0,2]] = np.clip(xyxy[:,[0,2]], 0, w_img)
                        xyxy[:,[1,3]] = np.clip(xyxy[:,[1,3]], 0, h_img)

                        all_boxes.append(xyxy)
                        all_scores.append(conf)

        outputs: List[Dict] = []
        if not all_boxes:
            outputs.append({
                'label': 0,
                'xc': None, 'yc': None, 'w': None, 'h': None,
                'score': None, 'w_img': w_img, 'h_img': h_img
            })
        else:
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_scores = np.concatenate(all_scores, axis=0)
            keep = _merge_nms_torch(all_boxes, all_scores, IOU_THRESHOLD)
            nms_boxes  = all_boxes[keep]
            nms_scores = all_scores[keep]

            fused_boxes, fused_scores = _wbf_post(nms_boxes, nms_scores, iou_thr=max(0.55, IOU_THRESHOLD))
            fused_boxes, fused_scores = _fuse_by_iou_or_center(
            fused_boxes, fused_scores,
            iou_thr=max(IOU_THRESHOLD, 0.8),  
            center_eps=0.3                      
)
            order = np.argsort(-fused_scores)
            fused_boxes  = fused_boxes[order]
            fused_scores = fused_scores[order]
            for (x1, y1, x2, y2), sc in zip(fused_boxes, fused_scores):
                x1 = max(0.0, min(float(x1), w_img)); x2 = max(0.0, min(float(x2), w_img))
                y1 = max(0.0, min(float(y1), h_img)); y2 = max(0.0, min(float(y2), h_img))
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = ((x1 + x2) * 0.5) / max(w_img, 1e-6)
                yc = ((y1 + y2) * 0.5) / max(h_img, 1e-6)
                ww = (x2 - x1) / max(w_img, 1e-6)
                hh = (y2 - y1) / max(h_img, 1e-6)
                xc = float(min(1.0, max(0.0, xc)))
                yc = float(min(1.0, max(0.0, yc)))
                ww = float(min(1.0, max(0.0, ww)))
                hh = float(min(1.0, max(0.0, hh)))
                outputs.append({
                    'label': 0,
                    'xc': xc, 'yc': yc, 'w': ww, 'h': hh,
                    'score': float(sc),
                    'w_img': w_img, 'h_img': h_img
                })

        results_batch.append(outputs)

    return results_batch
