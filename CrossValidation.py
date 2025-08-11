import os
import sys
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Please install with `pip install pyyaml`.") from e

from ultralytics import YOLO


# ---------------------- helpers from user's conventions ----------------------

def find_clusters(root: Path) -> List[int]:
    ids = []
    for p in sorted(root.glob("cluster_*")):
        try:
            cid = int(p.name.split("_")[-1])
        except Exception:
            continue
        if (p / "_lists" / "data.yaml").exists():
            ids.append(cid)
    return ids

def yaml_for_cluster(root: Path, cid: int) -> Path:
    return (root / f"cluster_{cid}" / "_lists" / "data.yaml").resolve()

def find_model_for_cluster(runs_root: Path, cid: int) -> Optional[Path]:
    best = runs_root / f"c{cid}" / "weights" / "best.pt"
    last = runs_root / f"c{cid}" / "weights" / "last.pt"
    if best.exists():
        return best.resolve()
    if last.exists():
        return last.resolve()
    # fallback to any .pt
    any_weights = list((runs_root / f"c{cid}" / "weights").glob("*.pt"))
    if any_weights:
        return sorted(any_weights)[0].resolve()
    return None

def cuda_clean():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------- dataset reading (val images & GT) ----------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _read_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _gather_images_from_dir(d: Path) -> List[Path]:
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(d.rglob(f"*{ext}"))
    return sorted(set(imgs))

def _read_list_file(p: Path) -> List[Path]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(Path(s).resolve())
    return out

def load_val_image_list(dyaml: Path) -> List[Path]:
    y = _read_yaml(dyaml)
    if "val" not in y or y["val"] is None:
        raise RuntimeError(f"'val' is not specified in {dyaml}")
    v = y["val"]
    if isinstance(v, list):
        # list of paths or dirs
        out = []
        for item in v:
            p = Path(item)
            if p.is_dir():
                out.extend(_gather_images_from_dir(p))
            elif p.is_file():
                if p.suffix.lower() in IMG_EXTS:
                    out.append(p.resolve())
                else:
                    # assume list file
                    out.extend(_read_list_file(p))
            else:
                # maybe relative to yaml
                p2 = (dyaml.parent / p).resolve()
                if p2.is_dir():
                    out.extend(_gather_images_from_dir(p2))
                elif p2.is_file():
                    if p2.suffix.lower() in IMG_EXTS:
                        out.append(p2)
                    else:
                        out.extend(_read_list_file(p2))
        return sorted(set(out))
    else:
        p = Path(v)
        # resolve relative to yaml if needed
        if not p.exists():
            p = (dyaml.parent / p).resolve()
        if p.is_dir():
            return _gather_images_from_dir(p)
        elif p.is_file():
            if p.suffix.lower() in IMG_EXTS:
                return [p.resolve()]
            else:
                return _read_list_file(p)
        else:
            raise RuntimeError(f"Cannot resolve val path: {v} in {dyaml}")

def guess_label_path(img_path: Path) -> Path:
    """
    Try common YOLO layout:
      .../images/.../name.jpg  ->  .../labels/.../name.txt
    fallback: same dir as image with .txt
    """
    p = img_path.as_posix()
    if "/images/" in p.replace("\\", "/"):
        p_lbl = p.replace("/images/", "/labels/")
        p_lbl = str(Path(p_lbl).with_suffix(".txt"))
        cand = Path(p_lbl)
        if cand.exists():
            return cand.resolve()
    # alt: images and labels are sibling dirs
    # try walking up once
    parent = img_path.parent
    if parent.name.lower() == "images":
        sib = parent.parent / "labels" / (img_path.stem + ".txt")
        if sib.exists():
            return sib.resolve()
    # fallback
    return (img_path.parent / (img_path.stem + ".txt")).resolve()

def read_yolo_labels(txt_path: Path) -> np.ndarray:
    """
    Returns array of shape (N, 5): [cls, xc, yc, w, h] in normalized coords.
    If no file or empty -> (0,5)
    """
    if not txt_path.exists():
        return np.zeros((0,5), dtype=np.float32)
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                rows.append([cls, xc, yc, w, h])
            except Exception:
                continue
    if not rows:
        return np.zeros((0,5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

def xywhn_to_xyxy_norm(xywhn: np.ndarray) -> np.ndarray:
    """[xc,yc,w,h] -> [x1,y1,x2,y2] in normalized coords [0,1]."""
    xc, yc, w, h = xywhn.T
    x1 = xc - w/2.0
    y1 = yc - h/2.0
    x2 = xc + w/2.0
    y2 = yc + h/2.0
    return np.stack([x1, y1, x2, y2], axis=1)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two boxes in [x1,y1,x2,y2] normalized coordinates.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aw = max(0.0, ax2 - ax1); ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1); bh = max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union

# ---------------------- offline matching & metric computation ----------------------

def greedy_match_single_image(
    det_xyxy: np.ndarray, det_cls: np.ndarray, det_conf: np.ndarray,
    gt_xyxy: np.ndarray, gt_cls: np.ndarray,
    iou_thr: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy one-to-one matching (like COCO):
      - sort detections by conf desc
      - for each det, assign to the unmatched GT of the same class with max IoU >= iou_thr
    Returns:
      is_tp (bool array for each detection in the *original order* of inputs)
      matched_gt_idx (int array with -1 for unmatched)
    """
    if det_xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.int32) - 1

    order = np.argsort(-det_conf)
    is_tp_sorted = np.zeros(det_xyxy.shape[0], dtype=bool)
    matched_idx_sorted = np.zeros(det_xyxy.shape[0], dtype=np.int32) - 1
    gt_used = np.zeros(gt_xyxy.shape[0], dtype=bool)

    for k, di in enumerate(order):
        best_j = -1
        best_iou = -1.0
        for j in range(gt_xyxy.shape[0]):
            if gt_used[j]:
                continue
            if det_cls[di] != gt_cls[j]:
                continue
            iou = iou_xyxy(det_xyxy[di], gt_xyxy[j])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            is_tp_sorted[k] = True
            matched_idx_sorted[k] = best_j
            gt_used[best_j] = True

    # map back to original detection order
    is_tp = np.zeros_like(is_tp_sorted)
    matched_idx = np.zeros_like(matched_idx_sorted)
    is_tp[order] = is_tp_sorted
    matched_idx[order] = matched_idx_sorted
    return is_tp, matched_idx

def build_global_records_for_model(
    model_path: Path,
    image_paths: List[Path],
    iou_thr: float,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
    conf_min: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Runs predict once on all images (conf_min) and matches to GT.
    Returns:
      det_conf_all: (M,) confidences for all detections
      det_is_tp_all: (M,) bool flags whether that detection is a true positive (at IoU thr)
      total_gt: total number of GT boxes across the dataset
    """
    model = YOLO(model_path.as_posix())
    # Run prediction once
    results = model.predict(
        source=[p.as_posix() for p in image_paths],
        device=device, imgsz=imgsz, batch=batch, workers=workers,
        conf=conf_min, verbose=False, iou=0.7  # iou param here is NMS IoU, separate from matching IoU
    )

    det_conf_list = []
    det_tp_list = []
    total_gt = 0

    # map image path -> index in our inputs for reliable pairing
    # Ultralytics 'results' aligns with inputs; each result has 'path'
    path_to_idx = {p.as_posix(): idx for idx, p in enumerate(image_paths)}

    # preload GT for all images
    gts = []
    for img_p in image_paths:
        lab = guess_label_path(img_p)
        gt = read_yolo_labels(lab)  # [cls, xc, yc, w, h], normalized
        if gt.shape[0] == 0:
            gt_xyxy = np.zeros((0,4), dtype=np.float32)
            gt_cls = np.zeros((0,), dtype=np.int32)
        else:
            gt_xyxy = xywhn_to_xyxy_norm(gt[:,1:5])
            gt_cls = gt[:,0].astype(np.int32)
        gts.append((gt_xyxy, gt_cls))
        total_gt += gt_cls.shape[0]

    # iterate results in the same order
    for r in results:
        img_path = Path(r.path).as_posix()
        idx = path_to_idx.get(img_path, None)
        if idx is None:
            # try basename match as fallback
            candidates = [i for i,p in enumerate(image_paths) if p.name == Path(img_path).name]
            idx = candidates[0] if candidates else None
        if idx is None:
            # skip if cannot align
            det_conf_list.append(np.zeros((0,), dtype=np.float32))
            det_tp_list.append(np.zeros((0,), dtype=bool))
            continue

        gt_xyxy, gt_cls = gts[idx]

        if getattr(r, "boxes", None) is None or r.boxes is None or r.boxes.data is None:
            det_conf_list.append(np.zeros((0,), dtype=np.float32))
            det_tp_list.append(np.zeros((0,), dtype=bool))
            continue

        det_conf = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        det_cls = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
        # use normalized xywh (already normalized to input image size)
        if hasattr(r.boxes, "xywhn") and r.boxes.xywhn is not None:
            det_xywhn = r.boxes.xywhn.detach().cpu().numpy().astype(np.float32)
        else:
            # fallback: compute from absolute coords using image shape; convert to normalized
            if hasattr(r.boxes, "xyxy") and r.boxes.xyxy is not None:
                xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                # we need image width/height
                # ultralytics Results has orig_shape (h,w)
                h, w = r.orig_shape
                # convert to normalized xywh
                x1, y1, x2, y2 = xyxy[:,0], xyxy[:,1], xyxy[:,2], xyxy[:,3]
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                ww = (x2 - x1) / w
                hh = (y2 - y1) / h
                det_xywhn = np.stack([xc,yc,ww,hh], axis=1).astype(np.float32)
            else:
                det_xywhn = np.zeros((0,4), dtype=np.float32)

        det_xyxy = xywhn_to_xyxy_norm(det_xywhn)

        is_tp, _ = greedy_match_single_image(det_xyxy, det_cls, det_conf, gt_xyxy, gt_cls, iou_thr=iou_thr)

        det_conf_list.append(det_conf)
        det_tp_list.append(is_tp)

    det_conf_all = np.concatenate(det_conf_list, axis=0) if det_conf_list else np.zeros((0,), dtype=np.float32)
    det_tp_all = np.concatenate(det_tp_list, axis=0) if det_tp_list else np.zeros((0,), dtype=bool)

    del model
    cuda_clean()
    return det_conf_all, det_tp_all, total_gt

def compute_pr_curve(det_conf: np.ndarray, det_is_tp: np.ndarray, total_gt: int):
    """
    Build PR curve by sweeping confidence threshold downward.
    Returns: thresholds, precision, recall, f1
    """
    if det_conf.shape[0] == 0:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
    order = np.argsort(-det_conf)
    conf_sorted = det_conf[order]
    tp_sorted = det_is_tp[order].astype(np.int32)

    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)
    # unique thresholds at each position
    thresholds = conf_sorted
    precision = cum_tp / np.maximum(1, (cum_tp + cum_fp))
    recall = cum_tp / max(1, total_gt)
    f1 = 2 * precision * recall / np.maximum(1e-12, (precision + recall))
    return thresholds, precision, recall, f1

def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute AP via 101-point interpolation (COCO-style uses 101 pts).
    Assumes recall is non-decreasing when sorted by threshold descending.
    We'll enforce monotonic precision envelope.
    """
    if precision.size == 0:
        return 0.0
    # sort by recall asc
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    # compute precision envelope (monotone decreasing when moving to higher recall)
    for i in range(p.size - 2, -1, -1):
        p[i] = max(p[i], p[i+1])
    # sample 101 recall points
    recall_points = np.linspace(0, 1, 101)
    p_interp = np.zeros_like(recall_points)
    j = 0
    for i, rp in enumerate(recall_points):
        while j < r.size and r[j] < rp:
            j += 1
        if j == 0:
            p_interp[i] = p[0]
        elif j >= r.size:
            p_interp[i] = p[-1]
        else:
            # linear interpolation
            if r[j] == r[j-1]:
                p_interp[i] = max(p[j], p[j-1])
            else:
                t = (rp - r[j-1]) / (r[j] - r[j-1])
                p_interp[i] = p[j-1] * (1 - t) + p[j] * t
    return float(np.mean(p_interp))


# ---------------------- main routine ----------------------

def offline_tune(
    root: Path,
    runs_root: Path,
    clusters: Optional[List[int]] = None,
    device: str = "0",
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    iou_match: float = 0.7,     # IoU for TP/FP matching to GT (target metric IoU)
    conf_min: float = 0.001,    # run predict once at this threshold
    metric: str = "F1",         # "F1" or "AP" (AP at iou_match)
    eval_scope: str = "all",    # "all" (all clusters' val) or "own" (only its cluster's val)
    save_traces: bool = False,
    out_csv: Path = Path("offline_conf_best.csv")
) -> pd.DataFrame:
    """
    Returns a DataFrame with best confidence per model for the requested metric.
    """
    all_ids = find_clusters(root)
    if not all_ids:
        raise RuntimeError(f"No clusters found under {root}")

    if clusters:
        eval_ids = [cid for cid in clusters if cid in all_ids]
        if not eval_ids:
            raise RuntimeError(f"Requested clusters {clusters} not found. Available: {all_ids}")
    else:
        eval_ids = all_ids

    yaml_map: Dict[int, Path] = {cid: yaml_for_cluster(root, cid) for cid in eval_ids}

    # models present
    model_map: Dict[int, Path] = {}
    for cid in sorted(all_ids):
        mp = find_model_for_cluster(runs_root, cid)
        if mp is not None:
            model_map[cid] = mp
    if not model_map:
        raise RuntimeError(f"No models found under {runs_root}")

    rows_best = []

    for mid, mpath in model_map.items():
        print(f"\n== Offline tuning for model c{mid} @ {mpath.name} ==")
        # which validation images to use
        if eval_scope == "own" and mid in yaml_map:
            target_ids = [mid]
        else:
            target_ids = list(yaml_map.keys())

        # accumulate global records across selected clusters
        det_conf_all = np.zeros((0,), dtype=np.float32)
        det_tp_all = np.zeros((0,), dtype=bool)
        total_gt = 0

        for did in target_ids:
            dyaml = yaml_map[did]
            img_list = load_val_image_list(dyaml)
            if not img_list:
                print(f"  [WARN] Empty val list for cluster {did} ({dyaml})")
                continue
            print(f"  - cluster {did}: {len(img_list)} images")
            conf_i, tp_i, gt_i = build_global_records_for_model(
                model_path=mpath,
                image_paths=img_list,
                iou_thr=iou_match,
                device=device,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                conf_min=conf_min
            )
            det_conf_all = np.concatenate([det_conf_all, conf_i], axis=0)
            det_tp_all = np.concatenate([det_tp_all, tp_i], axis=0)
            total_gt += gt_i

        # build PR and compute metric vs thresholds
        thr, prec, rec, f1 = compute_pr_curve(det_conf_all, det_tp_all, total_gt)
        if thr.size == 0:
            best_conf = float("nan")
            best_val = float("nan")
        else:
            if metric.upper() == "F1":
                best_idx = int(np.nanargmax(f1))
                best_conf = float(thr[best_idx])
                best_val = float(f1[best_idx])
            elif metric.upper() == "AP":
                # compute AP at given IoU; choose confidence where F1 is maximal as a practical operating point
                ap = compute_ap_from_pr(prec, rec)
                best_idx = int(np.nanargmax(f1))
                best_conf = float(thr[best_idx])
                best_val = float(ap)
            else:
                raise ValueError("Unsupported metric. Use 'F1' or 'AP'.")

        rows_best.append({
            "model_cluster": mid,
            "weights": mpath.as_posix(),
            "eval_scope": eval_scope,
            "iou_match": iou_match,
            "metric": metric.upper(),
            "best_conf": best_conf,
            "best_metric_value": best_val,
            "total_gt": int(total_gt),
            "num_detections": int(det_conf_all.shape[0])
        })

        if save_traces:
            trace = pd.DataFrame({
                "threshold": thr.astype(np.float32),
                "precision": prec.astype(np.float32),
                "recall": rec.astype(np.float32),
                "f1": f1.astype(np.float32)
            })
            trace.to_csv(f"offline_conf_trace_c{mid}.csv", index=False, encoding="utf-8")

        cuda_clean()

    df_best = pd.DataFrame(rows_best).sort_values("model_cluster").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_best.to_csv(out_csv, index=False, encoding="utf-8")
    print("\nBest confidence per model:")
    print(df_best.to_string(index=False))
    print(f"\nSaved: {out_csv.resolve().as_posix()}")
    return df_best


# ---------------------- CLI ----------------------

def parse_clusters_arg(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return [int(p) for p in parts]

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Offline confidence tuning (single-pass inference).")
    ap.add_argument("--root", type=str, default="tiles_offline",
                    help="Root dir with cluster_*/_lists/data.yaml")
    ap.add_argument("--runs", type=str, default="submission/IMPORTANT",
                    help="Runs root with weights under c{ID}/weights/*.pt")
    ap.add_argument("--clusters", type=str, default="", help="Subset of clusters to evaluate, e.g. '0,2,4'")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iou", type=float, default=0.7, help="IoU for GT matching & metric")
    ap.add_argument("--conf-min", type=float, default=0.001, help="Single-pass predict threshold")
    ap.add_argument("--metric", type=str, default="F1", choices=["F1","AP"], help="Target metric to optimize")
    ap.add_argument("--eval-scope", type=str, default="all", choices=["all","own"],
                    help="'all' = all clusters' val; 'own' = only its own cluster val")
    ap.add_argument("--save-traces", action="store_true", help="Save per-model PR traces")
    ap.add_argument("--out-csv", type=str, default="offline_conf_best.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    runs_root = Path(args.runs).resolve()
    clusters = parse_clusters_arg(args.clusters)

    offline_tune(
        root=root,
        runs_root=runs_root,
        clusters=clusters,
        device=args.device,
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        iou_match=float(args.iou),
        conf_min=float(args.conf_min),
        metric=args.metric,
        eval_scope=args.eval_scope,
        save_traces=bool(args.save_traces),
        out_csv=Path(args.out_csv)
    )

