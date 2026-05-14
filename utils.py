%%writefile utils.py
"""Utilities for training"""

import json, random
from pathlib import Path
import numpy as np
import torch
import rasterio
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, jaccard_score
import shutil

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_norm_stats(samples: list[dict], max_n: int = 500,
                       nodata_eo: float = 0.0, nodata_sar: float = 0.0) -> dict:
    from tqdm.auto import tqdm
    eo_sum  = np.zeros(3, np.float64); eo_sq = np.zeros(3, np.float64)
    eo_cnt  = np.zeros(3, np.float64)
    sar_sum = sar_sq = sar_cnt = 0.0
    subset = random.sample(samples, min(max_n, len(samples)))
    for s in tqdm(subset, desc="Norm stats", leave=False):
        with rasterio.open(s["eo"]) as src:
            eo = src.read([1,2,3]).astype(np.float32) / 255.0
        for c in range(3):
            ch = eo[c].ravel()
            valid = ch[ch > nodata_eo/255.0 + 1e-6]
            eo_sum[c] += valid.sum(); eo_sq[c] += (valid**2).sum()
            eo_cnt[c] += len(valid)
        with rasterio.open(s["sar"]) as src:
            sar = src.read(1).astype(np.float32).ravel() / 255.0
        valid = sar[sar > nodata_sar/255.0 + 1e-6]
        sar_sum += valid.sum(); sar_sq += (valid**2).sum(); sar_cnt += len(valid)
    eo_mean = (eo_sum / eo_cnt).tolist()
    eo_std  = np.sqrt(np.maximum(eo_sq/eo_cnt - (eo_sum/eo_cnt)**2, 0)).tolist()
    sar_mean = [float(sar_sum / sar_cnt)]
    sar_std  = [float(np.sqrt(max(sar_sq/sar_cnt - (sar_sum/sar_cnt)**2, 0)))]
    return {"eo_mean": eo_mean, "eo_std": eo_std,
            "sar_mean": sar_mean, "sar_std": sar_std}


def load_or_compute_stats(path, samples, **kwargs):
    path = Path(path)
    if path.exists():
        stats = json.loads(path.read_text())
        print(f"  Loaded norm stats from {path}")
        return stats
    print("  Computing normalisation statistics...")
    stats = compute_norm_stats(samples, **kwargs)
    path.write_text(json.dumps(stats, indent=2))
    print(f"  Saved norm stats → {path}")
    return stats


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0)
    iou = jaccard_score(targets, preds, average="binary", zero_division=0)
    cm  = confusion_matrix(targets, preds, labels=[0,1])
    return {"iou": float(iou), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "cm": cm}


def threshold_search(logits: np.ndarray, targets: np.ndarray,
                     low: float = 0.10, high: float = 0.90, step: float = 0.02):
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(low, high+step, step):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            targets, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def save_best_checkpoint(model, epoch, val_f1, val_iou, threshold, norm_stats, cfg, ckpt_dir):
    # Standard local save
    path = ckpt_dir / f"best_model.pth"
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "val_f1": val_f1, "val_iou": val_iou,
        "threshold": threshold, "norm_stats": norm_stats, "config": cfg,
    }, path)

    # REPLACED: Use shutil instead of !cp
    drive_path = "/content/drive/MyDrive/galaxeye_results/checkpoints/best_model.pth"

    # Ensure the destination directory exists before copying
    Path(drive_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(path), drive_path)

    return path


def analyse_imbalance(samples: list[dict], max_n: int = 500) -> float:
    from tqdm.auto import tqdm
    total_change = total_pixels = 0
    for s in tqdm(samples[:max_n], desc="Imbalance", leave=False):
        with rasterio.open(s["mask"]) as src:
            mask = src.read(1).astype(np.uint8)
        total_change += int((mask == 1).sum())
        total_pixels += mask.size
    ratio = total_change / total_pixels
    print(f"  Change pixels: {total_change:,} / {total_pixels:,} = {ratio*100:.2f}%")
    print(f"  Imbalance: 1:{(1-ratio)/ratio:.1f} (no-change:change)")
    return ratio

print("✓ utils.py created")


