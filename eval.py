%%writefile eval.py
"""Evaluation script"""
from scipy.ndimage import binary_opening, binary_closing
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

import sys
sys.path.append('.')

from dataset import EOSARDataset, collect_samples
from model import build_model
from utils import compute_metrics, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = args.threshold if args.threshold else ckpt["threshold"]
    norm_stats = ckpt.get("norm_stats")
    if not norm_stats:
        norm_stats = json.loads(Path(cfg["norm_stats_path"]).read_text())

    model.eval()
    print(f"Loaded: epoch {ckpt['epoch']} val_f1={ckpt['val_f1']:.4f}")
    print(f"Threshold: {threshold:.2f}")

    split_dir = Path(args.data_path)
    split_name = split_dir.name

    samples = collect_samples(
        split_dir.parent, split_name,
        cfg["data"]["eo_variants"], cfg["data"]["sar_variants"],
        cfg["data"]["mask_variants"])
    print(f"Samples: {len(samples)}")

    ds = EOSARDataset(samples, norm_stats, mode="test",
                     patch_size=cfg["train"]["patch_size"],
                     needs_remap=cfg["label"]["needs_remap"],
                     nodata_eo=cfg["data"]["nodata_eo"],
                     nodata_sar=cfg["data"]["nodata_sar"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    all_logits, all_targets = [], []
    all_targets = []
    all_cleaned_preds = []

    # Kernel sizes for morphological operations
    # 3x3 removes tiny noise, 5x5 fills in larger building gaps
    open_kernel = np.ones((3, 3), dtype=int)
    close_kernel = np.ones((5, 5), dtype=int)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {split_name}"):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].numpy()

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(imgs).squeeze(1).cpu().float().numpy()

            # Convert to probabilities and threshold immediately
            probs = 1.0 / (1.0 + np.exp(-logits))
            batch_preds = (probs >= threshold).astype(np.uint8)

            # Clean each prediction in the batch
            for i in range(batch_preds.shape[0]):
                # 1. Opening: Erase tiny false positives (like the forest noise)
                opened = binary_opening(batch_preds[i], structure=open_kernel)

                # 2. Closing: Smooth edges and fill gaps (for the blobby buildings)
                closed = binary_closing(opened, structure=close_kernel)

                all_cleaned_preds.append(closed.ravel())

            all_targets.append(masks.ravel())

    # Concatenate all processed predictions and targets
    preds = np.concatenate(all_cleaned_preds)
    all_targets = np.concatenate(all_targets)

    # Compute metrics on the CLEANED predictions
    metrics = compute_metrics(preds, all_targets)
    print(f"\n{'='*54}")
    print(f"  {split_name.upper()}  (threshold={threshold:.2f})")
    print(f"{'='*54}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"{'='*54}")

    out = {
        "split": split_name, "threshold": round(threshold, 4),
        "iou": round(metrics["iou"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "confusion_matrix": metrics["cm"].tolist(),
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    mpath = out_dir / f"metrics_{split_name}.json"
    mpath.write_text(json.dumps(out, indent=2))
    print(f"Saved → {mpath}")

if __name__ == "__main__":
    main()

print("✓ eval.py created")

