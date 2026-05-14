%%writefile train.py
"""Training script"""

import argparse, gc, json, sys
from pathlib import Path
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import EOSARDataset, collect_samples
from losses  import build_loss
from model   import build_model
from utils   import (seed_everything, load_or_compute_stats, compute_metrics,
                     threshold_search, save_best_checkpoint, analyse_imbalance)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data_root", type=str, default=None)
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        imgs  = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
        del imgs, masks, loss
    torch.cuda.empty_cache(); gc.collect()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total = 0.0
    all_logits, all_targets = [], []
    for batch in tqdm(loader, desc="  Eval", leave=False):
        imgs  = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        total += loss.item()
        all_logits.append(logits.squeeze(1).cpu().float().numpy().ravel())
        all_targets.append(masks.cpu().numpy().ravel())
        del imgs, masks, logits, loss
    torch.cuda.empty_cache(); gc.collect()

    all_logits  = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(preds, all_targets)
    metrics["loss"] = total / len(loader)
    return metrics, all_logits, all_targets


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    seed_everything(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    DATA_ROOT = Path(cfg["data"]["root"])
    CKPT_DIR  = Path(cfg["output"]["checkpoint_dir"])
    RES_DIR   = Path(cfg["output"]["results_dir"])
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    def _collect(split):
        return collect_samples(DATA_ROOT, split, cfg["data"]["eo_variants"],
                              cfg["data"]["sar_variants"], cfg["data"]["mask_variants"])

    train_samples = _collect("train")
    val_samples   = _collect("val")
    print(f"Samples — train: {len(train_samples)}  val: {len(val_samples)}")

    print("\nClass imbalance:")
    change_ratio = analyse_imbalance(train_samples)

    norm_stats = load_or_compute_stats(
        cfg["norm_stats_path"], train_samples,
        nodata_eo=cfg["data"]["nodata_eo"], nodata_sar=cfg["data"]["nodata_sar"])

    needs_remap = cfg["label"]["needs_remap"]
    patch_size  = cfg["train"]["patch_size"]

    train_ds = EOSARDataset(train_samples, norm_stats, mode="train",
                           patch_size=patch_size, needs_remap=needs_remap,
                           cfg_aug=cfg.get("augmentation"),
                           nodata_eo=cfg["data"]["nodata_eo"],
                           nodata_sar=cfg["data"]["nodata_sar"])
    val_ds = EOSARDataset(val_samples, norm_stats, mode="val",
                         patch_size=patch_size, needs_remap=needs_remap,
                         nodata_eo=cfg["data"]["nodata_eo"],
                         nodata_sar=cfg["data"]["nodata_sar"])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=True, num_workers=cfg["train"]["num_workers"],
                             pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                           shuffle=False, num_workers=cfg["train"]["num_workers"],
                           pin_memory=True, persistent_workers=True)

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    criterion = build_loss(cfg["loss"])
    print(f"\nModel: {cfg['model']['name']} ({n_params:.2f}M params)")

    optimizer = getattr(torch.optim, cfg["optimizer"]["name"])(
        model.parameters(), lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["num_epochs"],
        eta_min=cfg["scheduler"]["eta_min"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["mixed_precision"])

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_iou": []}
    best_val_f1 = 0.0; best_threshold = 0.5; best_ckpt_path = None
    no_improve = 0; patience = cfg["train"]["patience"]

    hdr = f"{'Ep':>4} | {'TrLoss':>7} | {'VaLoss':>7} | {'F1':>7} | {'IoU':>7} | {'LR':>9}"
    print(f"\n{hdr}"); print("-"*len(hdr))

    for epoch in range(1, cfg["train"]["num_epochs"]+1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion,
                             scaler, device, cfg["train"]["grad_clip"])
        va_m, _, _ = eval_epoch(model, val_loader, criterion, device, best_threshold)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_m["loss"])
        history["val_f1"].append(va_m["f1"])
        history["val_iou"].append(va_m["iou"])

        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]; no_improve = 0
            best_ckpt_path = save_best_checkpoint(
                model, epoch, va_m["f1"], va_m["iou"],
                best_threshold, norm_stats, cfg, CKPT_DIR)
            flag = " ★"
        else:
            no_improve += 1; flag = ""

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>4} | {tr_loss:>7.4f} | {va_m['loss']:>7.4f} | "
              f"{va_m['f1']:>7.4f} | {va_m['iou']:>7.4f} | {lr_now:>9.2e}{flag}")

        if no_improve >= patience:
            print(f"\nEarly stop @ epoch {epoch}")
            break

    print(f"\nBest Val F1: {best_val_f1:.4f}")

    print("\nThreshold search...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, val_logits, val_targets = eval_epoch(model, val_loader, criterion, device, 0.5)
    best_threshold, best_f1 = threshold_search(val_logits, val_targets)
    print(f"Best threshold: {best_threshold:.2f} (Val F1={best_f1:.4f})")

    best_ckpt_path = save_best_checkpoint(
        model, ckpt["epoch"], best_f1, ckpt["val_iou"],
        best_threshold, norm_stats, cfg, CKPT_DIR)

    hist_path = RES_DIR / "training_history.json"
    hist_path.write_text(json.dumps({**history, "best_val_f1": best_val_f1,
                                     "best_threshold": best_threshold}, indent=2))
    print(f"\nDone. Checkpoint → {best_ckpt_path}")

if __name__ == "__main__":
    main()

print("✓ train.py created")



