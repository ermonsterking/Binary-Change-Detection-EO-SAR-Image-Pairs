%%writefile dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import albumentations as A
from pathlib import Path

def collect_samples(data_root: Path, split: str, *args, **kwargs):
    split_dir = data_root / split
    pre_dir, post_dir, target_dir = split_dir / "pre-event", split_dir / "post-event", split_dir / "target"

    samples = []
    if not pre_dir.exists(): return samples

    file_list = sorted([f.name for f in pre_dir.glob("*.tif")])
    for fname in file_list:
        samples.append({
            "eo": str(pre_dir / fname),
            "sar": str(post_dir / fname),
            "mask": str(target_dir / fname)
        })
    return samples

def normalise(arr, mean, std):
    return ((arr - np.array(mean)) / (np.array(std) + 1e-6)).astype(np.float32)

class EOSARDataset(Dataset):
    def __init__(self, samples, norm_stats, mode="train", patch_size=256, **kwargs):
        self.samples = samples
        self.stats = norm_stats
        self.patch_size = patch_size
        self.transforms = A.Compose([
            A.Resize(patch_size, patch_size),
            A.HorizontalFlip(p=0.5) if mode == "train" else A.NoOp(),
            A.VerticalFlip(p=0.5) if mode == "train" else A.NoOp(),
        ], additional_targets={"sar": "image"})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with rasterio.open(s["eo"]) as src:
            eo = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32) / 255.0
        with rasterio.open(s["sar"]) as src:
            sar = src.read(1).astype(np.float32)[..., np.newaxis] / 255.0
        with rasterio.open(s["mask"]) as src:
            mask = src.read(1).astype(np.uint8)

        # CRITICAL FIX: Ensure mask is strictly binary (0 or 1)
        # We treat any value >= 1 as change.
        # If there are no-data values (like 255), we map those to 0.
        binary_mask = np.where((mask >= 1) & (mask < 255), 1, 0).astype(np.uint8)

        aug = self.transforms(image=eo, sar=sar, mask=binary_mask)

        eo = normalise(aug["image"], self.stats["eo_mean"], self.stats["eo_std"])
        sar = normalise(aug["sar"], self.stats["sar_mean"], self.stats["sar_std"])

        img = np.concatenate([eo, sar], axis=-1)
        return {
            "image": torch.from_numpy(img).permute(2, 0, 1),
            "mask": torch.from_numpy(aug["mask"]).long(),
            "path": s["mask"]
        }
