# Quick Start — 5 Steps to Train & Evaluate

**For those who want to start immediately without reading full docs.**

---

## Prerequisites
- Kaggle account (free GPU)
- Dataset from [HuggingFace](https://huggingface.co/datasets/doron333/change-detection-dataset)

---

## Step 1 — Upload to Kaggle

1. Create new notebook at [kaggle.com/code](https://www.kaggle.com/code)
2. Set GPU: **T4 x2** (Settings → Accelerator)
3. Upload all `.py` files + `config.yaml` + `requirements.txt` to `/kaggle/working/`
4. Add dataset as input (mounts at `/kaggle/input/change-detection-dataset/`)

---

## Step 2 — Install Dependencies

**Notebook cell 1:**
```python
!pip install -q rasterio albumentations pyyaml
```

---

## Step 3 — Train

**Notebook cell 2:**
```python
import os
os.chdir('/kaggle/working')

!python train.py \
    --config config.yaml \
    --data_root /kaggle/input/change-detection-dataset
```

**⏱ Time:** 3-4 hours. Training progress prints every epoch.

**Output:** `checkpoints/best_ep<N>_f1<X.XXXX>.pth`

---

## Step 4 — Evaluate on Test Set

**Notebook cell 3:**
```python
!python eval.py \
    --data_path /kaggle/input/change-detection-dataset/test \
    --weights checkpoints/best_ep*.pth \
    --save_visuals \
    --n_visuals 8
```

**Outputs:**
- `results/metrics_test.json` ← **Use these numbers in your report**
- `results/confusion_matrix_test.png`
- `results/error_analysis_test.png`

---

## Step 5 — Download Results

In Kaggle file browser (left sidebar):
1. Navigate to `checkpoints/` → download `.pth` file
2. Navigate to `results/` → download all `.json` and `.png` files

**Or zip everything:**
```python
import shutil
shutil.make_archive('submission', 'zip', '/kaggle/working')
# Download submission.zip from file browser
```

---

## What's Next?

1. **Upload checkpoint to Google Drive** → get shareable link
2. **Update README.md** with the link (under "Model Weights")
3. **Push code to GitHub** (make repo public)
4. **Write technical report PDF** using metrics from `results/metrics_test.json`
5. **Submit:**
   - GitHub repo link (via form)
   - ZIP containing: `best_model.pth`, `technical_report.pdf`, `time_resource_log.txt`

---

## Config Tweaks (Optional)

Edit `config.yaml` before Step 3:

**If GPU runs out of memory:**
```yaml
train:
  batch_size: 8    # reduce from 16
  patch_size: 128  # reduce from 256
```

**If training is too slow:**
```yaml
train:
  num_epochs: 50   # reduce from 80
  patience: 10     # reduce from 15
```

**To try different loss weights:**
```yaml
loss:
  dice_weight:  0.6   # increase Dice emphasis
  focal_weight: 0.4   # decrease Focal
```

---

**That's it!** For detailed explanations, see `STEP_BY_STEP_GUIDE.md` or `README.md`.
