# Step-by-Step Execution Guide

This guide shows **exactly** where and how to run each piece of code for the GalaxEye assignment.

---

## Overview: Two Workflows

### **Workflow A: Kaggle Notebook (Recommended for Assignment)**
- **Best for:** Training with free GPU access
- **Storage:** 20 GB persistent + temp workspace
- **GPU:** Tesla T4 x2 (30 hrs/week free)

### **Workflow B: Local/Colab (Alternative)**
- **Best for:** Testing code structure, quick iteration
- **Requires:** Local GPU or Colab Pro

---

## Workflow A — Kaggle Notebook (RECOMMENDED)

### Step 1: Download This Repository
```bash
git clone https://github.com/your-username/galaxeye-change-detection.git
cd galaxeye-change-detection
```

Or download as ZIP from GitHub and extract.

---

### Step 2: Upload to Kaggle

**2a. Create New Notebook**
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Set accelerator: **GPU T4 x2** (right sidebar under "Settings")
4. Set persistence: **ON** (saves outputs between sessions)

**2b. Upload Code Files**
In the Kaggle notebook interface:
1. Click **"File" → "Upload"** or drag files into the file browser (left sidebar)
2. Upload ALL `.py` files:
   - `dataset.py`
   - `model.py`
   - `losses.py`
   - `utils.py`
   - `train.py`
   - `eval.py`
3. Upload `config.yaml`
4. Upload `requirements.txt`

**Important:** Files uploaded to `/kaggle/working/` persist between sessions. Files in `/kaggle/input/` are read-only dataset mounts.

---

### Step 3: Add Dataset as Kaggle Input

**Option 1 — Use Kaggle Datasets (preferred):**
1. Search for "change detection EO SAR" in Kaggle Datasets
2. If the dataset is already on Kaggle, click **"Add Data"** in your notebook
3. It mounts at `/kaggle/input/change-detection-dataset/`

**Option 2 — Upload from HuggingFace:**
Create a notebook cell and run:
```python
!pip install -q huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="doron333/change-detection-dataset",
    repo_type="dataset",
    local_dir="/kaggle/working/data",
)
```
This takes ~10 min (dataset is ~3 GB). Run **once**, then comment out for future runs.

---

### Step 4: Install Dependencies

**Create a cell at the top of your notebook:**
```python
!pip install -q rasterio albumentations pyyaml
```

**Run this cell once.** Kaggle already has PyTorch, NumPy, scikit-learn, matplotlib.

---

### Step 5: Verify File Alignment

**Create another cell:**
```python
import os
os.chdir('/kaggle/working')  # All code files should be here

# Check code files are present
for f in ['dataset.py', 'model.py', 'losses.py', 'utils.py', 'train.py', 'eval.py', 'config.yaml']:
    assert os.path.exists(f), f"Missing {f} — upload it!"
print("✓ All code files present")

# Check dataset
data_paths = ['/kaggle/input/change-detection-dataset', '/kaggle/working/data']
DATA_ROOT = None
for p in data_paths:
    if os.path.exists(p):
        DATA_ROOT = p
        break
assert DATA_ROOT is not None, "Dataset not found! Upload via Step 3."
print(f"✓ Dataset found at {DATA_ROOT}")
```

---

### Step 6: Run Training

**Create a new cell:**
```python
!python train.py --config config.yaml --data_root {DATA_ROOT}
```

**What happens:**
1. Loads config from `config.yaml`
2. Collects samples from `{DATA_ROOT}/train/`, `{DATA_ROOT}/val/`
3. Computes normalisation stats (cached to `norm_stats.json`)
4. Trains for up to 80 epochs (early stopping after 15 no-improve)
5. Saves best checkpoint to `checkpoints/best_ep<N>_f1<X.XXXX>.pth`
6. Optimises threshold on validation set
7. Saves training curves to `results/training_curves.png`

**Time:** ~3-4 hours on T4 x2.

**Monitor:** Epoch progress prints to cell output. Loss/F1/IoU printed every epoch.

**If interrupted:** Resume with:
```python
!python train.py --config config.yaml --data_root {DATA_ROOT} --resume checkpoints/best_ep<N>_f1<X.XXXX>.pth
```

---

### Step 7: Run Evaluation (Validation Set)

**After training completes, create a new cell:**
```python
!python eval.py \
    --data_path {DATA_ROOT}/val \
    --weights checkpoints/best_ep<TAB_COMPLETE>.pth \
    --save_visuals \
    --n_visuals 8
```

**Replace `<TAB_COMPLETE>`** with actual checkpoint name (use tab completion or `!ls checkpoints/`).

**Outputs:**
- `results/metrics_val.json`
- `results/confusion_matrix_val.png`
- `results/error_analysis_val.png`

---

### Step 8: Run Evaluation (Test Set)

**Create another cell:**
```python
!python eval.py \
    --data_path {DATA_ROOT}/test \
    --weights checkpoints/best_ep<TAB_COMPLETE>.pth \
    --save_visuals \
    --n_visuals 8
```

**Outputs:**
- `results/metrics_test.json` ← **Report these metrics in your PDF**
- `results/confusion_matrix_test.png`
- `results/error_analysis_test.png`

---

### Step 9: Download Outputs

**From Kaggle notebook:**
1. Open file browser (left sidebar)
2. Navigate to:
   - `checkpoints/` → download `best_model.pth`
   - `results/` → download all `.json` and `.png` files
3. Right-click → **Download**

**Or programmatically in a cell:**
```python
import shutil
shutil.make_archive('galaxeye_outputs', 'zip', '/kaggle/working', 'results')
shutil.move('galaxeye_outputs.zip', '/kaggle/working/')
# Click download icon next to galaxeye_outputs.zip in file browser
```

---

### Step 10: Package for Submission

**Locally (on your computer):**

1. **GitHub Repo:**
   ```bash
   # Create repo, add all code files
   git init
   git add dataset.py model.py losses.py utils.py train.py eval.py config.yaml requirements.txt README.md
   git commit -m "Initial submission"
   git remote add origin https://github.com/your-username/galaxeye-change-detection.git
   git push -u origin main
   ```

2. **Checkpoint (upload to Google Drive):**
   - Upload `best_model.pth` to Google Drive
   - Set sharing to "Anyone with link can view"
   - Copy link, paste into `README.md` under "Model Weights"

3. **ZIP file for form submission:**
   ```
   FirstName_LastName_GalaxEye.zip
   ├── best_model.pth             ← checkpoint file
   ├── technical_report.pdf        ← your written report
   └── time_resource_log.txt       ← time breakdown
   ```

---

## Workflow B — Local / Google Colab

### If Running Locally

```bash
# 1. Clone repo
git clone https://github.com/your-username/galaxeye-change-detection.git
cd galaxeye-change-detection

# 2. Setup environment
conda create -n galaxeye python=3.9 -y
conda activate galaxeye
pip install -r requirements.txt

# 3. Download dataset (place in data/)
# Follow HuggingFace download instructions

# 4. Train
python train.py --config config.yaml --data_root data/

# 5. Evaluate
python eval.py --data_path data/test --weights checkpoints/best_model.pth --save_visuals
```

### If Running on Colab

1. Upload all `.py` files + `config.yaml` to Colab session storage (file browser on left)
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Download dataset to `/content/data/` (use HuggingFace snippet)
4. Run training:
   ```python
   !python train.py --config config.yaml --data_root /content/data
   ```

---

## File Alignment Summary

| File             | Purpose                          | Where It Runs             |
|------------------|----------------------------------|---------------------------|
| `config.yaml`    | All hyperparameters              | Read by `train.py`, `eval.py` |
| `dataset.py`     | Dataset class, augmentation      | Imported by `train.py`, `eval.py` |
| `model.py`       | Neural network architecture      | Imported by `train.py`, `eval.py` |
| `losses.py`      | Loss functions                   | Imported by `train.py`    |
| `utils.py`       | Metrics, checkpoints, helpers    | Imported by all scripts   |
| `train.py`       | **RUN THIS** — training loop     | **Command:** `python train.py --config config.yaml --data_root <path>` |
| `eval.py`        | **RUN THIS** — evaluation script | **Command:** `python eval.py --data_path <path> --weights <ckpt>` |
| `requirements.txt` | Dependencies                   | **Once:** `pip install -r requirements.txt` |
| `README.md`      | Documentation for GitHub         | Reference only            |

**Nothing needs manual editing except:**
- `config.yaml` if you want to tune hyperparameters
- `README.md` Model Weights section (add your Google Drive link after upload)

---

## Quick Troubleshooting

### Error: "No training samples found"
**Fix:** Check `config.yaml` → `data.eo_variants`, `data.sar_variants`, `data.mask_variants` match your actual filenames. Run `!ls data/train/` in Kaggle to inspect.

### Error: "CUDA out of memory"
**Fix:** In `config.yaml`, reduce:
```yaml
train:
  batch_size: 8   # was 16
  patch_size: 128 # was 256
```

### Training too slow on CPU
**Fix:** Use Kaggle GPU (free) or Colab GPU. CPU training takes 10× longer.

### Checkpoint file not found
**Fix:** After training, run `!ls checkpoints/` to see actual filename, copy it exactly into `eval.py` command.

---

## Timeline Estimate (for Assignment Planning)

| Task                     | Time      |
|--------------------------|-----------|
| Setup environment        | 30 min    |
| Dataset download/upload  | 30 min    |
| Training (GPU)           | 3-4 hours |
| Validation evaluation    | 10 min    |
| Test evaluation          | 10 min    |
| Write technical report   | 4-6 hours |
| Package submission       | 1 hour    |
| **TOTAL**                | **10-12 hours** |

**Recommendation:** Start training first (it's the longest step), write report sections while it runs.

---

## What to Report in Your PDF

From the outputs, include in your technical report:

1. **Methodology section:**
   - Architecture diagram (draw CrossModalChangeNet from `model.py`)
   - Loss function formula (from `losses.py`)
   - Augmentation pipeline (from `config.yaml` → `augmentation`)

2. **Results section:**
   - Copy metrics from `results/metrics_val.json` and `results/metrics_test.json`
   - Embed confusion matrices: `confusion_matrix_val.png`, `confusion_matrix_test.png`
   - Include ≥5 qualitative examples from `error_analysis_test.png`

3. **Training curves:**
   - Embed `training_curves.png` (shows loss, F1, IoU over epochs)

4. **Time/Resource log:**
   - Total time: from `training_history.json` → sum epoch times
   - GPU model: Tesla T4 (if Kaggle), specify VRAM
   - Training time per epoch: ~2-3 min at BATCH_SIZE=16
   - Wall-clock total: ~3-4 hours

---

**You're now ready to train, evaluate, and submit!** 🚀
