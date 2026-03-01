## Pediatric Brain Tumor MRI Pipeline

This repository contains a small end‑to‑end preprocessing pipeline for **pediatric brain tumor MRI** volumes (BraTS‑style data with multi‑modal MR and segmentation masks).  
It focuses on:
- **Quick visual sanity checks** of raw NIfTI volumes.
- **Standardized preprocessing** (z‑score normalization, axial slice cropping).
- **Saving NumPy tensors** and a **train/val/test manifest** for downstream modeling.

### Dataset assumptions

The code expects a BraTS‑like directory layout under `raw/`:

- **Root directory**: `raw/`
- **Per‑patient folder**: `raw/BraTS2021_XXXXX/`
- **Files per patient**:
  - `BraTS2021_XXXXX_t1.nii.gz`
  - `BraTS2021_XXXXX_t1ce.nii.gz`
  - `BraTS2021_XXXXX_t2.nii.gz`
  - `BraTS2021_XXXXX_flair.nii.gz`
  - `BraTS2021_XXXXX_seg.nii.gz`

You can either download BraTS 2021 manually and arrange it in this format, or adapt the (currently empty) `scripts/download_data.py` to automate it.

### Repository structure

- `scripts/explore_data.py` – load a single patient, print basic stats, and save a 2D montage (`first_look.png`) across modalities + segmentation.
- `scripts/pipeline.py` – main preprocessing script:
  - Z‑score normalization per modality (ignoring background zeros).
  - Keep only the **middle 60% of axial slices** (removes mostly empty edges).
  - Save stacked multi‑modal volumes and segmentation masks as `.npy`.
  - Create a **70/15/15 train/val/test split** and persist it in `manifest.json`.
- `scripts/download_data.py` – placeholder for data download logic.
- `raw/` – expected location of unprocessed NIfTI volumes.
- `processed/` – generated NumPy arrays after running the pipeline.
- `first_look.png` – example visualization from `explore_data.py`.

### Installation

You will need Python 3 and a few scientific Python libraries. A minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
pip install nibabel numpy matplotlib tqdm scikit-learn
```

### Quick data exploration

To visually inspect a single case and confirm paths are correct:

```bash
python scripts/explore_data.py
```

This will:
- Load `BraTS2021_00000` from `raw/` (see the script for the patient ID).
- Print each modality’s shape and intensity range.
- Save `first_look.png` at the project root with T1, T1ce, T2, FLAIR, and segmentation side‑by‑side.

### Preprocessing pipeline

Run the main preprocessing script:

```bash
python scripts/pipeline.py
```

What it does:
- **Reads** all patient folders under `raw/` whose names start with `BraTS2021_`.
- For each patient:
  - Loads `t1`, `t1ce`, `t2`, `flair`, and `seg` volumes with `nibabel`.
  - Applies **z‑score normalization** per modality over non‑zero voxels.
  - Stacks modalities into a tensor of shape `(4, 240, 240, N_slices)` and keeps only the middle 60% of axial slices.
  - Crops the segmentation volume to the same slice range.
  - Saves:
    - `processed/BraTS2021_XXXXX_data.npy` – float32 tensor of stacked modalities.
    - `processed/BraTS2021_XXXXX_seg.npy` – uint8 segmentation labels.
- After processing, it:
  - Splits all patient IDs into **train/val/test = 70/15/15** using a fixed random seed.
  - Writes a `manifest.json` file at the project root.

The manifest has the following structure:

```json
{
  "dataset": "BraTS2021",
  "total_patients": <int>,
  "modalities": ["t1", "t1ce", "t2", "flair"],
  "split": {
    "train": ["BraTS2021_XXXXX", "..."],
    "val":   ["BraTS2021_YYYYY", "..."],
    "test":  ["BraTS2021_ZZZZZ", "..."]
  }
}
```

### Output shapes and normalization

- Each `_data.npy` file:
  - Shape: `(4, 240, 240, N_mid_slices)`
  - Channels (axis 0): `[t1, t1ce, t2, flair]`
  - Intensities: approximately **zero mean, unit variance** where the original volume was non‑zero.
- Each `_seg.npy` file:
  - Shape: `(240, 240, N_mid_slices)`
  - Integer labels (e.g. 0–4 for background and tumor subregions, following BraTS conventions).

### Next steps / ideas

- Add unit tests for preprocessing and shape invariants.
- Implement `scripts/download_data.py` to fetch and organize BraTS data.
- Extend the pipeline to handle **pediatric‑specific cohorts**, class imbalance, or additional imaging modalities as needed.