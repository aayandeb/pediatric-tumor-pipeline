# Roadmap MVP Implementation (2026-04-02)

This note captures smoke-validation outputs for the first MVP pass of the remaining roadmap modules.

## Implemented modules

- MC Dropout uncertainty estimation (`scripts/mc_dropout.py`)
- Grad-CAM explainability overlays (`scripts/gradcam.py`)
- Tumor quantification export (`scripts/quantify.py`)
- Streamlit single-case explorer (`streamlit_app.py`)

## Smoke test commands and outcomes

- `python3 scripts/evaluate.py --split test --checkpoint checkpoints/unet_best.pth`
  - `TEST Loss=0.5274 | Dice=0.8907`
- `python3 scripts/mc_dropout.py --patient-id BraTS2021_00000 --slice-idx 53 --passes 10`
  - Saved: `images/uncertainty/BraTS2021_00000_slice53_uncertainty.png`
- `python3 scripts/gradcam.py --patient-id BraTS2021_00000 --slice-idx 53`
  - Saved: `images/gradcam/BraTS2021_00000_slice53_gradcam.png`
- `python3 scripts/quantify.py --patient-id BraTS2021_00000`
  - Saved:
    - `experiments/quantification_metrics.json`
    - `experiments/quantification_metrics.csv`
- `streamlit run streamlit_app.py --server.headless true --server.port 8503`
  - App startup successful (local URL printed, no import/runtime crash during boot).

## Example quantification output (single patient)

For `BraTS2021_00000` with spacing defaults `(1.0, 1.0, 1.0)`:

- Prediction volume: `57171.0 mm^3`
- Ground-truth volume: `57305.0 mm^3`
- Prediction bbox: `(x=111..164, y=44..121, z=3..42)`
- Ground-truth bbox: `(x=118..163, y=45..120, z=1..47)`

## Caveats in this MVP

- Quantification currently assumes unit voxel spacing unless overridden by CLI flags.
- Streamlit quantification panel compares full GT volume against the currently selected predicted slice proxy.
- Grad-CAM uses a simple segmentation target (`mean(sigmoid(logits))`) for interpretability; this is intended as a practical first-pass visualization.
