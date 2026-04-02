import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from core import (
    PROCESSED_DIR,
    UNet,
    get_device,
    load_manifest,
    load_patient_volume,
)
from inference_utils import compute_volume_metrics, save_metrics


def predict_patient_mask(model, data, device, threshold=0.5):
    pred_slices = []
    with torch.no_grad():
        for s in range(data.shape[3]):
            image = data[:, :, :, s]
            image_tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
            probs = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()
            pred_slices.append((probs > threshold).astype(np.uint8))
    return np.stack(pred_slices, axis=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tumor quantification from predictions.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Manifest split to process.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Optional single patient override.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(PROCESSED_DIR), "checkpoints", "unet_best.pth"),
        help="Path to checkpoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spacing-x", type=float, default=1.0)
    parser.add_argument("--spacing-y", type=float, default=1.0)
    parser.add_argument("--spacing-z", type=float, default=1.0)
    parser.add_argument(
        "--out-json",
        default=os.path.join(
            os.path.dirname(PROCESSED_DIR), "experiments", "quantification_metrics.json"
        ),
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(
            os.path.dirname(PROCESSED_DIR), "experiments", "quantification_metrics.csv"
        ),
    )
    args = parser.parse_args()

    manifest = load_manifest()
    if args.patient_id:
        patient_ids = [args.patient_id]
    else:
        patient_ids = manifest["split"][args.split]

    spacing = (args.spacing_x, args.spacing_y, args.spacing_z)
    device = get_device()
    model = UNet(dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    for pid in tqdm(patient_ids, desc="Quantifying"):
        data, seg = load_patient_volume(pid, PROCESSED_DIR)
        gt = (seg > 0).astype(np.uint8)
        pred = predict_patient_mask(model, data, device, threshold=args.threshold)

        rows.append(
            {
                "patient_id": pid,
                "prediction": compute_volume_metrics(pred, spacing=spacing),
                "ground_truth": compute_volume_metrics(gt, spacing=spacing),
            }
        )

    save_metrics(rows, args.out_json, args.out_csv)
    print(f"Saved metrics JSON: {args.out_json}")
    print(f"Saved metrics CSV:  {args.out_csv}")
