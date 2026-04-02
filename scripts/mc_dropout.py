import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import PROCESSED_DIR, UNet, get_device, load_patient_volume
from inference_utils import mc_dropout_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty for one slice.")
    parser.add_argument("--patient-id", required=True, help="BraTS patient ID.")
    parser.add_argument(
        "--slice-idx", type=int, default=None, help="Axial slice index (default: middle)."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(PROCESSED_DIR), "checkpoints", "unet_best.pth"),
        help="Path to checkpoint.",
    )
    parser.add_argument("--passes", type=int, default=20, help="MC passes.")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for final mask."
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.2,
        help="Dropout probability used during stochastic inference.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(PROCESSED_DIR), "images", "uncertainty"),
        help="Output directory for uncertainty artifacts.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data, seg = load_patient_volume(args.patient_id, PROCESSED_DIR)
    depth = data.shape[3]
    slice_idx = args.slice_idx if args.slice_idx is not None else depth // 2
    if slice_idx < 0 or slice_idx >= depth:
        raise ValueError(f"slice-idx must be in [0, {depth - 1}]")

    image = data[:, :, :, slice_idx]
    gt = (seg[:, :, slice_idx] > 0).astype(np.uint8)

    device = get_device()
    model = UNet(dropout_p=args.dropout_p).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    image_tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
    mean_prob, var_map, pred_mask = mc_dropout_predict(
        model, image_tensor, passes=args.passes, threshold=args.threshold
    )

    stem = f"{args.patient_id}_slice{slice_idx}"
    np.save(os.path.join(args.out_dir, f"{stem}_mean_prob.npy"), mean_prob)
    np.save(os.path.join(args.out_dir, f"{stem}_var_map.npy"), var_map)
    np.save(os.path.join(args.out_dir, f"{stem}_pred_mask.npy"), pred_mask)

    flair = image[3]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(flair.T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")
    axes[1].imshow(gt.T, cmap="Greens", origin="lower")
    axes[1].set_title("GT Mask")
    axes[2].imshow(mean_prob.T, cmap="magma", origin="lower")
    axes[2].set_title("Mean Probability")
    axes[3].imshow(var_map.T, cmap="viridis", origin="lower")
    axes[3].set_title("Uncertainty (Variance)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{args.patient_id} | slice={slice_idx} | passes={args.passes}")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"{stem}_uncertainty.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
