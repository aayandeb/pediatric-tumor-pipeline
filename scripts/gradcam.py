import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import PROCESSED_DIR, UNet, get_device, load_patient_volume
from inference_utils import compute_gradcam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM overlay for one slice.")
    parser.add_argument("--patient-id", required=True, help="BraTS patient ID.")
    parser.add_argument(
        "--slice-idx", type=int, default=None, help="Axial slice index (default: middle)."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(PROCESSED_DIR), "checkpoints", "unet_best.pth"),
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(PROCESSED_DIR), "images", "gradcam"),
        help="Output directory for Grad-CAM images.",
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
    model = UNet(dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image_tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)
    cam = compute_gradcam(model, image_tensor, target_module=model.bottleneck.block[3])
    logits = model(image_tensor)
    pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8)

    stem = f"{args.patient_id}_slice{slice_idx}"
    np.save(os.path.join(args.out_dir, f"{stem}_gradcam.npy"), cam)

    flair = image[3]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(flair.T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")
    axes[1].imshow(gt.T, cmap="Greens", origin="lower")
    axes[1].imshow(pred_mask.T, cmap="Reds", origin="lower", alpha=0.45)
    axes[1].set_title("GT+Pred")
    axes[2].imshow(cam.T, cmap="jet", origin="lower")
    axes[2].set_title("Grad-CAM")
    axes[3].imshow(flair.T, cmap="gray", origin="lower")
    axes[3].imshow(cam.T, cmap="jet", origin="lower", alpha=0.45)
    axes[3].set_title("FLAIR + Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{args.patient_id} | slice={slice_idx}")
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"{stem}_gradcam.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
