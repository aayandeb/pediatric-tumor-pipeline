import csv
import json
import os

import numpy as np
import torch

try:
    from core import set_mc_dropout
except ImportError:
    from scripts.core import set_mc_dropout


def mc_dropout_predict(model, image_tensor, passes=20, threshold=0.5):
    model.eval()
    set_mc_dropout(model, enabled=True)

    preds = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(image_tensor)
            preds.append(torch.sigmoid(logits).squeeze().cpu().numpy())

    pred_stack = np.stack(preds, axis=0)
    mean_prob = pred_stack.mean(axis=0)
    var_map = pred_stack.var(axis=0)
    mask = (mean_prob > threshold).astype(np.uint8)
    return mean_prob, var_map, mask


def compute_gradcam(model, image_tensor, target_module):
    activations = []
    gradients = []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        score = torch.sigmoid(logits).mean()
        score.backward()

        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def bbox_3d(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return {
        "x_min": int(mins[0]),
        "y_min": int(mins[1]),
        "z_min": int(mins[2]),
        "x_max": int(maxs[0]),
        "y_max": int(maxs[1]),
        "z_max": int(maxs[2]),
    }


def compute_volume_metrics(mask_3d, spacing=(1.0, 1.0, 1.0)):
    voxel_count = int(mask_3d.sum())
    bbox = bbox_3d(mask_3d)
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    volume_mm3 = float(voxel_count * voxel_volume)
    positive_slices = int((mask_3d.sum(axis=(0, 1)) > 0).sum())
    mean_slice_area = float(mask_3d.sum(axis=(0, 1)).mean())

    if bbox is None:
        bbox_volume_vox = 0
    else:
        bbox_volume_vox = (
            (bbox["x_max"] - bbox["x_min"] + 1)
            * (bbox["y_max"] - bbox["y_min"] + 1)
            * (bbox["z_max"] - bbox["z_min"] + 1)
        )
    extent = float(voxel_count / bbox_volume_vox) if bbox_volume_vox > 0 else 0.0

    return {
        "voxel_count": voxel_count,
        "volume_mm3": volume_mm3,
        "positive_slices": positive_slices,
        "mean_slice_area_px": mean_slice_area,
        "extent": extent,
        "bbox": bbox,
        "spacing_mm": {"x": spacing[0], "y": spacing[1], "z": spacing[2]},
    }


def save_metrics(metrics_rows, json_path, csv_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics_rows, f, indent=2)

    fieldnames = [
        "patient_id",
        "pred_voxel_count",
        "pred_volume_mm3",
        "pred_positive_slices",
        "pred_mean_slice_area_px",
        "pred_extent",
        "gt_voxel_count",
        "gt_volume_mm3",
        "gt_positive_slices",
        "gt_mean_slice_area_px",
        "gt_extent",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(
                {
                    "patient_id": row["patient_id"],
                    "pred_voxel_count": row["prediction"]["voxel_count"],
                    "pred_volume_mm3": row["prediction"]["volume_mm3"],
                    "pred_positive_slices": row["prediction"]["positive_slices"],
                    "pred_mean_slice_area_px": row["prediction"]["mean_slice_area_px"],
                    "pred_extent": row["prediction"]["extent"],
                    "gt_voxel_count": row["ground_truth"]["voxel_count"],
                    "gt_volume_mm3": row["ground_truth"]["volume_mm3"],
                    "gt_positive_slices": row["ground_truth"]["positive_slices"],
                    "gt_mean_slice_area_px": row["ground_truth"]["mean_slice_area_px"],
                    "gt_extent": row["ground_truth"]["extent"],
                }
            )
