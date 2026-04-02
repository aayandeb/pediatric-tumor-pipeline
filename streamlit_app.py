import os

import numpy as np
import streamlit as st
import torch

from scripts.core import PROCESSED_DIR, UNet, get_device, load_manifest, load_patient_volume
from scripts.inference_utils import compute_gradcam, compute_volume_metrics, mc_dropout_predict


@st.cache_resource
def load_model(checkpoint_path):
    device = get_device()
    model = UNet(dropout_p=0.2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


st.set_page_config(page_title="Pediatric Tumor Explorer", layout="wide")
st.title("Pediatric Tumor Pipeline - Single Case Explorer")

manifest = load_manifest()
default_ckpt = os.path.join(os.path.dirname(PROCESSED_DIR), "checkpoints", "unet_best.pth")
checkpoint_path = st.sidebar.text_input("Checkpoint path", value=default_ckpt)
split = st.sidebar.selectbox("Split", ["train", "val", "test"], index=2)
patient_id = st.sidebar.selectbox("Patient ID", manifest["split"][split])
mc_passes = st.sidebar.slider("MC Dropout passes", min_value=5, max_value=50, value=20)
threshold = st.sidebar.slider("Mask threshold", min_value=0.1, max_value=0.9, value=0.5)

model, device = load_model(checkpoint_path)
data, seg = load_patient_volume(patient_id, PROCESSED_DIR)
slice_idx = st.sidebar.slider("Slice index", 0, data.shape[3] - 1, data.shape[3] // 2)
image = data[:, :, :, slice_idx]
gt_mask = (seg[:, :, slice_idx] > 0).astype(np.uint8)
image_tensor = torch.tensor(image[None, ...], dtype=torch.float32, device=device)

mean_prob, var_map, pred_mask = mc_dropout_predict(
    model, image_tensor, passes=mc_passes, threshold=threshold
)
cam = compute_gradcam(model, image_tensor, target_module=model.bottleneck.block[3])

pred_volume_mask = np.zeros_like(seg, dtype=np.uint8)
pred_volume_mask[:, :, slice_idx] = pred_mask
gt_volume_mask = (seg > 0).astype(np.uint8)
pred_metrics = compute_volume_metrics(pred_volume_mask)
gt_metrics = compute_volume_metrics(gt_volume_mask)

flair = image[3]
t1ce = image[1]

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("T1ce")
    st.image(t1ce.T, clamp=True)
    st.subheader("FLAIR")
    st.image(flair.T, clamp=True)
with col2:
    st.subheader("Segmentation Overlay (GT green, Pred red)")
    overlay = np.stack([pred_mask, gt_mask, np.zeros_like(gt_mask)], axis=-1).astype(np.float32)
    st.image((overlay.transpose(1, 0, 2) * 255).astype(np.uint8))
    st.subheader("Mean Probability")
    st.image(mean_prob.T, clamp=True)
with col3:
    st.subheader("Uncertainty (Variance)")
    st.image(var_map.T, clamp=True)
    st.subheader("Grad-CAM")
    st.image(cam.T, clamp=True)

st.markdown("### Quantification")
q1, q2 = st.columns(2)
with q1:
    st.write("Prediction (selected slice volume proxy)")
    st.json(pred_metrics)
with q2:
    st.write("Ground Truth (full patient)")
    st.json(gt_metrics)
