import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

patient_id = "BraTS2021_00000"
base_path = f"../raw/{patient_id}"

modalities = {
    "t1":    f"{base_path}/{patient_id}_t1.nii.gz",
    "t1ce":  f"{base_path}/{patient_id}_t1ce.nii.gz",
    "t2":    f"{base_path}/{patient_id}_t2.nii.gz",
    "flair": f"{base_path}/{patient_id}_flair.nii.gz",
    "seg":   f"{base_path}/{patient_id}_seg.nii.gz",
}

data = {}
for mod, path in modalities.items():
    img = nib.load(path)
    data[mod] = img.get_fdata()
    print(f"{mod}: shape={data[mod].shape}, min={data[mod].min():.1f}, max={data[mod].max():.1f}")

slice_idx = data["t1"].shape[2] // 2

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
titles = ["T1", "T1ce", "T2", "FLAIR", "Segmentation"]
keys   = ["t1", "t1ce", "t2", "flair", "seg"]

for ax, key, title in zip(axes, keys, titles):
    ax.imshow(data[key][:, :, slice_idx].T, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(f"Patient: {patient_id} | Axial slice: {slice_idx}", fontsize=13)
plt.tight_layout()
plt.savefig("../first_look.png", dpi=150)
plt.show()
print("Saved first_look.png")