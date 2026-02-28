import nibabel as nib
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

RAW_DIR = "../raw"
OUT_DIR = "../processed"
os.makedirs(OUT_DIR, exist_ok=True)

MODALITIES = ["t1", "t1ce", "t2", "flair"]

def zscore_normalize(volume):
    """Normalize to zero mean, unit std — ignoring background (zeros)."""
    mask = volume > 0
    mean = volume[mask].mean()
    std  = volume[mask].std()
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean) / (std + 1e-8)
    return normalized

def get_middle_slices(volume, fraction=0.6):
    """Keep only the middle 60% of axial slices — edges are mostly empty."""
    total = volume.shape[2]
    start = int(total * 0.2)
    end   = int(total * 0.8)
    return start, end

def process_patient(patient_id):
    patient_path = os.path.join(RAW_DIR, patient_id)
    
    volumes = {}
    for mod in MODALITIES:
        path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
        volumes[mod] = nib.load(path).get_fdata().astype(np.float32)
    
    seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    
    # Normalize each modality
    for mod in MODALITIES:
        volumes[mod] = zscore_normalize(volumes[mod])
    
    # Stack modalities into shape (4, 240, 240, 155)
    stacked = np.stack([volumes[mod] for mod in MODALITIES], axis=0)
    
    # Get middle slice range
    start, end = get_middle_slices(stacked)
    stacked = stacked[:, :, :, start:end]
    seg     = seg[:, :, start:end]
    
    # Save
    np.save(os.path.join(OUT_DIR, f"{patient_id}_data.npy"), stacked)
    np.save(os.path.join(OUT_DIR, f"{patient_id}_seg.npy"),  seg)
    
    return patient_id

# Get all patient IDs
all_patients = sorted([
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("BraTS2021_")
])

print(f"Found {len(all_patients)} patients")

# Process first 5 patients as a test
print("Processing first 5 patients as test run...")
for pid in tqdm(all_patients[:5]):
    process_patient(pid)

print("Done! Checking output...")
sample = np.load(os.path.join(OUT_DIR, f"{all_patients[0]}_data.npy"))
print(f"Sample shape: {sample.shape}")
print(f"Sample min: {sample.min():.3f}, max: {sample.max():.3f}")
print(f"Mean ~0: {sample[sample != 0].mean():.3f}")

# Split all patients 70/15/15
train, temp   = train_test_split(all_patients, test_size=0.30, random_state=42)
val,   test   = train_test_split(temp,         test_size=0.50, random_state=42)

print(f"\nSplit: {len(train)} train | {len(val)} val | {len(test)} test")

# Save manifest
manifest = {
    "dataset": "BraTS2021",
    "total_patients": len(all_patients),
    "modalities": MODALITIES,
    "split": {
        "train": train,
        "val":   val,
        "test":  test
    }
}

with open("../manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Saved manifest.json")