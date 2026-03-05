import argparse
import nibabel as nib
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_DIR, "raw")
OUT_DIR = os.path.join(PROJECT_DIR, "processed")
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

    for mod in MODALITIES:
        volumes[mod] = zscore_normalize(volumes[mod])

    stacked = np.stack([volumes[mod] for mod in MODALITIES], axis=0)

    start, end = get_middle_slices(stacked)
    stacked = stacked[:, :, :, start:end]
    seg     = seg[:, :, start:end]

    np.save(os.path.join(OUT_DIR, f"{patient_id}_data.npy"), stacked)
    np.save(os.path.join(OUT_DIR, f"{patient_id}_seg.npy"),  seg)

    return patient_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BraTS2021 patients")
    parser.add_argument("-n", "--num-patients", type=int, default=None,
                        help="Number of patients to process (default: all)")
    args = parser.parse_args()

    all_patients = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("BraTS2021_")
    ])
    print(f"Found {len(all_patients)} patients total")

    to_process = all_patients[:args.num_patients] if args.num_patients else all_patients

    already_done = {
        f.replace("_data.npy", "")
        for f in os.listdir(OUT_DIR)
        if f.endswith("_data.npy")
    }
    remaining = [p for p in to_process if p not in already_done]
    print(f"Processing {len(remaining)} new patients ({len(already_done)} already done, "
          f"{len(to_process)} requested)")

    for pid in tqdm(remaining, desc="Processing"):
        process_patient(pid)

    # Verify a sample
    sample_pid = to_process[0]
    sample = np.load(os.path.join(OUT_DIR, f"{sample_pid}_data.npy"))
    print(f"\nSample check ({sample_pid}): shape={sample.shape}, "
          f"min={sample.min():.3f}, max={sample.max():.3f}, "
          f"mean(nonzero)={sample[sample != 0].mean():.3f}")

    # 70/15/15 split over the selected patients
    train, temp = train_test_split(to_process, test_size=0.30, random_state=42)
    val,   test = train_test_split(temp,       test_size=0.50, random_state=42)

    print(f"Split: {len(train)} train | {len(val)} val | {len(test)} test")

    manifest = {
        "dataset": "BraTS2021",
        "total_patients": len(all_patients),
        "patients_processed": len(to_process),
        "modalities": MODALITIES,
        "split": {
            "train": train,
            "val":   val,
            "test":  test
        }
    }

    manifest_path = os.path.join(PROJECT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {manifest_path}")
