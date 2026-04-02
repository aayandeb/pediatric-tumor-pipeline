import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import (
    BraTSDataset,
    DiceLoss,
    MANIFEST_PATH,
    PROJECT_DIR,
    PROCESSED_DIR,
    UNet,
    dice_score,
    get_device,
    load_manifest,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2D U-Net checkpoint.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which manifest split to evaluate (default: test).",
    )
    parser.add_argument(
        "--checkpoint",
        default=f"{PROJECT_DIR}/checkpoints/unet_best.pth",
        help="Path to model checkpoint (state_dict).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    manifest = load_manifest(MANIFEST_PATH)
    patient_ids = manifest["split"][args.split]
    print(f"Split '{args.split}': {len(patient_ids)} patients")

    ds = BraTSDataset(patient_ids, PROCESSED_DIR)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError(
            "No slices found for this split. Ensure processed/*_data.npy files exist."
        )

    model = UNet(dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    criterion = DiceLoss()

    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating {args.split}"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            total_loss += criterion(preds, masks).item()
            total_dice += dice_score(preds, masks)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    print(f"{args.split.upper()} Loss={avg_loss:.4f} | Dice={avg_dice:.4f}")
