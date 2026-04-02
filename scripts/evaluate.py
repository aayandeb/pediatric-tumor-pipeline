import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


class BraTSDataset(Dataset):
    def __init__(self, patient_ids, processed_dir):
        self.processed_dir = processed_dir
        self.slices = []

        for pid in tqdm(patient_ids, desc="Indexing"):
            data_path = os.path.join(processed_dir, f"{pid}_data.npy")
            if os.path.exists(data_path):
                n_slices = np.load(data_path, mmap_mode="r").shape[3]
                for s in range(n_slices):
                    self.slices.append((pid, s))

        print(f"Total slices: {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        pid, slice_idx = self.slices[idx]
        data = np.load(os.path.join(self.processed_dir, f"{pid}_data.npy"))
        seg = np.load(os.path.join(self.processed_dir, f"{pid}_seg.npy"))
        image = data[:, :, :, slice_idx]
        mask = (seg[:, :, slice_idx] > 0).astype(np.float32)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.encoder.append(DoubleConv(in_channels, f))
            in_channels = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            x = torch.cat([skips[i // 2], x], dim=1)
            x = self.decoder[i + 1](x)
        return self.final(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


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
        default=os.path.join(PROJECT_DIR, "checkpoints", "unet_best.pth"),
        help="Path to model checkpoint (state_dict).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 4).",
    )
    args = parser.parse_args()

    processed_dir = os.path.join(PROJECT_DIR, "processed")
    manifest_path = os.path.join(PROJECT_DIR, "manifest.json")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    patient_ids = manifest["split"][args.split]
    print(f"Split '{args.split}': {len(patient_ids)} patients")

    ds = BraTSDataset(patient_ids, processed_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError(
            "No slices found for this split. Ensure processed/*_data.npy files exist."
        )

    model = UNet().to(device)
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
