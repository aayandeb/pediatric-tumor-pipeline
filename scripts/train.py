import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── Dataset ──────────────────────────────────────────────────────────────────
class BraTSDataset(Dataset):
    def __init__(self, patient_ids, processed_dir):
        self.processed_dir = processed_dir
        self.slices = []

        for pid in tqdm(patient_ids, desc="Indexing"):
            data_path = os.path.join(processed_dir, f"{pid}_data.npy")
            if os.path.exists(data_path):
                n_slices = np.load(data_path, mmap_mode='r').shape[3]
                for s in range(n_slices):
                    self.slices.append((pid, s))

        print(f"Total slices: {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        pid, slice_idx = self.slices[idx]
        data = np.load(os.path.join(self.processed_dir, f"{pid}_data.npy"))
        seg  = np.load(os.path.join(self.processed_dir, f"{pid}_seg.npy"))
        image = data[:, :, :, slice_idx]
        mask  = (seg[:, :, slice_idx] > 0).astype(np.float32)
        return (torch.tensor(image, dtype=torch.float32),
                torch.tensor(mask,  dtype=torch.float32).unsqueeze(0))

# ── U-Net ─────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder   = nn.ModuleList()
        self.decoder   = nn.ModuleList()
        self.pool      = nn.MaxPool2d(2, 2)

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
            x = layer(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            x = torch.cat([skips[i // 2], x], dim=1)
            x = self.decoder[i + 1](x)
        return self.final(x)

# ── Loss & Metrics ────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred   = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed")
    MANIFEST_PATH = os.path.join(PROJECT_DIR, "manifest.json")
    EPOCHS        = 10
    BATCH_SIZE    = 4
    LR            = 1e-4

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    train_ids = manifest["split"]["train"]
    val_ids   = manifest["split"]["val"]
    print(f"Manifest: {len(train_ids)} train, {len(val_ids)} val")

    train_ds = BraTSDataset(train_ids, PROCESSED_DIR)
    val_ds   = BraTSDataset(val_ids,   PROCESSED_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceLoss()

    best_dice = 0.0
    ckpt_dir  = os.path.join(PROJECT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = val_dice = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                images, masks = images.to(device), masks.to(device)
                preds     = model(images)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks)

        avg_train = train_loss / len(train_loader)
        avg_vloss = val_loss / len(val_loader)
        avg_dice  = val_dice / len(val_loader)

        print(f"\nEpoch {epoch+1}: "
              f"Train Loss={avg_train:.4f} | "
              f"Val Loss={avg_vloss:.4f} | "
              f"Val Dice={avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_path = os.path.join(ckpt_dir, "unet_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved (Dice={best_dice:.4f})")
        print()

    final_path = os.path.join(ckpt_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Best Val Dice={best_dice:.4f}")
    print(f"  Best checkpoint: {best_path}")
    print(f"  Final checkpoint: {final_path}")