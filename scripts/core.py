import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed")
MANIFEST_PATH = os.path.join(PROJECT_DIR, "manifest.json")


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_manifest(path=MANIFEST_PATH):
    with open(path) as f:
        return json.load(f)


def load_patient_volume(patient_id, processed_dir=PROCESSED_DIR):
    data = np.load(os.path.join(processed_dir, f"{patient_id}_data.npy"))
    seg = np.load(os.path.join(processed_dir, f"{patient_id}_seg.npy"))
    return data, seg


class BraTSDataset(Dataset):
    def __init__(self, patient_ids, processed_dir=PROCESSED_DIR):
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
        data, seg = load_patient_volume(pid, self.processed_dir)
        image = data[:, :, :, slice_idx]
        mask = (seg[:, :, slice_idx] > 0).astype(np.float32)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self, in_channels=4, out_channels=1, features=[32, 64, 128, 256], dropout_p=0.0
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.encoder.append(DoubleConv(in_channels, f, dropout_p=dropout_p))
            in_channels = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_p=dropout_p)

        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f, dropout_p=dropout_p))

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


def set_mc_dropout(model, enabled=True):
    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            module.train(enabled)
