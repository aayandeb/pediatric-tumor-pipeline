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

device = get_device()
print(f"Using device: {device}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    EPOCHS        = 10
    BATCH_SIZE    = 4
    LR            = 1e-4

    manifest = load_manifest(MANIFEST_PATH)

    train_ids = manifest["split"]["train"]
    val_ids   = manifest["split"]["val"]
    print(f"Manifest: {len(train_ids)} train, {len(val_ids)} val")

    train_ds = BraTSDataset(train_ids, PROCESSED_DIR)
    val_ds   = BraTSDataset(val_ids,   PROCESSED_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = UNet(dropout_p=0.0).to(device)
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