"""
Cell 7: Train U-Net (ResNet34) + BCE-Dice Loss + Dice Score Tracking
========================================================================
รันต่อจาก cell 6b (v2) ที่มี train_loader / val_loader พร้อมแล้ว
"""

!pip install -q segmentation-models-pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm

torch.manual_seed(42)  # ให้ผลรันซ้ำได้ใกล้เคียงเดิม

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ กำลังใช้ Device: {device}")


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)


@torch.no_grad()
def dice_score(y_pred_logits, y_true, threshold=0.5, eps=1e-7):
    """Dice score (ไม่ใช่ loss) เอาไว้ดูว่าโมเดล 'แม่น' แค่ไหนจริงๆ อ่านง่ายกว่าค่า loss"""
    y_pred = (torch.sigmoid(y_pred_logits) > threshold).float()
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

criterion = CombinedBCEDiceLoss(bce_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_dice += dice_score(outputs, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            running_dice += dice_score(outputs, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n


EPOCHS = 5
best_val_loss = float('inf')

print("🚀 เริ่มฝึกโมเดล Baseline U-Net (ResNet34)...")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | "
          f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
          f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_baseline_unet.pth")
        print("  💾 บันทึก Best Model Weight สำเร็จ!")

print("\n🎉 ฝึกโมเดล Baseline เรียบร้อยแล้ว!")
