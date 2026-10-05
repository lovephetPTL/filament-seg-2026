"""
Cell 7: Continue Training U-Net (ResNet34) จาก epoch 6-10
================================================================
รันต่อจาก cell 6b (v2) ที่มี train_loader / val_loader พร้อมแล้ว
ต้องอัปโหลด best_baseline_unet.pth (จากเมื่อวาน) เข้า Colab ก่อนรันเซลล์นี้
"""

!pip install -q segmentation-models-pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ กำลังใช้ Device: {device}")


# ---------------------------------------------------------
# 1. นิยาม Loss function
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 2. นิยามฟังก์ชันวัดผล (ต้องมาก่อนจุดที่เรียกใช้เสมอ)
# ---------------------------------------------------------
@torch.no_grad()
def dice_score(y_pred_logits, y_true, threshold=0.5, eps=1e-7):
    y_pred = (torch.sigmoid(y_pred_logits) > threshold).float()
    intersection = (y_pred * y_true).sum(dim=(1, 2, 3))
    union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


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


# ---------------------------------------------------------
# 3. สร้างโมเดล + โหลดน้ำหนักเดิมจากเมื่อวาน
# ---------------------------------------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load("best_baseline_unet.pth", map_location=device))
print("✅ โหลดน้ำหนักเดิมจากเมื่อวานสำเร็จ พร้อมเทรนต่อ")

criterion = CombinedBCEDiceLoss(bce_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)


# ---------------------------------------------------------
# 4. วัดค่า dice จริงของ checkpoint ก่อนเทรนต่อ (ตอนนี้ฟังก์ชันถูกนิยามแล้ว เรียกได้)
# ---------------------------------------------------------
print("📏 กำลังวัดค่า Val Dice จากไฟล์ Weight ที่โหลดมาจริง...")
_, real_val_dice = validate_one_epoch(model, val_loader, criterion, device)
print(f"👉 ค่า Val Dice จริงของไฟล์ปัจจุบัน: {real_val_dice:.4f}")
best_val_dice = real_val_dice


# ---------------------------------------------------------
# 5. เทรนต่อ epoch 6-10
# ---------------------------------------------------------
START_EPOCH = 6
EPOCHS = 10

print("🚀 เริ่มฝึกโมเดลต่อ...")
for epoch in range(START_EPOCH, EPOCHS + 1):
    train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | "
          f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
          f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_baseline_unet.pth")
        print("  💾 บันทึก Best Model Weight สำเร็จ!")

print("\n🎉 ฝึกโมเดลต่อเรียบร้อยแล้ว!")
