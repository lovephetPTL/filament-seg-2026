"""Loss, tile-level Dice, and a resumable training loop.

The submitted checkpoint was produced in THREE consecutive sessions (see README):
  1) 5 epochs from ImageNet weights (lr 3e-4), checkpoint chosen by validation loss
  2) 5 epochs resumed from the best checkpoint of (1) (lr 1e-4), chosen by validation loss
  3) 8 epochs resumed from the best checkpoint of (2), chosen by validation Dice (lr 1e-4)
Every session builds a fresh AdamW optimiser (optimiser state is not stored).
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, y_pred, y_true):
        return self.bce_weight * self.bce(y_pred, y_true) + (1 - self.bce_weight) * self.dice(y_pred, y_true)


@torch.no_grad()
def dice_score(logits, target, threshold=0.5, eps=1e-7):
    """Tile-level Dice. A tile that is empty in GT and prediction scores 1.0, so this
    number is higher than the filament-only Dice and than PQ (see README)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def _run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    loss_sum = dice_sum = 0.0
    with torch.set_grad_enabled(train):
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images, masks = images.to(device), masks.to(device)
            if train:
                optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, masks)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.size(0)
            dice_sum += dice_score(out, masks) * images.size(0)
    n = len(loader.dataset)
    return loss_sum / n, dice_sum / n


def fit(model, train_loader, val_loader, device, start_epoch, end_epoch, lr,
        select_by="dice", save_path="best_baseline_unet.pth", weight_decay=1e-2, seed=42,
        initial_best_loss=float("inf")):
    """Train epochs start_epoch..end_epoch (inclusive) and keep the best checkpoint.

    select_by="loss": save when validation loss improves (sessions 1-2)
    select_by="dice": save when validation Dice improves (session 3). The Dice of the
                      loaded weights is measured first and used as the bar to beat.
    """
    torch.manual_seed(seed)
    criterion = CombinedBCEDiceLoss(0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_dice = initial_best_loss, -1.0
    if select_by == "dice":
        _, best_dice = _run_epoch(model, val_loader, criterion, device, desc="baseline val")
        print(f"validation Dice of the loaded weights: {best_dice:.4f}")
    for epoch in range(start_epoch, end_epoch + 1):
        tl, td = _run_epoch(model, train_loader, criterion, device, optimizer, "train")
        vl, vd = _run_epoch(model, val_loader, criterion, device, None, "val")
        print(f"Epoch {epoch:02d}/{end_epoch:02d} | train loss {tl:.4f} dice {td:.4f} | "
              f"val loss {vl:.4f} dice {vd:.4f}")
        better = (vl < best_loss) if select_by == "loss" else (vd > best_dice)
        if better:
            best_loss, best_dice = min(best_loss, vl), max(best_dice, vd)
            torch.save(model.state_dict(), save_path)
            print("  saved best checkpoint")
