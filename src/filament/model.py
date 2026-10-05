"""U-Net with a ResNet34 encoder (segmentation_models_pytorch)."""
import torch
import segmentation_models_pytorch as smp


def build_model(pretrained_encoder=False):
    """pretrained_encoder=True  -> ImageNet weights (used to START training)
    pretrained_encoder=False -> random init, to be overwritten by our checkpoint
    """
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained_encoder else None,
        in_channels=3,
        classes=1,
    )


def load_checkpoint(path, device):
    model = build_model(pretrained_encoder=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
