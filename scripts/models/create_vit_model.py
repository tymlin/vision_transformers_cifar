import torch

from vision_transformers_cifar.models import VisionTransformer

if __name__ == "__main__":
    vit = VisionTransformer(
        embed_dim=256,
        hidden_dim=512,
        num_channels=3,
        num_heads=8,
        num_layers=6,
        num_classes=10,
        patch_size=4,
        num_patches=64,
        dropout=0.2,
    )
    B, C, H, W = (2, 3, 32, 32)
    x = torch.randn(size=(B, C, H, W))

    o = vit(x)
    print(f"input shape: {x.shape}, output shape: {o.shape}")
