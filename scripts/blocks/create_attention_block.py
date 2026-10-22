import torch

from vision_transformers_cifar.blocks import AttentionBlock

if __name__ == "__main__":
    att_block = AttentionBlock(
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
    )
    B, L, E = (2, 16, 32)
    x = torch.randn(size=(B, L, E))

    o = att_block(x)
    print(f"input shape: {x.shape}, output shape: {o.shape}")
