import torch
from torch import nn

from vision_transformers_cifar.blocks import AttentionBlock
from vision_transformers_cifar.models.ops_utils import img_to_patch


class VisionTransformer(nn.Module):
    """Vision Transformer.

    embed_dim: Dimensionality of the input feature vectors to the Transformer
    hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                 within the Transformer
    num_channels: Number of channels of the input (3 for RGB)
    num_heads: Number of heads to use in the Multi-Head Attention block
    num_transformer_layers: Number of layers to use in the Transformer
    num_classes: Number of classes to predict
    patch_size: Number of pixels that the patches have per dimension
    num_patches: Maximum number of patches an image can have
    dropout: Amount of dropout to apply in the feed-forward network and
              on the input encoding
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

        # ------------------------------------------- Layers/Networks --------------------------------------------------

        # A linear projection layer that maps the input patches to a feature vector of larger size. It is implemented
        # by a simple linear layer that takes each patch independently as input.

        # input tensor shape is (B, num of patches, C, patch_size, patch_size) or last three dimension combined
        input_tensor_feature_size = num_channels * patch_size * patch_size
        self.input_layer = nn.Linear(input_tensor_feature_size, embed_dim)

        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers))
        )

        # An MLP head that takes the output feature vector of the CLS token, and maps it to a classification prediction.
        # This is usually implemented by a small feed-forward network or even a single linear layer.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------- Parameters/Embeddings -----------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # num of patches + one class token
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess input
        patches = img_to_patch(
            x,
            self.patch_size,
        )
        B, T, _ = patches.shape
        x = self.input_layer(patches)

        # Add class token
        cls_token = self.cls_token.repeat(B, 1, 1)  # works similar to np.tile
        # shape (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        out = self.mlp_head(cls)

        return out
