from torch import Tensor, nn


class AttentionBlock(nn.Module):
    """Attention block.

    :param embed_dim: int - dimensionality of input and attention feature vectors
    :param hidden_dim: int - dimensionality of hidden layer in feed-forward network
                 (usually 2-4x larger than embed_dim)
    :param num_heads: int - number of heads to use in the Multi-Head Attention block
    :param dropout: float - amount of dropout to apply in the feed-forward network
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        yln1 = self.layer_norm_1(x)
        # yln1 shape: (B, L, E)
        att_o, _ = self.attn(yln1, yln1, yln1)
        # att_o shape: (B, L, E)
        y1 = x + self.attn_dropout(att_o)
        # y1 shape: (B, L, E)

        yln2 = self.layer_norm_2(y1)
        # yln2 shape: (B, L, E)
        yl2 = self.linear(yln2)
        # yl2 shape: (B, L, E)

        y2 = y1 + yl2
        # y2 shape: (B, L, E)

        return y2
