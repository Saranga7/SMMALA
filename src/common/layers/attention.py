import torch
from torch import nn
from torch import Tensor
from einops import rearrange


class Attention(nn.Module):
    """
    Attention mechanism computing attention on a CLS token.
    It considers the CLS token as a sequence chunks, and performs attention at the chunks level.
    Each head will be able to attend to different aspects of the feature space, allowing for more complex feature interactions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, C = x.shape

        # Apply qkv transformation
        qkv = self.qkv(x)

        # Reshape to separate Q, K, V and split heads
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention and combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
