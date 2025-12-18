import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm


class AttentionPooling(nn.Module):
    """
    Applies attention to the patches extracted from the trunk with the CLS token.

    Args:
        dimensions: The dimension of the whole architecture.
        num_heads: The number of attention heads.
        dim_feedforward: The dimension of the feedforward network model.
        dropout: The dropout rate.
        activation: The activation function.
        layer_norm_eps: The epsilon value for layer normalization.
        batch_first: The flag to indicate if the batch size is the first dimension.
        device: The device to run the model.
        dtype: The data type to run the model.

    Inputs:
        Flattened patches from the trunk.

    Outputs:
        cls_token: The modified CLS token.
        attn_weights: The attention weights.
    """

    def __init__(
        self,
        dimensions,
        num_heads=1,
        dim_feedforward=None,
        dropout=0.2,
        activation=nn.GELU(),
        layer_norm_eps=1e-5,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.dim_feedforward = dimensions * 4 if dim_feedforward is None else dim_feedforward
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimensions))

        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = MultiheadAttention(
            embed_dim=dimensions, num_heads=num_heads, batch_first=True, dropout=dropout, **factory_kwargs
        )
        self.layer_norm1 = LayerNorm(dimensions, eps=layer_norm_eps, **factory_kwargs)
        self.layer_norm2 = LayerNorm(dimensions, eps=layer_norm_eps, **factory_kwargs)
        self.layer_norm3 = LayerNorm(dimensions, eps=layer_norm_eps, **factory_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(dimensions, self.dim_feedforward, **factory_kwargs),
            activation,
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, dimensions, **factory_kwargs),
            activation,
        )

    def forward(self, x, use_cls_token_query=False):
        batch_size = x.shape[0]
        class_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([class_token, x], dim=1)

        x = self.layer_norm1(x)

        if use_cls_token_query:
            cls_token_query = x[:, 0:1]
            x, attn_weights = self.self_attn(
                query=cls_token_query, key=x, value=x, need_weights=True, average_attn_weights=False
            )

        else:
            x, attn_weights = self.self_attn(
                query=x, key=x, value=x, need_weights=True, average_attn_weights=False
            )

        class_token = x[:, 0:1, :] + class_token  # Residual connection
        class_token = self.layer_norm2(class_token.squeeze(1))
        class_token = self.mlp(class_token)
        class_token = self.layer_norm3(class_token)

        return class_token, attn_weights
