from torch import nn
from torch.nn import functional as F

from src.common.layers.attention import Attention


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) layer.

    GLUs allow the network to control information flow by learning which information to pass through and which to filter out.
    They can be seen as a learnable activation function that can adapt to the data.
    GLUs can help mitigate the vanishing gradient problem by providing a gating mechanism that allows gradients to flow more easily through the network.
    """

    def __init__(self, input_size, output_size):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_size, output_size * 2)

    def forward(self, x):
        x = self.linear(x)
        return F.glu(x, dim=-1)


class ResidualGatedAttention(nn.Module):
    """
    Residual Gated Attention (RGA) layer.

    RGA layers combine residual connections with gated linear units and attention mechanisms.
    - The attention mechanism allows the network to focus on relevant parts of the input.
    - The GLU provides adaptive gating of information flow.
    - The residual connection (x + residual) helps with gradient flow in deep networks and allows the network to learn incremental transformations.
    - Layer normalization helps stabilize the learning process by normalizing the inputs to each layer.
    """

    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.attention = Attention(dim=in_features, num_heads=num_heads)
        self.glu = GatedLinearUnit(in_features, in_features)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)
