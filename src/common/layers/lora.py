import torch
from torch import nn


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def add_lora(model, layer_type, rank, alpha, num_blocks_to_modify):
    """Add LoRA to all Linear layers in the last `num_blocks_to_modify` blocks of the model, and to all other Linear layers outside of these blocks once."""

    rank = int(rank)
    alpha = float(alpha)
    num_blocks_to_modify = int(num_blocks_to_modify)

    _modify_linear_layers_outside_blocks(model, layer_type, rank, alpha)

    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
        start_block_idx = len(model.blocks) - num_blocks_to_modify
        for block_idx in range(start_block_idx, len(model.blocks)):
            block = model.blocks[block_idx]
            _modify_linear_layers(block, layer_type, rank, alpha)

    return model


def _modify_linear_layers_outside_blocks(module, layer_type, rank, alpha):
    """Recursively modify Linear layers in the model, excluding the 'blocks' module."""

    for name, child in module.named_children():
        if isinstance(child, layer_type):
            setattr(module, name, LinearWithLoRA(child, rank, alpha))
        elif isinstance(child, nn.Module) and not (name == "blocks" and isinstance(child, nn.ModuleList)):
            _modify_linear_layers_outside_blocks(child, layer_type, rank, alpha)


def _modify_linear_layers(module, layer_type, rank, alpha):
    for name, child in module.named_children():
        if isinstance(child, layer_type):
            setattr(module, name, LinearWithLoRA(child, rank, alpha))
        elif isinstance(child, nn.Module):
            _modify_linear_layers(child, layer_type, rank, alpha)
