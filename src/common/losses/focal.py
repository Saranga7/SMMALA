import torch
import torch.nn as nn
import torchvision


class FocalLoss(nn.Module):
    """
    Loss function that implements the Focal Loss
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, num_classes, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        if self.num_classes > 2:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        else:
            focal_loss = torchvision.ops.sigmoid_focal_loss(
                inputs=outputs.squeeze(), targets=targets.float(), alpha=self.alpha, gamma=self.gamma, reduction="mean"
            )
        return focal_loss
