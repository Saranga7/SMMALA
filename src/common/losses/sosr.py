import torch
import torch.nn as nn


class SOSRLoss(nn.Module):
    def __init__(self, cost_matrix):
        """
        Smooth One-Sided Regression (SOSR) Loss.

        Args:
            cost_matrix (torch.Tensor): A matrix of shape [num_classes, num_classes] specifying the cost
                                        of misclassifying class i as class j.
        """
        super(SOSRLoss, self).__init__()
        self.cost_matrix = cost_matrix

    def forward(self, logits, targets):
        """
        Compute the SOSR loss.

        Args:
            logits (torch.Tensor): Raw logits of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground-truth labels of shape [batch_size].

        Returns:
            torch.Tensor: Computed SOSR loss.
        """
        cost_table = self.cost_matrix[targets]  # [batch_size, num_classes]

        # 1 for incorrect classes // -1 for correct class
        delta = torch.ones_like(cost_table)
        delta.scatter_(1, targets.unsqueeze(1), -1)  # delta[i, target] = -1

        loss = torch.log1p(torch.exp(delta * (logits - cost_table)))  # log1p = log(1 + x)
        return loss.mean()  # TODO: Check if this is correct

    @staticmethod
    def compute_risks_and_probs(logits, cost_matrix):
        """
        Compute risks and probabilities by applying softmax after calculating risks.

        Args:
            logits (torch.Tensor): Raw model outputs, shape [batch_size, num_classes].

        Returns:
            risks (torch.Tensor): Risk vectors, shape [batch_size, num_classes].
            probs (torch.Tensor): Probabilities, shape [batch_size, num_classes].
        """
        # Compute risk vectors
        risks = torch.matmul(logits, cost_matrix)  # [batch_size, num_classes]

        # Apply softmax to convert risks into probabilities
        probs = torch.softmax(-risks, dim=1)  # Negative sign to prioritize lower risks
        return probs
