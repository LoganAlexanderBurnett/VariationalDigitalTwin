from __future__ import annotations

import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, input1, input2):
        seq_len = input1.shape[1]
        weight = torch.linspace(seq_len / 5, 1, steps=seq_len, device=input1.device)
        mae = self.l1_loss(input1, input2)
        weighted_mae = torch.mul(mae, weight)
        return torch.mean(weighted_mae)


class WeightedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input1, input2):
        seq_len = input1.shape[1]
        weight = torch.linspace(seq_len / 5, 1, steps=seq_len, device=input1.device)
        mse = self.mse(input1, input2)
        weighted_mse = torch.mul(mse, weight)
        return torch.mean(weighted_mse)
