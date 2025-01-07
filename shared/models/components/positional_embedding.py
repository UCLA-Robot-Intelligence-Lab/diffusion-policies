import math
import torch
import torch.nn as nn

"""
Selected Dimension key:

B: batch size
D: embedding dimension
H: (temporary) half of D
"""


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim_D):
        super().__init__()
        self.dim_D = dim_D

    def forward(self, x_B):
        device = x_B.device
        half_dim_H = self.dim_D // 2
        emb_H = math.log(10000) / (half_dim_H - 1)
        emb_H = torch.exp(torch.arange(half_dim_H, device=device) * -emb_H)
        emb_BH = x_B[:, None] * emb_H[None, :]
        emb_BD = torch.cat((emb_BH.sin(), emb_BH.cos()), dim=-1)
        return emb_BD
