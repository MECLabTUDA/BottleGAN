import torch.nn as nn
import torch

class MultiArgsIdentity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        return args


class TemperatureScaledPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5, requires_grad=True)

    def forward(self, x):
        return  x / self.T