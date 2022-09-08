from torch import nn
from numpy import random
import numpy as np
import torch
from torch import Tensor
import math
from torch.nn import init
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.modules.module import _addindent

class Conv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)

        self.w = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        nn.init.dirac_(self.w.weight.data)
        self.std = nn.Parameter(torch.zeros_like(self.weight.data), requires_grad=False)
        self.gauss_scale = 0.2


    def forward(self, input: Tensor) -> Tensor:
        self.weight.data += (torch.normal(0, torch.nan_to_num(self.std) + 1e-6) * self.gauss_scale)
        outputs = super().forward(self.w(input))
        return outputs

        
      