import math
from typing import Callable, Tuple, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from siren_pytorch import SirenNet


class SinusLayer(nn.Module):

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out

        self.freq = nn.Parameter(torch.zeros(self.num_out))
        self.phase = nn.Parameter(torch.zeros(self.num_out))

        #if self.num_in != self.num_out:
        self.input = nn.Linear(self.num_in, self.num_out)

        self.init_()

    def init_(self) -> "SinusLayer":
        # fac = 1. / math.sqrt(self.num_in)
        w = 5.
        with torch.no_grad():
            self.freq.uniform_(-w, w)# = nn.Parameter(torch.randn(self.num_out) * 15. * fac)
            self.phase.uniform_(-w, w)# = nn.Parameter(torch.randn(self.num_out) * 3. * fac)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input(x)
        y = torch.sin(self.freq * y + self.phase)
        return y


class Net1(nn.Module):

    def __init__(self, num_feat: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            SinusLayer(3, num_feat),
            SinusLayer(num_feat, num_feat),
            SinusLayer(num_feat, num_feat),
            nn.Linear(num_feat, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).reshape(-1)


class Net2(nn.Module):

    def __init__(self, num_feat: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            SinusLayer(3, num_feat),
            SinusLayer(num_feat, num_feat),
            SinusLayer(num_feat, num_feat),
            SinusLayer(num_feat, num_feat),
            SinusLayer(num_feat, num_feat),
            SinusLayer(num_feat, num_feat),
            nn.Linear(num_feat, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).reshape(-1)


class Net3(SirenNet):

    def __init__(self):
        super().__init__(3, 32, 1, num_layers=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).reshape(-1)
