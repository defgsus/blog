import math
from typing import Callable, Tuple, Sequence, Optional

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


class Mat4Layer(nn.Module):
    def __init__(self, n_in: int = 1, n_out: int = 1, act: Optional[Callable] = torch.sin, w_std: float = 1.):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight = nn.Parameter(torch.rand(4 * self.n_out, 4 * self.n_in) * w_std * 2. - w_std)
        self.bias = nn.Parameter(torch.rand(4 * self.n_out) * w_std * 2. - w_std)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.act is not None:
            y = self.act(y)
        return y


class Mat4Net(nn.Module):

    def __init__(self, num_layers: int = 4, num_features: int = 4):
        super().__init__()
        self.layers = nn.Sequential(*(
            Mat4Layer(
                n_in=1 if i == 0 else num_features,
                n_out=1 if i == num_layers -1 else num_features,
                act=torch.sin if i < num_layers - 1 else None,
                w_std=2. if i < num_layers - 1 else .5,
            )
            for i in range(num_layers)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # append x to xyz to form a vec4
        if x.shape[-1] == 3:
            x = torch.cat([x, x[:, 0].unsqueeze(-1)], dim=-1)

        y = self.layers(x)

        # drop yzw
        y = y[:, 0]
        return y
