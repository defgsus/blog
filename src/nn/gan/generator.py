from typing import List, Optional, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, act: Optional[Union[str, Callable]] = "tanh"):
        super().__init__()
        scale = 1. / np.sqrt(n_in)
        self.weight = torch.nn.Parameter(torch.rand(n_in, n_out) * scale)
        self.bias = torch.nn.Parameter(torch.randn(n_out) * scale)
        if act is None:
            self.act = None
        elif isinstance(act, str):
            if act == "tanh":
                self.act = torch.tanh
            else:
                self.act = getattr(F, act)
        else:
            self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"x {x.shape} @ weight {self.weight.shape}")
        y = x @ self.weight + self.bias
        if self.act is not None:
            y = self.act(y)
        return y


class Generator(nn.Module):

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.layers = nn.Sequential(
            LinearLayer(n_in, 512, F.relu),
            LinearLayer(512, 512, F.relu),
            LinearLayer(512, n_out, F.tanh),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return y
        #return torch.clamp(y, 0, 1)


class Discriminator(nn.Module):

    def __init__(self, n_in: int):
        super().__init__()
        self.n_in = n_in

        self.layers = nn.Sequential(
            LinearLayer(n_in, 512, F.relu),
            LinearLayer(512, 256, F.relu),
            LinearLayer(256, 1, F.tanh),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
