from typing import List, Optional, Callable, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(torch.nn.Module):

    def __init__(
            self,
            n_in: int,
            n_out: int,
            act: Optional[Union[str, Callable]] = None,
            bias: bool = True,
            batch_norm: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = bias
        self.use_batchnorm = batch_norm

        scale = 1. / np.sqrt(n_in)
        self.weight = torch.nn.Parameter(torch.rand(n_in, n_out) * scale)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.randn(n_out) * scale)
        if self.use_batchnorm:
            self.batch_norm = torch.nn.BatchNorm1d(self.n_out)
        if act is None:
            self.act = None
        elif isinstance(act, str):
            if hasattr(torch, act):
                self.act = getattr(torch, act)
            else:
                self.act = getattr(F, act)
        else:
            self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"x {x.shape} @ weight {self.weight.shape}")
        y = x @ self.weight
        if self.use_bias:
            y = y + self.bias
        if self.use_batchnorm:
            y = self.batch_norm(y)
        if self.act is not None:
            y = self.act(y)
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"n_in={self.n_in}, n_out={self.n_out})"


class Generator(nn.Module):

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.layers = nn.Sequential(
            LinearLayer(n_in, 512, F.leaky_relu, batch_norm=True, bias=False),
            # nn.Dropout(0.5),
            LinearLayer(512, 1024, F.leaky_relu, batch_norm=True),
            # nn.Dropout(0.5),
            LinearLayer(1024, n_out, "tanh", batch_norm=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return y
        #return torch.clamp(y, 0, 1)


class DiscriminatorLinear(nn.Module):

    def __init__(self, n_in: int, batch_norm: bool = False):
        super().__init__()
        self.n_in = n_in

        self.layers = nn.Sequential(
            LinearLayer(n_in, min(2000, n_in//2), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, n_in//2), min(2000, n_in//4), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, n_in//4), 1, torch.sigmoid, batch_norm=batch_norm),
        )
        print("discriminator:", self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


Discriminator = DiscriminatorLinear
