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

    def __init__(self, width: int, height: int, channels: int, batch_norm: bool = False):
        super().__init__()
        self.n_in = width * height * channels

        self.layers = nn.Sequential(
            LinearLayer(self.n_in, min(2000, self.n_in//2), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, self.n_in//2), min(2000, self.n_in//4), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, self.n_in//4), 1, torch.sigmoid, batch_norm=batch_norm),
        )
        print("discriminator:", self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvLayer(nn.Module):

    def __init__(
            self,
            chan_in: int,
            chan_out: int,
            kernel_size: int,
            padding: int = 0,
            act: Optional[Union[str, Callable]] = None,
            bias: bool = True,
            batch_norm: bool = False,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=self.chan_in,
            out_channels=self.chan_out,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )

        self.batch_norm = None#torch.nn.BatchNorm1d()

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
        y = self.conv(x)

        if self.batch_norm is not None:
            y = self.batch_norm(y)
        if self.act is not None:
            y = self.act(y)
        return y


class DiscriminatorConv(nn.Module):

    def __init__(self, width: int, height: int, channels: int, batch_norm: bool = False):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels

        self.layers = nn.Sequential(
            ConvLayer(chan_in=self.channels, chan_out=32, kernel_size=3, padding=1, act="relu"),
            ConvLayer(chan_in=32, chan_out=32, kernel_size=5, padding=2, act="relu"),
            ConvLayer(chan_in=32, chan_out=32, kernel_size=7, padding=3, act="relu"),
        )

        self.out_size = 32 * self.width * self.height
        self.l_out = LinearLayer(self.out_size, 1, torch.sigmoid, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1, self.channels, self.height, self.width)
        y = self.layers(y)
        y = y.reshape(-1, self.out_size)
        y = self.l_out(y)
        return y



Discriminator = DiscriminatorConv
