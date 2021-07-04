from typing import List, Optional, Callable, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import PIL.Image


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


class ConvLayer(nn.Module):

    def __init__(
            self,
            chan_in: int,
            chan_out: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            act: Optional[Union[str, Callable]] = None,
            bias: bool = True,
            batch_norm: bool = False,
            transpose: bool = False,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.reverse = transpose

        klass = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = klass(
            in_channels=self.chan_in,
            out_channels=self.chan_out,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(self.chan_out)
        else:
            self.batch_norm = None

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


class FanOut(nn.Module):
    """
    one input to parallel outputs.
    """
    def __init__(self, *layers: Callable):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [
            layer(x)
            for layer in self.layers
        ]
        return torch.cat(out, dim=-1)


class SinusLayer(torch.nn.Module):

    def __init__(
            self, n_in: int, freq: float = 10., max_freq: float = 1000.,
            bypass: bool = False,
            learn_freq: bool = True,
            learn_amp: bool = True,
    ):
        super().__init__()
        self.n_in = n_in
        self.max_freq = max_freq
        self.frequency = torch.nn.Parameter(
            torch.clip(torch.rand(n_in) * freq, -self.max_freq, self.max_freq),
            #torch.linspace(0, freq, n_in),
            requires_grad=learn_freq,
        )
        self.phase = torch.nn.Parameter(torch.randn(n_in))
        self.amp = torch.nn.Parameter(torch.randn(n_in) / np.sqrt(n_in), requires_grad=learn_amp)
        self.bypass = bypass
        if self.bypass:
            self.bypass_amp = torch.nn.Parameter(torch.randn(n_in))

    def forward(self, x):
        f = torch.clip(self.frequency, -self.max_freq, self.max_freq)
        y = torch.sin(x * f + self.phase) * self.amp
        if self.bypass:
            y *= self.bypass_amp
        return y
