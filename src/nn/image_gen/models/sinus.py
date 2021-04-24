from typing import Union, Sequence

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F


class LinearLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int, act="tanh"):
        super().__init__()
        scale = 1. / np.sqrt(n_in)
        self.weight = torch.nn.Parameter(torch.rand(n_in, n_out) * scale)
        self.bias = torch.nn.Parameter(torch.randn(n_out) * scale)
        if act is None:
            self.act = None
        elif act == "tanh":
            self.act = torch.nn.Tanh()
        else:
            self.act = act

    def forward(self, x):
        y = (x @ self.weight) + self.bias
        if self.act is not None:
            y = self.act(y)
        return y


class Sinus(torch.nn.Module):

    def __init__(self, n_in: int, freq: float = 10., bypass: bool = False):
        super().__init__()
        self.n_in = n_in
        self.frequency = torch.nn.Parameter(torch.randn(n_in) * freq)
        self.phase = torch.nn.Parameter(torch.randn(n_in))
        self.amp = torch.nn.Parameter(torch.randn(n_in))
        self.bypass = bypass
        if self.bypass:
            self.bypass_amp = torch.nn.Parameter(torch.randn(n_in))

    def forward(self, x):
        y = torch.sin(x * self.frequency + self.phase) * self.amp
        if self.bypass:
            y *= self.bypass_amp
        return y


class SinModel2(torch.nn.Module):

    def __init__(
            self,
            layer_size: Sequence[int] = (512, 18, 12, 6)
    ):
        super().__init__()
        layers = [
            LinearLayer(2, 32),
            Sinus(32, freq=10),
            LinearLayer(32, layer_size[0]),
            Sinus(layer_size[0], freq=100),
        ]
        for size, next_size in zip(layer_size, layer_size[1:]):
            layers.append(LinearLayer(size, next_size))

        self.layers = torch.nn.Sequential(*layers)
        self.wout = torch.nn.Parameter(torch.randn(layer_size[-1], 3))

    def weight_info(self) -> str:
        return ", ".join(
            f"w{i+1}={l.weight.abs().sum() / (l.weight.shape[0] * l.weight.shape[1]):.4f} "
            for i, l in enumerate(self.layers)
            if isinstance(l, LinearLayer)
        )

    def forward(self, position):
        n = self.layers(position)
        n = (n @ self.wout)
        return torch.clip(n, 0, 1)


class SinModel1(torch.nn.Module):

    def __init__(
            self,
            layer_size: Sequence[int] = (512, 8,)
    ):
        super().__init__()
        layer_size = [2] + list(layer_size)
        layers = []
        for i, (size, next_size) in enumerate(zip(layer_size, layer_size[1:])):
            layers += [
                LinearLayer(size, next_size),
                Sinus(next_size, freq=10., bypass=True),
            ]

        self.layers = torch.nn.Sequential(*layers)
        self.wout = torch.nn.Parameter(torch.randn(layer_size[-1], 3))

    def weight_info(self) -> str:
        return ", ".join(
            f"w{i+1}={l.weight.abs().sum() / (l.weight.shape[0] * l.weight.shape[1]):.4f} "
            for i, l in enumerate(self.layers)
            if isinstance(l, LinearLayer)
        )

    def forward(self, position):
        n = self.layers(position)
        n = (n @ self.wout)
        return torch.clip(n, 0, 1)

