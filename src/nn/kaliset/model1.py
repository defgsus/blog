from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

import PIL.Image

from tqdm import tqdm

device = "cuda"


class LinearLayer(torch.nn.Module):

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        scale = 1. / np.sqrt(n_in)
        self.weight = torch.nn.Parameter(torch.rand(n_in, n_out) * scale)
        self.bias = torch.nn.Parameter(torch.randn(n_out) * scale)
        self.act = torch.nn.Tanh()

    def forward(self, x):
        return self.act((x @ self.weight) + self.bias)


class Model1(torch.nn.Module):

    def __init__(
            self,
            layer_size: Sequence[int] = (32, 24, 12)
    ):
        super().__init__()

        layers = [LinearLayer(2, layer_size[0])]
        for size, next_size in zip(layer_size, layer_size[1:]):
            layers.append(LinearLayer(size, next_size))

        self.layers = torch.nn.Sequential(*layers)
        self.wout = torch.nn.Parameter(torch.randn(layer_size[-1], 3))

    def weight_info(self) -> str:
        return ", ".join(
            f"w{i+1}={l.weight.abs().sum() / (l.weight.shape[0] * l.weight.shape[1]):.4f} "
            for i, l in enumerate(self.layers)
        )

    def forward(self, position):
        n = self.layers(position)
        n = (n @ self.wout)
        return torch.clip(n, 0, 1)
