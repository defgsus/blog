from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

import PIL.Image

from tqdm import tqdm

device = "cuda"


def get_uv(resolution: int, dimensions: int):
    space = torch.zeros((resolution, resolution, dimensions))
    for x in range(resolution):
        for y in range(resolution):
            space[y, x] = torch.Tensor(
                [x / (resolution-1), y / (resolution-1)] + [.5] * (dimensions - 2)
            )

    return (space - .5) * 2.


class KaliSetModule(torch.nn.Module):

    def __init__(
            self,
            resolution: int,
            num_iterations: int,
            dimensions: int = 3,
            zoom: float = 1.,
            position: Sequence[float] = (0, 0, 0),
            learn_parameters: bool = True,
            learn_zoom: bool = True,
            learn_position: bool = True,
            learn_space: bool = False,
            accum: str = "avg",
    ):
        super().__init__()
        self._num_iterations = num_iterations
        self._resolution = resolution
        self._dimensions = dimensions
        self._accum = accum

        self.kali_parameters = torch.nn.Parameter(
            torch.ones((self._num_iterations, self._dimensions)) * .5,
            requires_grad=learn_parameters,
        )

        self.position_parameters = torch.nn.Parameter(
            torch.Tensor(position),
            requires_grad=learn_position,
        )

        self.zoom_parameter = torch.nn.Parameter(
            torch.Tensor([zoom]),
            requires_grad=learn_zoom,
        )

        self.start_space = torch.nn.Parameter(
            get_uv(self._resolution, self._dimensions),
            requires_grad=learn_space,
        )
        # self.start_space = get_uv(self._resolution, self._dimensions).to(device)

    def forward(self):
        space = self.start_space

        space = space * self.zoom_parameter
        space += self.position_parameters

        if self._accum == "avg":
            accum = torch.zeros(space.shape).to(device)
            for i in range(self._num_iterations):
                dot_p = torch.sum(space * space, dim=-1, keepdim=True)
                space = torch.abs(space) / (.00001 + dot_p)
                accum += space
                space -= self.kali_parameters[i]

            accum /= self._num_iterations
            return torch.clip(accum, 0, 1)

        else:
            for i in range(self._num_iterations - 1):
                dot_p = torch.sum(space * space, dim=-1, keepdim=True)
                space = torch.abs(space) / (.00001 + dot_p)
                space -= self.kali_parameters[i]

            dot_p = torch.sum(space * space, dim=-1, keepdim=True)
            space = torch.abs(space) / (.00001 + dot_p)

            return torch.clip(space, 0, 1)
