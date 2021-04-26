from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

from .base import ImageGenBase


class Kaliset(ImageGenBase):
    """
    The infamous kaliset fractal
    """
    def __init__(
            self,
            num_iterations: int = 17,
            dimensions: int = 8,
            zoom: float = .02,
            position: Sequence[float] = (0.5, 0.5, 0.5),
            learn_parameters: bool = True,
            learn_zoom: bool = True,
            learn_position: bool = True,
            accum: str = "final",
    ):
        super().__init__()

        assert dimensions >= 2

        self.num_iterations = num_iterations
        self.dimensions = dimensions
        self.accum = accum

        self.kali_parameters = torch.nn.Parameter(
            torch.randn((self.num_iterations, self.dimensions)) * .1 + .5,
            requires_grad=learn_parameters,
        )

        self.position_parameters = torch.nn.Parameter(
            self._expand_dims(torch.Tensor(position)) + torch.randn(dimensions) * 0.01,
            requires_grad=learn_position,
        )

        self.zoom_parameter = torch.nn.Parameter(
            torch.Tensor([zoom]),
            requires_grad=learn_zoom,
        )

        self.w_out = torch.nn.Parameter(
            torch.randn((self.dimensions, 3)) / np.sqrt(self.dimensions)
        )

    def _expand_dims(self, t: torch.Tensor):
        if t.shape[-1] < self.dimensions:
            if t.dim() == 1:
                zeros = torch.zeros(self.dimensions - t.shape[-1]).to(t.device)
                return torch.cat((t, zeros), dim=-1)
            else:
                zeros = torch.zeros((t.shape[-2], self.dimensions - t.shape[-1])).to(t.device)
                return torch.cat((t, zeros), dim=-1)
        return t

    def forward(self, position):
        position = self._expand_dims(position)

        space = position * self.zoom_parameter + self.position_parameters

        if self.accum == "avg":
            accum = torch.zeros(space.shape).to(self.device)
            for i in range(self.num_iterations):
                dot_p = torch.sum(space * space, dim=-1, keepdim=True)
                space = torch.abs(space) / (.00001 + dot_p)
                accum += space
                space -= self.kali_parameters[i]

            accum /= self.num_iterations
            result = accum

        else:
            for i in range(self.num_iterations - 1):
                dot_p = torch.sum(space * space, dim=-1, keepdim=True)
                space = torch.abs(space) / (.00001 + dot_p)
                space -= self.kali_parameters[i]

            dot_p = torch.sum(space * space, dim=-1, keepdim=True)
            space = torch.abs(space) / (.00001 + dot_p)

            result = space

        out = result @ self.w_out

        return torch.clip(out, 0, 1)
