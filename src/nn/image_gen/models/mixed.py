from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

from .base import FixedBase, get_uv
from .dist import CircleGen
from .pixels import FixedPixels


class CirclesPixels(FixedBase):
    def __init__(
            self,
            resolution: Sequence[int] = (48, 48),
    ):
        super().__init__(resolution)

        self.circles = CircleGen()
        self.pixels = FixedPixels(resolution)

    def forward(self):
        uv = get_uv(self.resolution, 2).reshape(-1, 2).to(self.device)
        return (
            self.pixels.forward()
            + self.circles(uv).reshape(self.resolution[1], self.resolution[0], 3)
        ) / 2.
