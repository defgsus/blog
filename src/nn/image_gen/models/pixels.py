import math
from typing import Union, Sequence

import PIL.Image
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torchvision.transforms import Resize, GaussianBlur

from .base import FixedBase
from .util import hsv_to_rgb


def load_image(filename: str, resolution: Sequence):
    image = PIL.Image.open(filename)
    rgb_image = PIL.Image.new("RGB", image.size)
    rgb_image.paste(image)
    rgb_image = rgb_image.resize(resolution, PIL.Image.LANCZOS)
    rgb_image = np.asarray(rgb_image).astype(np.float) / 255.
    return torch.Tensor(rgb_image)


class FixedPixels(FixedBase):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__(resolution)
        self.pixels = torch.nn.Parameter(
            #load_image("/home/bergi/Pictures/__diverse/Annual-Sunflower.jpg", resolution)
            torch.rand((resolution[1], resolution[0], 3))
        )
        self.amp = torch.nn.Parameter(torch.rand(3) * .2 + .5)
        self.bias = torch.nn.Parameter(torch.rand(3) * .1)
        self.gauss_blur = GaussianBlur(3, .35)

    def blur(self):
        with torch.no_grad():
            pixels = self.pixels.permute(2, 0, 1)
            pixels = self.gauss_blur(pixels)
            self.pixels[:, :, :] = pixels.permute(1, 2, 0)

    def forward(self):
        return torch.clamp(
            self.pixels * self.amp + self.bias, 0, 1
        )

    def train_step(self, epoch: int):
        self.blur()


class FixedPixelsHSV(FixedPixels):

    def forward(self):
        hsv = super().forward().permute(2, 0, 1)
        rgb = hsv_to_rgb(hsv)
        return rgb.permute(1, 2, 0)
