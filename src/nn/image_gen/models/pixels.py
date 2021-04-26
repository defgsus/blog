import random
from typing import Union, Sequence

import PIL.Image
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from .base import FixedBase


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
            resolution: Sequence[int] = (48, 48)
    ):
        super().__init__(resolution)
        self.pixels = torch.nn.Parameter(
            #load_image("/home/bergi/Pictures/__diverse/Annual-Sunflower.jpg", resolution)
            torch.rand((resolution[1], resolution[0], 3)) * .1 + .4
        )
        self.amp = torch.nn.Parameter(torch.rand(3) * .2 + .5)
        self.bias = torch.nn.Parameter(torch.rand(3) * .1)

    def forward(self):
        return torch.clamp(
            self.pixels * self.amp + self.bias, 0, 1
        )

        pixels = self.pixels

        if 1:  # shift offset randomly
            off_x = random.randint(0, 100)
            off_y = random.randint(0, 100)
            #print(self.pixels.shape)
            if off_x:
                pixels = torch.cat((self.pixels[:,-off_x:,:], self.pixels[:,off_x:,:]), dim=1)
            if off_y:
                pixels = torch.cat((self.pixels[-off_y:,:,:], self.pixels[off_y:,:,:]), dim=0)

            #pixels = pixels * .9 + .1 * self.pixels

        #print(pixels.shape)
        pixels = pixels + torch.rand((self.resolution[1], self.resolution[0], 3)).to(pixels.device) * .2
        return pixels * self.amp + self.offset
