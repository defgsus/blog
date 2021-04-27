import random
from typing import Union, Sequence

import PIL.Image
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from .base import ShadertoyBase


class CircleGen(ShadertoyBase):
    """
    Draws `num_features` shaded circles in the space [-1, 1]
    """
    def __init__(
            self,
            num_features: int = 96,
            min_radius: float = 0.05,
            max_radius: float = 2.0,
            min_gradient: float = 0.2,
    ):
        assert min_radius > 0
        assert max_radius > 0

        super().__init__()
        self.positions = torch.nn.Parameter(
            torch.randn((num_features, 2)) / 4.
        )
        self.radii = torch.nn.Parameter(
            torch.rand(num_features) * (max_radius - min_radius) + min_radius,
            #requires_grad=False,
        )
        self.gradients = torch.nn.Parameter(
            torch.ones(num_features) * (1. - min_gradient) + min_gradient,
            #requires_grad=False,
        )
        self.min_gradient = min_gradient

        radii_scale = self.radii.reshape(num_features, 1).repeat(1, 3)
        self.colors = torch.nn.Parameter(
            (torch.rand((num_features, 3)) * .5 + .5) / np.sqrt(num_features) / torch.pow(1. + radii_scale, 3)
        )
        self.min_radius = min_radius
        self.max_radius = max_radius

        self.bias = torch.nn.Parameter(
            (torch.rand(3) * .1)
        )

    def forward(self, pos):
        n_in = pos.shape[0]
        n_feat = self.positions.shape[0]

        # repeat and reshape input position batch
        #   and own feature position batch to match shape
        in_pos = pos.repeat(1, n_feat).reshape(n_in, -1, 2)
        feature_pos = self.positions.repeat(n_in, 1).reshape(n_in, -1, 2)

        # calculate distance of each input position to each feature
        dist = (in_pos - feature_pos).square()
        dist = torch.sqrt(dist[:, :, 0] + dist[:, :, 1])
        # shape is now num_in x num_feat

        # convert distance to shaded circle
        radii = torch.clamp(self.radii.repeat(n_in, 1), self.min_radius, self.max_radius)
        amt = (radii - dist) / radii

        gradients = torch.clamp_min(self.gradients.repeat(n_in, 1), self.min_gradient)
        amt = 1. - (gradients - amt) / gradients
        amt = torch.clamp(amt, 0, 1)

        # apply feature colors, shape num_in x num_feat x 3
        colors = self.colors.repeat(n_in, 1).reshape(n_in, -1, 3)
        amt = amt.reshape(amt.shape[0], amt.shape[1], 1).repeat(1, 1, 3)
        amt = amt * colors

        # sum down to shape num_in x 3
        amt = amt.sum(dim=-2)

        amt = amt + self.bias

        return torch.clip(amt, 0, 1)


if __name__ == "__main__":

    model = CircleGen()

    # ten 2d positions
    position = torch.rand((10, 2))
    print(position)
    print(position.shape)

    output = model(position)
    print(f"out\n{output}")
