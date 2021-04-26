from typing import Union, Sequence

import numpy as np
import torch
import torch.nn
from torchvision.transforms import Resize

import PIL.Image

from tqdm import tqdm


class ImageGenBase(torch.nn.Module):
    """
    Base class for models that generate images.
    """

    @property
    def device(self):
        return next(self.parameters()).device

    def render_tensor(self, resolution: Sequence) -> torch.Tensor:
        """
        Render an image in the shape of [H, W, 3]
        :param resolution: tuple of Width and Height
        """
        raise NotImplementedError

    def render_image(self, resolution: Sequence) -> PIL.Image.Image:
        return to_image(self.render_tensor(resolution))


class ShadertoyBase(torch.nn.Module):
    """
    Base class for models that process [N, x, y] to [N, r, g, b]
    where N is some batch size.

    It is called after the shadertoy.com style of
    position -> color functions
    """

    @property
    def device(self):
        return next(self.parameters()).device

    def render_tensor(self, resolution: Sequence) -> torch.Tensor:
        """
        Render an image in the shape of [H, W, 3]
        :param resolution: tuple of Width and Height
        """
        uv = get_uv(resolution, 2).reshape(-1, 2).to(self.device)
        output = self(uv).reshape(resolution[1], resolution[0], 3)

        return output

    def render_image(self, resolution: Sequence) -> PIL.Image.Image:
        return to_image(self.render_tensor(resolution))


class FixedBase(ImageGenBase):
    """
    Base class for models that simply output a fixed
    resolution [Y, X, 3]

    They do not require an input to .forward() function
    """
    def __init__(self, resolution: Sequence[int]):
        super().__init__()
        assert len(resolution) == 2
        self.resolution = tuple(resolution)

    def render_tensor(self, resolution: Sequence) -> torch.Tensor:
        """
        Render an image in the shape of [H, W, 3]
        :param resolution: tuple of Width and Height
        """
        assert len(resolution) == 2
        output = self()
        if output.shape[0] != resolution[1] or output.shape[1] != resolution[0]:
            output = Resize((resolution[1], resolution[0]))(
                output.permute(2, 0, 1)
            ).permute(1, 2, 0)

        return output


def get_uv(shape: Sequence[int], dimensions: int) -> torch.Tensor:
    """
    Create a 2d matrix of vectors of size `dimensions`,
    where first 2 dims are filled with numbers in the range [-1, 1]
    according to their positions.
    """
    space = torch.zeros((shape[1], shape[0], dimensions))
    for x in range(shape[1]):
        for y in range(shape[0]):
            space[y, x] = torch.Tensor(
                [x / (shape[1]-1), y / (shape[0]-1)] + [0.] * (dimensions - 2)
            )

    return (space - .5) * 2.


def to_image(tensor: Union[np.ndarray, torch.Tensor]) -> PIL.Image:
    """
    Convert [H, W, RGB] matrix to Pillow Image.
    Color-channels are clipped to the range [0, 1]
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    img = np.clip(tensor, 0, 1) * 255
    img = img.astype(np.uint8)
    return PIL.Image.fromarray(img)
