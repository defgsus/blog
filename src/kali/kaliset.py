from copy import copy
from typing import Sequence, Union, Any, Optional
import PIL
import PIL.Image
import numpy as np


class KaliSpace:
    """
    Just some location in space with 2d-rendering array output
    """
    def __init__(
            self,
            position: Union[float, Sequence] = 0.,
            scale: Union[float, Sequence] = 1.,
    ):
        self.position = position
        self.scale = scale

    def uv(self, size: Sequence, dimensions: int = 2, dtype=np.float):
        uv = np.ndarray([size[0], size[1], dimensions], dtype=dtype)
        for y in range(size[1]):
            for x in range(size[0]):
                uv[y, x] = [x+.5, y+.5] + [0.] * (dimensions - 2)

        uv[:, :, 0] -= size[0] * .5
        uv[:, :, 1] -= size[1] * .5
        uv /= size[1] * .5

        uv *= self.scale
        uv += self.position
        return uv


class KaliSet:
    """
    The infamous kali set fractal
    """
    def __init__(
            self,
            param: Sequence,
            iterations: int = 7,
    ):
        self.param = np.asarray(param)
        self.iterations = iterations

    def __call__(
            self,
            space: np.ndarray
    ) -> np.ndarray:
        space = copy(space)
        acc = self.acc_start(space)
        for i in range(self.iterations):
            dot_prod = np.sum(space * space, axis=-1, keepdims=True)
            space = np.abs(space) / (0.000000001 + dot_prod)

            self.accumulate(acc, space, dot_prod)

            space -= self.param

        return self.acc_end(acc, space)

    def acc_start(self, space: np.ndarray) -> Any:
        return np.zeros(space.shape, dtype=space.dtype)

    def accumulate(self, acc: Any, space: np.ndarray, dot_product: np.ndarray):
        acc += space

    def acc_end(self, acc: Any, space: np.ndarray) -> Any:
        return acc / self.iterations

    def plot(
            self,
            space: KaliSpace,
            size: Sequence = (100, 100),
            dimension: int = 0,
            max: Optional[float] = None,
    ):
        import plotly.express as px
        k = self(space.uv(size=size, dimensions=self.param.shape[0], dtype=self.param.dtype))
        k = k[:, :, dimension]
        if max is not None:
            k = np.clip(k, 0, max)
        px.imshow(k).show()

    def img(
            self,
            space: KaliSpace,
            size: Sequence = (100, 100),
            color_offset: float = 0.,
            color_scale: float = 1.,
    ) -> PIL.Image.Image:
        k = self(space.uv(size=size, dimensions=self.param.shape[0], dtype=self.param.dtype))
        rgb = np.apply_along_axis(lambda c: np.append(c, [0]), -1, k)
        rgb = color_offset + rgb * color_scale
        rgb = np.clip(rgb.round(), 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(rgb)


def create_space_2d(
        size: Sequence,
        dimensions: int = 2,
        offset: Optional[Union[float, Sequence]] = 0.,
        scale: Optional[Union[float, Sequence]] = 1.,
) -> np.ndarray:
    uv = np.ndarray([size[0], size[1], dimensions], dtype=np.float)
    for y in range(size[1]):
        for x in range(size[0]):
            uv[y, x] = [x+.5, y+.5] + [0.] * (dimensions - 2)

    uv[:, :, 0] -= size[0] * .5
    uv[:, :, 1] -= size[1] * .5
    uv /= size[1] * .5

    uv *= scale
    uv += offset
    return uv
