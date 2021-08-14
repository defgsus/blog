from typing import Callable, Tuple, Union
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import PIL.Image

try:
    from .sdf import *
except ImportError:
    from sdf import *


def get_pos_grid(
        size: int = 256,
        space: float = 1.25,
        z: float = 0.,
        flat: bool = False,
):
    grid = torch.linspace(-space, space, size).reshape(size, 1).expand(size, -1, 3).clone()
    grid[:, :, 1] = torch.linspace(space, -space, size).reshape(size, 1).expand(-1, size)
    grid[:, :, 2] = z
    if flat:
        grid = grid.reshape(size * size, 3)
    return grid


def render_sdf_grid(
        sdf: Callable,
        size: int = 256,
        space: float = 1.25,
        z: float = 0.,
        edge_width: float = 0.02,
):
    grid = get_pos_grid(size=size, space=space, z=z)

    dist = sdf(grid.reshape(size*size, 3)).reshape(size, size)
    rgb = torch.clamp(1.-torch.abs(dist) / edge_width, 0, 1).repeat(3, 1, 1)
    rgb[1, :, :] += torch.clamp(dist / 1., 0, 1)
    rgb[2, :, :] += torch.clamp(dist / -1., 0, 1)
    return VF.to_pil_image(rgb)


def raymarch(
        sdf: Callable,
        pos: torch.Tensor,
        size: int = 256,
        surface_threshold: float = 0.001,
        max_rays: int = 100,
        as_pil: bool = False,
        visible_radius: float = 2.,
) -> Tuple[
        Union[PIL.Image.Image, torch.Tensor],
        dict,
]:
    ray_pos = get_pos_grid(size=size, space=0.2, flat=True) + pos
    ray_dir = normalize(get_pos_grid(size=size, space=1., z=1.5, flat=True))
    # indices into original rays batch
    indices = torch.Tensor(list(range(size * size))).type(torch.long)
    # final color buffer
    color_buffer = torch.zeros(size * size, 3)

    statistics = {
        "ray_casts": 0,
        "hits": 0,
        "time": 0.,
    }

    start_time = time.time()
    for i in range(max_rays):
        dist = sdf(ray_pos)
        statistics["ray_casts"] += ray_pos.shape[0]

        # check for surface
        is_surface = torch.abs(dist) < surface_threshold
        statistics["hits"] = torch.count_nonzero(is_surface)

        if torch.any(is_surface):
            hit_pos = ray_pos[is_surface]
            normals = sdf_normal(sdf, hit_pos)

            # assign colors to color buffer
            color = normals * .5 + .5
            color[length(hit_pos) > visible_radius] *= 0
            color_buffer[indices[is_surface]] = color

        if torch.all(is_surface):
            break

        # remove all rays that hit surface
        keep_ray = ~is_surface
        indices = indices[keep_ray]
        dist = dist[keep_ray]
        ray_pos = ray_pos[keep_ray]
        ray_dir = ray_dir[keep_ray]

        # forward rays
        ray_pos = ray_pos + ray_dir * dist.reshape(-1, 1)

    statistics["time"] = time.time() - start_time

    color_buffer = color_buffer.reshape(size, size, 3)
    if as_pil:
        color_buffer = VF.to_pil_image(color_buffer.permute(2, 0, 1))
    return color_buffer, statistics

