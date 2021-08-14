"""
A collection of signed distance functions that work
with a batch of 3D positions
"""
from typing import Callable
import torch


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def length(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(dot(x, x))


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / length(x).reshape(-1, 1)


def sdf_normal(sdf: Callable, pos: torch.Tensor, e: float = 0.001) -> torch.Tensor:
    return normalize(torch.cat([
        (sdf(pos + torch.Tensor([e, 0, 0])) - sdf(pos - torch.Tensor([e, 0, 0]))).reshape(-1, 1),
        (sdf(pos + torch.Tensor([0, e, 0])) - sdf(pos - torch.Tensor([0, e, 0]))).reshape(-1, 1),
        (sdf(pos + torch.Tensor([0, 0, e])) - sdf(pos - torch.Tensor([0, 0, e]))).reshape(-1, 1),
    ], dim=-1))


def sdf_sphere(pos: torch.Tensor, radius: float = 1.) -> torch.Tensor:
    return torch.sqrt(dot(pos, pos)) - radius #+ torch.sin(pos[:, 0] * 10.) * .1


def sdf_plane(pos: torch.Tensor, norm: torch.Tensor, offset: float = 0.) -> torch.Tensor:
    if norm.ndim != 2:
        norm = norm.reshape(1, 3).expand(pos.shape[0], -1)
    return dot(pos, norm) - offset


def sdf_plane_y(pos: torch.Tensor, offset: float = 0.) -> torch.Tensor:
    return pos[:, 1] - offset

