"""
A collection of signed distance functions that work
with a batch of 3D positions
"""
from typing import Callable, Union
import torch


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2, f"got {a.ndim}"
    assert a.shape[1] == 3, f"got {a.shape[1]}"
    assert b.ndim == 2, f"got {b.ndim}"
    assert b.shape[1] == 3, f"got {b.shape[1]}"
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def length(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(dot(x, x))


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / length(x).reshape(-1, 1)


# --------------- combinations ---------------

def sdf_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def sdf_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def sdf_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, -b)


# ---------------- primitives ----------------


def sdf_plane(pos: torch.Tensor, norm: torch.Tensor, offset: float = 0.) -> torch.Tensor:
    if norm.ndim != 2:
        norm = norm.reshape(1, 3).expand(pos.shape[0], -1)
    return dot(pos, norm) - offset


def sdf_plane_x(pos: torch.Tensor, offset: float = 0.) -> torch.Tensor:
    return pos[:, 0] - offset


def sdf_plane_y(pos: torch.Tensor, offset: float = 0.) -> torch.Tensor:
    return pos[:, 1] - offset


def sdf_plane_z(pos: torch.Tensor, offset: float = 0.) -> torch.Tensor:
    return pos[:, 2] - offset


def sdf_sphere(pos: torch.Tensor, radius: float = 1.) -> torch.Tensor:
    return torch.sqrt(dot(pos, pos)) - radius #+ torch.sin(pos[:, 0] * 10.) * .1


def sdf_tube(pos: torch.Tensor, radius: float = 1., axis: int = 0) -> torch.Tensor:
    pos = pos.clone()
    pos[:,axis] = 0
    return sdf_sphere(pos, radius=radius)


def sdf_box(pos: torch.Tensor, radius: Union[float, torch.Tensor]) -> torch.Tensor:
    q = torch.abs(pos) - radius
    return (
        length(torch.clamp_min(q, 0))
        + torch.clamp_max(torch.max(q, dim=-1, keepdim=True).values.reshape(-1), 0)
    )


# ------------ sampling ------------


def sdf_normal(
        sdf: Callable,
        pos: torch.Tensor,
        e: float = 0.001,
        normalized: bool = True,
) -> torch.Tensor:
    n = torch.cat([
        (sdf(pos + torch.Tensor([e, 0, 0])) - sdf(pos - torch.Tensor([e, 0, 0]))).reshape(-1, 1),
        (sdf(pos + torch.Tensor([0, e, 0])) - sdf(pos - torch.Tensor([0, e, 0]))).reshape(-1, 1),
        (sdf(pos + torch.Tensor([0, 0, e])) - sdf(pos - torch.Tensor([0, 0, e]))).reshape(-1, 1),
    ], dim=-1)
    if normalized:
        return normalize(n)
    return n


def get_random_surface_positions(
        sdf: Callable,
        count: int,
        threshold: float = 0.001,
        num_iterations: int = 100,
) -> torch.Tensor:
    pos = torch.rand(count, 3) * 4. - 2.
    pos_on_surface = []

    num_pos_on_surface = 0
    for i in range(num_iterations):
        dist = sdf(pos)
        is_on_surface = torch.abs(dist) <= threshold
        pos_on_surface.append(pos[is_on_surface])
        num_pos_on_surface += torch.count_nonzero(is_on_surface)
        pos = pos[~is_on_surface]
        if not pos.shape[0]:
            break

        pos -= dist[~is_on_surface].reshape(-1, 1) * sdf_normal(sdf, pos)

        if i != 0 and i % 10 == 0:
            pos = torch.rand(count - num_pos_on_surface, 3) * 4. - 2.

    pos_on_surface = torch.cat(pos_on_surface)

    if pos_on_surface.shape[0] != count:
        raise RuntimeError(
            f"Could not find {count} surface positions with threshold {threshold}"
            f", found {pos_on_surface.shape[0]}"
        )

    return pos_on_surface
