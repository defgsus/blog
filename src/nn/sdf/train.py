import time
import math
from typing import Callable, Tuple, Sequence
import sys
sys.path.insert(0, "../..")

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .sdf import *
    from .render import *
    from .models import *
except ImportError:
    from sdf import *
    from render import *
    from models import *

from little_server import LittleServer


render_pos = torch.Tensor([0, .5, -2])


def train_sdf(
        server: LittleServer,
        model: nn.Module,
        sdf: Callable,
        batch_size: int,
        epochs: int,
        position_mode: str = "surface",
):
    assert batch_size % 2 == 0, "batch_size must be even"

    criterion = nn.L1Loss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.0001)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=1., weight_decay=0.0001)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, momentum=0.5)

    def prod(*values):
        p = values[0]
        for v in values[1:]:
            p *= v
        return p

    num_params = sum(
        sum(prod(*p.shape) for p in g["params"])
        for g in optimizer.param_groups
    )
    print("  trainable params:", num_params)

    losses = []
    last_server_time = 0
    last_image_time = 0

    if position_mode == "surface":
        surface_positions = get_random_surface_positions(sdf, 100000)
        print("surface_positions:", torch.min(surface_positions), "-", torch.max(surface_positions))

    print("start training")
    for epoch in tqdm(range(epochs)):
        # print("-"*10, "epoch", epoch, "-"*10)

        if position_mode == "surface":
            pos_batch = (
                surface_positions[torch.randperm(surface_positions.shape[0])[:batch_size]]
                + torch.randn(batch_size, 3) * 0.2
            )
            target_dist_batch = sdf(pos_batch)

        elif position_mode == "random":
            pos_batch = torch.rand(batch_size, 3) * 4 - 2
            target_dist_batch = sdf(pos_batch)

        elif position_mode == "half":
            pos_batch, target_dist_batch = build_position_batch_inout(sdf, batch_size // 2, batch_size // 2)
            #pos_batch, target_dist_batch = build_position_batch(sdf, batch_size * 8 // 10, batch_size * 2 // 10)

        dist_batch = model.forward(pos_batch)

        loss_dist = criterion(dist_batch, target_dist_batch)

        if False:# % 20 == 0:
            target_normal_batch = sdf_normal(sdf, pos_batch, e=0.1, normalized=False)
            normal_batch = sdf_normal(model, pos_batch, e=0.1, normalized=False)
            loss_normal = criterion(normal_batch, target_normal_batch)

            loss = loss_dist + loss_normal
        else:
            loss = loss_dist

        losses.append(float(loss))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        cur_time = time.time()
        if cur_time - last_server_time > 2:
            last_server_time = cur_time
            plot_loss_history(server, epoch, losses)

        if cur_time - last_image_time > 20 or epoch == epochs - 1:
            with torch.no_grad():
                server.set_cell("image", image=raymarch(model, render_pos, as_pil=True)[0])
            last_image_time = cur_time


def build_position_batch(
        sdf: Callable,
        num_in: int,
        num_out: int,
        pos_min: float = -2.,
        pos_max: float = 2.,
) -> Tuple[torch.Tensor, torch.Tensor]:

    pos_batch = []
    dist_batch = []
    cur_num_in = 0
    cur_num_out = 0

    num_tries = 0
    while cur_num_in < num_in or cur_num_out < num_out:
        # print("---", num_tries, cur_num_in, cur_num_out)
        if num_tries > 50 * (num_in + num_out):
            raise RuntimeError(
                f"Could not find surface points, got {cur_num_in} in / {cur_num_out} out"
            )
        num_tries += 1

        num_left = num_in + num_out  #num_in - cur_num_in + num_out - cur_num_out

        pos = torch.rand(num_left, 3) * (pos_max - pos_min) + pos_min
        dist = sdf(pos)

        dist_inside = torch.abs(dist) <= 0.01
        if cur_num_in < num_in and torch.any(dist_inside):
            pos_inside = pos[dist_inside]
            pos_batch.append(pos_inside[:num_in - cur_num_in])
            dist_batch.append(dist[dist_inside][:num_in - cur_num_in])
            cur_num_in += pos_inside.shape[0]

        dist_outside = ~dist_inside
        if cur_num_out < num_out and torch.any(dist_outside):
            pos_outside = pos[dist_outside]
            pos_batch.append(pos_outside[:num_out - cur_num_out])
            dist_batch.append(dist[dist_outside][:num_out - cur_num_out])
            cur_num_out += pos_outside.shape[0]

    return torch.cat(pos_batch), torch.cat(dist_batch)#.reshape(num_in + num_out, 1)


def build_position_batch_inout(
        sdf: Callable,
        num_in: int,
        num_out: int,
        pos_min: float = -2.,
        pos_max: float = 2.,
) -> Tuple[torch.Tensor, torch.Tensor]:

    pos_batch = []
    dist_batch = []
    cur_num_in = 0
    cur_num_out = 0

    num_tries = 0
    while cur_num_in < num_in or cur_num_out < num_out:
        # print("---", num_tries, cur_num_in, cur_num_out)
        if num_tries > 50 * (num_in + num_out):
            raise RuntimeError(
                f"Could not find surface points, got {cur_num_in} in / {cur_num_out} out"
            )
        num_tries += 1

        num_left = num_in + num_out  #num_in - cur_num_in + num_out - cur_num_out

        pos = torch.rand(num_left, 3) * (pos_max - pos_min) + pos_min
        dist = sdf(pos)

        dist_inside = dist <= 0.
        if cur_num_in < num_in and torch.any(dist_inside):
            pos_inside = pos[dist_inside]
            pos_batch.append(pos_inside[:num_in - cur_num_in])
            dist_batch.append(dist[dist_inside][:num_in - cur_num_in])
            cur_num_in += pos_inside.shape[0]

        dist_outside = dist >= 0.
        if cur_num_out < num_out and torch.any(dist_outside):
            pos_outside = pos[dist_outside]
            pos_batch.append(pos_outside[:num_out - cur_num_out])
            dist_batch.append(dist[dist_outside][:num_out - cur_num_out])
            cur_num_out += pos_outside.shape[0]

    return torch.cat(pos_batch), torch.cat(dist_batch)#.reshape(num_in + num_out, 1)


def plot_loss_history(server: LittleServer, epoch: int, losses: Sequence[float]):
    df = pd.DataFrame(losses)

    fig, ax = plt.subplots(figsize=(6, 2))
    df.iloc[-1000:].plot(title="training loss", ax=ax)

    fig2, ax = plt.subplots(figsize=(6, 2))
    df.rolling(100).mean().plot(title="training loss (ma)", ax=ax)

    server.set_cell("loss", images=[fig, fig2])
    server.set_cell("status", code=f"epoch {epoch}\nloss {losses[-1]}")


def sdf_scene(pos: torch.Tensor) -> torch.Tensor:
    #d = sdf_sphere(pos, 1)
    #d = torch.minimum(d, sdf_sphere(pos - torch.Tensor([-1,0,0]), .5))
    #d = torch.minimum(d, sdf_sphere(pos - torch.Tensor([1,0,0]), .5))
    #d = torch.minimum(d, sdf_sphere(pos - torch.Tensor([0,1.25,0]), .25))
    d = sdf_difference(
        sdf_box(pos, torch.Tensor([0.5, .7, .6])),
        sdf_box(pos, torch.Tensor([0.3, .5, .7]))
    )
    d = sdf_union(d, sdf_sphere(pos - torch.Tensor([0,1.25,0]), .25))
    #d = sdf_tube(pos, .5, 1)
    d = sdf_union(d, sdf_tube(pos - torch.Tensor([0,1.25,0]), .1, 0))
    return d


def print_model(model: nn.Module):
    print(model.state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=.001,
        weight_decay=0.0001,
    )


if __name__ == "__main__":
    model = Mat4Net()
    # print_model(model); exit()

    server = LittleServer()
    server.start()
    server.set_cell_layout("loss", [1, 5], [1, 7], fit=True)
    server.set_cell_layout("status", [1, 5], [7, 13], fit=True)
    server.set_cell_layout("target", [5, 9], [1, 4], fit=True)
    server.set_cell_layout("image", [5, 9], [4, 7], fit=True)

    server.set_cell("target", image=raymarch(sdf_scene, render_pos, as_pil=True)[0])

    train_sdf(
        server=server,
        sdf=sdf_scene,
        model=model,
        batch_size=1000,
        epochs=10000,
    )

    torch.save(model.state_dict(), "./model-snapshot.pt")
