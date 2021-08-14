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

try:
    from .sdf import *
    from .render import *
    from .models import *
except ImportError:
    from sdf import *
    from render import *
    from models import *

from little_server import LittleServer


def train_sdf(
        server: LittleServer,
        model: nn.Module,
        sdf: Callable,
        batch_size: int,
        epochs: int,
):
    assert batch_size % 2 == 0, "batch_size must be even"

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=.001,
        weight_decay=0.0001,
    )

    losses = []
    last_server_time = 0

    for epoch in range(epochs):
        # print("-"*10, "epoch", epoch, "-"*10)

        pos_batch, target_dist_batch = build_position_batch(sdf, batch_size * 8 // 10, batch_size * 2 // 10)

        dist_batch = model.forward(pos_batch)

        loss = criterion(dist_batch, target_dist_batch)
        losses.append(float(loss))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        cur_time = time.time()
        if cur_time - last_server_time > 2:
            last_server_time = cur_time
            plot_loss_history(server, epoch, losses)


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
    df.iloc[-200:].plot(title="training loss", ax=ax)

    fig2, ax = plt.subplots(figsize=(6, 2))
    df.rolling(100).mean().clip(0, 1.5).plot(title="training loss (ma)", ax=ax)

    server.set_cell("loss", images=[fig, fig2])
    server.set_cell("status", code=f"epoch {epoch}\nloss {losses[-1]}")


def sdf_scene(pos: torch.Tensor) -> torch.Tensor:
    d = sdf_sphere(pos, 1)
    d = torch.minimum(d, sdf_sphere(pos - torch.Tensor([-1,0,0]), .5))
    d = torch.minimum(d, sdf_sphere(pos - torch.Tensor([1,0,0]), .5))
    d = torch.minimum(d, sdf_sphere(pos - torch.Tensor([0,1.25,0]), .25))
    return d


if __name__ == "__main__":
    server = LittleServer()
    server.start()
    server.set_cell_layout("loss", [1, 5], [1, 7], fit=True)
    server.set_cell_layout("status", [1, 5], [7, 13], fit=True)

    model = Net1()

    train_sdf(
        server=server,
        sdf=sdf_scene,
        model=model,
        batch_size=1000,
        epochs=5000,
    )

    torch.save(model.state_dict(), "./model-snapshot.pt")
