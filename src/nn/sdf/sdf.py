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


from little_server import LittleServer


def train_sdf(
        server: LittleServer,
        model: nn.Module,
        sdf: Callable,
        batch_size: int,
):
    assert batch_size % 2 == 0, "batch_size must be even"

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=.001,
        weight_decay=0.0001,
    )

    losses = []
    last_server_time = 0

    for epoch in range(10000):
        # print("-"*10, "epoch", epoch, "-"*10)

        pos_batch, target_dist_batch = build_position_batch(sdf, batch_size // 2, batch_size // 2)

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

        if num_tries > 50 * (num_in + num_out):
            raise RuntimeError(
                f"Could not find surface points, got {cur_num_in} in / {cur_num_out} out"
            )
        num_tries += 1

        pos = torch.rand(3) * (pos_max - pos_min) + pos_min
        dist = sdf(pos)
        #print(pos, dist)
        if (
                (dist <= 0. and cur_num_in < num_in)
            or  (dist > 0. and cur_num_out < num_out)
        ):
            pos_batch.append(pos.unsqueeze(0))
            dist_batch.append(dist)
            if dist <= 0:
                cur_num_in += 1
            else:
                cur_num_out += 1

    return torch.cat(pos_batch), torch.Tensor(dist_batch).reshape(num_in + num_out, 1)


def plot_loss_history(server: LittleServer, epoch: int, losses: Sequence[float]):
    df = pd.DataFrame(losses)

    fig, ax = plt.subplots(figsize=(6, 2))
    df.iloc[-200:].plot(title="training loss", ax=ax)

    fig2, ax = plt.subplots(figsize=(6, 2))
    df.rolling(100).mean().clip(0, 1.5).plot(title="training loss (ma)", ax=ax)

    server.set_cell("loss", images=[fig, fig2])
    server.set_cell("status", code=f"epoch {epoch}\nloss {losses[-1]}")

def sdf_sphere(pos: Sequence[float]) -> float:
    return math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2) - 1.


class Net1(nn.Module):

    def __init__(self, num_feat: int = 8):
        super().__init__()
        self.inp = nn.Linear(3, num_feat)

        self.freq = nn.Parameter(torch.randn(num_feat))
        self.phase = nn.Parameter(torch.randn(num_feat))

        self.out = nn.Linear(num_feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inp(x)
        y = torch.sin(self.freq * y + self.phase)
        return self.out(y)


if __name__ == "__main__":
    server = LittleServer()
    server.start()
    server.set_cell_layout("loss", [1, 5], [1, 7], fit=True)
    server.set_cell_layout("status", [1, 5], [7, 13], fit=True)

    model = Net1()

    train_sdf(
        server=server,
        sdf=sdf_sphere,
        model=model,
        batch_size=1024,
    )
