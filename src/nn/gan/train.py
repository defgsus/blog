import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchvision.datasets import VisionDataset, FashionMNIST
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
from torchvision.utils import make_grid

import PIL.Image

from tqdm import tqdm

from generator import Generator, Discriminator


def train(data: VisionDataset, device: str = "cuda"):
    width, height = data.data.shape[-1], data.data.shape[-2]
    n_generator_in = 8
    batch_size = 50

    generator = Generator(n_generator_in, width * height).to(device)
    discriminator = Discriminator(width * height).to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=0.001,
        weight_decay=0.001,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.00005,
        weight_decay=0.001,
    )

    expected_discriminator_result_for_gen = torch.ones(batch_size).reshape(-1, 1).to(device)
    expected_discriminator_result_for_dis = torch.Tensor(
        [1] * batch_size + [0] * batch_size
    ).reshape(-1, 1).to(device)

    last_print_time = 0
    last_snapshot_time = time.time()
    for epoch in tqdm(range(50000)):

        # -- generate batch of images --

        generator_noise = torch.randn(batch_size, generator.n_in).to(device)
        generator_image_batch = generator.forward(generator_noise)

        # -- discriminate --
        # print("G", generator_image_batch.shape)
        discriminator_result = discriminator.forward(generator_image_batch)

        # -- train generator on discriminator result --

        generator_loss = F.mse_loss(
            discriminator_result,
            expected_discriminator_result_for_gen,
        )

        generator.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # -- build image batch for discriminator --

        generator_noise = torch.randn(batch_size, generator.n_in).to(device)
        generator_image_batch = generator.forward(generator_noise)

        dis_image_batch = torch.cat([
            torch.cat([
                data[random.randrange(len(data))][0].reshape(1, -1)
                for i in range(batch_size)
            ]).to(device),
            generator_image_batch
        ])

        # -- train discriminator --

        discriminator_result = discriminator.forward(dis_image_batch)

        discriminator_loss = F.mse_loss(
            discriminator_result,
            expected_discriminator_result_for_dis,
        )

        discriminator.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        cur_time = time.time()
        if cur_time - last_print_time >= .5:
            last_print_time = cur_time

            correct_all = (torch.abs(
                discriminator_result - expected_discriminator_result_for_dis
            ) < .5).type(torch.int8)
            #print(correct_all)
            correct_real = correct_all[:batch_size].sum()
            correct_gen = correct_all[batch_size:].sum()

            print(
                f"loss"
                f" G {generator_loss:.3f}"
                f" D {discriminator_loss:.3f} correct real/gen {correct_real} / {correct_gen}"
                #"generator", model.weight_info() if hasattr(model, "weight_info") else "-",
                #f"(img {round(float(image_loss), 3)}"
                #f" param {round(float(parameter_loss), 3)})"
            )

        if cur_time - last_snapshot_time >= 3:
            last_snapshot_time = cur_time

            image = torch.clamp(generator_image_batch[:6*6], 0, 1).reshape(-1, 1, height, width)
            image = make_grid(image, nrow=6)
            image = VF.resize(image, [image.shape[-2]*4, image.shape[-1]*4], PIL.Image.NEAREST)
            image = VF.to_pil_image(image)
            image.save("./snapshot.png")



def main():

    data = FashionMNIST(
        root=str(Path("~").expanduser() / "prog" / "data" / "datasets" / "fashion-mnist"),
        train=True,
        download=True,
        transform=VF.to_tensor,
    )

    print(data.data.shape)
    print(data.targets.shape)

    train(data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

