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


class Trainer:

    def __init__(
            self,
            data: VisionDataset,
            device: str = "cuda"
    ):
        self.data = data
        self.device = device
        self.width, self.height = data.data.shape[-1], data.data.shape[-2]
        self.n_generator_in = 8
        self.batch_size = 20

        self.generator = Generator(self.n_generator_in, self.width * self.height).to(self.device)
        self.discriminator = Discriminator(self.width * self.height).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.001,
            weight_decay=0.001,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.00005,
            weight_decay=0.001,
        )

    def generate_images(self, count: int) -> torch.Tensor:
        #generator_noise = torch.randn(count, self.n_generator_in)
        generator_noise = torch.randn(count, self.n_generator_in) * .2
        for row in generator_noise:
            row[random.randrange(row.shape[0])] += 1.

        generator_image_batch = self.generator.forward(generator_noise.to(self.device))
        return generator_image_batch

    def random_real_images(self, count: int) -> torch.Tensor:
        return torch.cat([
            self.data[random.randrange(len(self.data))][0].reshape(1, -1)
            for i in range(count)
        ]).to(self.device)

    def train(self):
        expected_discriminator_result_for_gen = torch.ones(self.batch_size*2).reshape(-1, 1).to(self.device)
        expected_discriminator_result_for_dis = torch.Tensor(
            [1] * self.batch_size + [0] * self.batch_size
        ).reshape(-1, 1).to(self.device)

        mutual_inhibit = 5.
        last_discriminator_loss = 0.

        last_print_time = 0
        last_snapshot_time = time.time()
        for epoch in tqdm(range(50000)):

            # -- generate batch of images --

            generator_image_batch = self.generate_images(self.batch_size * 2)

            # -- discriminate --

            discriminator_result = self.discriminator.forward(generator_image_batch)

            # -- train generator on discriminator result --

            generator_loss = F.mse_loss(
                discriminator_result,
                expected_discriminator_result_for_gen,
            )
            last_generator_loss = float(generator_loss)
            generator_loss = generator_loss / (1. + mutual_inhibit * last_discriminator_loss)

            gen_image_diversity = generator_image_batch.std(dim=0).std()
            # generator_loss += torch.pow(.5 - gen_image_diversity, 2)

            self.generator.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            # -- build image batch for discriminator --

            dis_image_batch = torch.cat([
                self.random_real_images(self.batch_size),
                self.generate_images(self.batch_size),
            ])

            # -- train discriminator --

            discriminator_result = self.discriminator.forward(dis_image_batch)

            discriminator_loss = F.mse_loss(
                discriminator_result,
                expected_discriminator_result_for_dis,
            )
            last_discriminator_loss = float(discriminator_loss)
            discriminator_loss = discriminator_loss / (1. + mutual_inhibit * last_generator_loss)

            self.discriminator.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            cur_time = time.time()
            if cur_time - last_print_time >= .5:
                last_print_time = cur_time

                correct_all = (torch.abs(
                    discriminator_result - expected_discriminator_result_for_dis
                ) < .5).type(torch.int8)
                #print(correct_all)
                correct_real = correct_all[:self.batch_size].sum()
                correct_gen = correct_all[self.batch_size:].sum()

                print(
                    f"loss"
                    f" G {last_generator_loss:.3f}"
                    f" D {last_discriminator_loss:.3f} correct real/gen {correct_real} / {correct_gen}"
                    f" | diversity {gen_image_diversity:.3f}"
                )

            if cur_time - last_snapshot_time >= 3:
                last_snapshot_time = cur_time

                image = (
                    torch.clamp(generator_image_batch[:6*6], 0, 1)
                    .reshape(-1, 1, self.height, self.width)
                )
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

    t = Trainer(data)
    t.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

