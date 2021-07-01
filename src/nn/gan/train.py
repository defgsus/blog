import time
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchvision import datasets
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
from torchvision.utils import make_grid
import PIL.Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from models import get_module
from little_server import LittleServer
from datasets import ImageDataset


class Trainer:

    def __init__(
            self,
            Generator: nn.Module,
            Discriminator: nn.Module,
            data: ImageDataset,
            device: str = "cuda",
            server: Optional[LittleServer] = None,
            batch_size: int = 50,
            n_generator_in: int = 32,
            checkpoint: Optional[str] = None,
    ):
        self.data = data
        self.device = device
        self.server = server
        self.width, self.height = self.data.width, self.data.height
        self.channels = self.data.channels
        self.n_generator_in = n_generator_in
        self.batch_size = batch_size

        self.generator = Generator(self.n_generator_in, self.width, self.height, self.channels).to(self.device)
        self.discriminator = Discriminator(self.width, self.height, self.channels).to(self.device)

        #self.criterion = F.binary_cross_entropy
        self.criterion = F.l1_loss

        if checkpoint:
            state_dict = torch.load(checkpoint)
            self.generator.load_state_dict(state_dict)

        self.stats = {
            "gen_loss": [],
            "dis_loss": [],
            "dis_correct_real": [],
            "dis_correct_gen": [],
        }

        for key in ("generator", "discriminator"):
            model = getattr(self, key)
            num_params = 0
            for p in model.parameters():
                num_params += p.shape.numel()
            print(key, "params:", num_params)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            weight_decay=0.001,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.00002,
            weight_decay=0.001,
        )

    def plot_loss_history(self):
        if self.server:
            df = pd.DataFrame(self.stats)

            fig, ax = plt.subplots(figsize=(6, 2))
            df.plot(title="training loss", ax=ax)
            self.server.set_cell("loss", image=fig)
            fig.clear()

            fig, ax = plt.subplots(figsize=(6, 2))
            df.rolling(20).mean().plot(title="training loss (ma)", ax=ax)
            self.server.set_cell("loss_ma", image=fig)
            fig.clear()

            #fig = plt.figure(figsize=(6, 2))
            #plt.title(f"training loss")
            #plt.plot(self.stats["gen_loss"], label="gen loss")
            #plt.plot(self.stats["dis_loss"], label="dis loss")
            #plt.plot(self.stats["dis_correct_real"], label="corr. real")
            #plt.plot(self.stats["dis_correct_gen"], label="corr. gen")
            #plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            #self.server.set_cell("loss", image=fig)
            #fig.clear()

    def generate_images(self, count: int, random_transform: bool = False) -> torch.Tensor:
        generator_noise = torch.randn(count, self.n_generator_in)
        #generator_noise = torch.randn(count, self.n_generator_in) * .2
        #for row in generator_noise:
        #    row[random.randrange(row.shape[0])] += 1.

        # generator_noise = generator_noise / generator_noise.norm(dim=-1, keepdim=True)

        generator_image_batch = self.generator.forward(generator_noise.to(self.device))
        if generator_image_batch.ndim > 2:
            generator_image_batch = generator_image_batch.reshape(count, -1)

        if random_transform:
            if not hasattr(self, "_random_transforms"):
                self._random_transforms = VT.RandomRotation(7, center=[1./3, 2./3])

            generator_image_batch = \
                    generator_image_batch.reshape(-1, self.channels, self.height, self.width)

            generator_image_batch = torch.cat([
                self._random_transforms(generator_image_batch[i]).unsqueeze(0)
                for i in range(generator_image_batch.shape[0])
            ])

            generator_image_batch = \
                    generator_image_batch.reshape(-1, self.channels * self.height * self.width)

        return generator_image_batch

    def random_real_images(self, count: int) -> torch.Tensor:
        batch = self.data.random_samples(count).to(self.device)
        return batch

    def discriminate(self, image_batch: torch.Tensor) -> torch.Tensor:
        d = self.discriminator.forward(image_batch)
        # d = torch.round(d * 5) / 5
        return d

    def train(self):
        expected_discriminator_result_for_gen = torch.ones(self.batch_size*2).reshape(-1, 1).to(self.device)
        expected_discriminator_result_for_dis = torch.Tensor(
            [1] * self.batch_size + [0] * self.batch_size
        ).reshape(-1, 1).to(self.device)

        mutual_inhibit = 0.
        last_discriminator_loss = 0.

        last_print_time = 0
        last_snapshot_time = time.time()
        for epoch in tqdm(range(50000)):

            for gen_iter in range(10):
                # -- generate batch of images --

                generator_image_batch = self.generate_images(self.batch_size * 2)

                # -- discriminate --

                discriminator_result = discriminator_result_for_gen = self.discriminate(generator_image_batch)
                # self.server.set_cell("gen_dis_result", text=f"{discriminator_result}")
                # -- train generator on discriminator result --

                generator_loss = self.criterion(
                    discriminator_result,
                    expected_discriminator_result_for_gen,
                )

                last_generator_loss = float(generator_loss)
                generator_loss = generator_loss / (1. + mutual_inhibit * last_discriminator_loss)

                #gen_image_diversity = generator_image_batch.std(dim=0).std()
                gen_image_diversity = (
                    generator_image_batch[:-1] - generator_image_batch[1:]
                ).abs().mean()
                # generator_loss += 10.*torch.pow(1. - gen_image_diversity, 2)

                self.generator.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()

            # -- build image batch for discriminator --

            dis_image_batch = torch.cat([
                self.random_real_images(self.batch_size),
                self.generate_images(self.batch_size),
            ])

            # -- train discriminator --

            discriminator_result = self.discriminate(dis_image_batch)

            discriminator_loss = F.binary_cross_entropy(
                discriminator_result,
                expected_discriminator_result_for_dis,
            )
            last_discriminator_loss = float(discriminator_loss)
            discriminator_loss = discriminator_loss / (1. + mutual_inhibit * last_generator_loss)

            if last_generator_loss <= 1.:
                self.discriminator.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

            with torch.no_grad():
                correct_all = (torch.abs(
                    discriminator_result - expected_discriminator_result_for_dis
                ) < .5).type(torch.int8)
                #print(correct_all)
                correct_real = int(correct_all[:self.batch_size].sum())
                correct_gen = int(correct_all[self.batch_size:].sum())

            self.stats["gen_loss"].append(last_generator_loss)
            self.stats["dis_loss"].append(last_discriminator_loss)
            self.stats["dis_correct_real"].append(correct_real / self.batch_size)
            self.stats["dis_correct_gen"].append(correct_gen / self.batch_size)

            cur_time = time.time()
            if cur_time - last_print_time >= .5:
                last_print_time = cur_time

                print(
                    f"loss"
                    f" G {last_generator_loss:.3f}"
                    f" D {last_discriminator_loss:.3f} correct real/gen {correct_real} / {correct_gen}"
                    f" | diversity {gen_image_diversity:.3f}"
                )
                if self.server:
                    self.server.set_cell(
                        "stats",
                        text=f"epoch {epoch}"
                             f", loss G {last_generator_loss:.3f}"
                             f" D {last_discriminator_loss:.3f} correct real/gen {correct_real} / {correct_gen}"
                    )

            if self.server:
                if epoch == 0 or cur_time - last_snapshot_time >= 3:
                    last_snapshot_time = cur_time

                    image = (
                        torch.clamp(generator_image_batch[:6*6], 0, 1)
                        .reshape(-1, self.channels, self.height, self.width)
                    )
                    image = make_grid(image, nrow=6)
                    #image = VF.resize(image, [image.shape[-2]*4, image.shape[-1]*4], PIL.Image.NEAREST)
                    image = VF.to_pil_image(image)
                    #image.save("./snapshot.png")

                    self.server.set_cell("samples", image=image)
                    self.plot_loss_history()

                    df = pd.DataFrame({
                        "gen": discriminator_result_for_gen.detach().cpu().reshape(self.batch_size*2).numpy(),
                        "real/gen": discriminator_result.detach().cpu().reshape(self.batch_size*2).numpy(),
                    })
                    fig, ax = plt.subplots(figsize=(6, 2))
                    df.plot(title="discriminator result", ax=ax)
                    self.server.set_cell("dis_result", image=fig)
                    fig.clear()



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "generator", type=str,
        help="Name of the Generator class"
    )
    parser.add_argument(
        "discriminator", type=str,
        help="Name of the Discriminator class"
    )
    parser.add_argument(
        "-ds", "--dataset", type=str, nargs="?", default="MNIST",
        help="Name of the torchvision dataset"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, nargs="?", default=50,
        help="Batch-size: Number of images to be able to reproduce simultaneously"
    )
    parser.add_argument(
        "-f", "--features", type=int, nargs="?", default=32,
        help="Length of generator input"
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, nargs="?", default=None,
        help="filepath to state_dict for generator"
    )
    parser.add_argument(
        "-d", "--device", type=str, nargs="?", default="cuda",
        help="Device to run on (cpu, cuda, cuda:0, ..)"
    )

    return parser.parse_args()


def main(args):

    data = ImageDataset(
        args.dataset
        #"MNIST"
        #"FashionMNIST"
        #"CIFAR10"
        #"CIFAR100"
        #"STL10"
    )
    # print(data.targets.shape)

    server = LittleServer()
    server.start()

    t = Trainer(
        Generator=get_module(args.generator),
        checkpoint=args.checkpoint,
        Discriminator=get_module(args.discriminator),
        data=data,
        device=args.device,
        server=server,
        batch_size=args.batch_size,
        n_generator_in=args.features,
    )
    t.train()


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        pass

