import time
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchvision import datasets
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
from torchvision.utils import make_grid

import PIL.Image

from tqdm import tqdm

from generator import Discriminator
#from generator import GeneratorLinear as Generator
from generator_cppm import GeneratorCPPM2 as Generator


class ImageDataset:

    ROOT = str(Path("~").expanduser() / "prog" / "data" / "datasets")

    def __init__(self, name: str):
        self.name = name

        transform = VF.to_tensor
        #if self.name == "STL10":
        #    transform = self._transform_cifar10

        self.data = getattr(datasets, self.name)(
            root=self.ROOT,
            download=True,
            transform=transform,
        )
        self.shape = self.data.data.shape
        self.num = self.shape[0]
        if self.name.startswith("CIFAR"):
            self.height, self.width = self.shape[-3:-1]
            self.channels = self.shape[-1] if len(self.shape) == 4 else 1
        else:
            self.height, self.width = self.shape[-2:]
            self.channels = self.shape[-3] if len(self.shape) == 4 else 1

        print(f"dataset: {self.num} x {self.width}x{self.height}x{self.channels}")

    def random_samples(self, count: int) -> torch.Tensor:
        return torch.cat([
            self.data[random.randrange(len(self.data))][0].reshape(1, -1)
            for i in range(count)
        ])

    def random_samples_with_labels(self, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = [
            self.data[random.randrange(len(self.data))]
            for i in range(count)
        ]
        return (
            torch.cat([s[0].reshape(1, -1) for s in samples]),
            torch.Tensor([s[1] for s in samples]),
        )

    @classmethod
    def _transform_cifar10(cls, image: PIL.Image.Image) -> torch.Tensor:
        out = VF.to_tensor(image)
        print(out.shape)
        return out


class ReproTestTrainer:

    def __init__(
            self,
            data: ImageDataset,
            device: str = "cuda",
    ):
        self.data = data
        self.device = device
        self.width, self.height = self.data.width, self.data.height
        self.channels = self.data.channels
        self.n_generator_in = 32
        self.batch_size = 50
        self.label_features = dict()

        self.samples = self.data.random_samples(self.batch_size).to(self.device)
        self.features = torch.randn(self.batch_size, self.n_generator_in).to(self.device)

        self.generator = Generator(self.n_generator_in, self.width, self.height, self.channels).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            weight_decay=0.001,
        )

    def train(self, epochs: int = 10000):

        last_print_time = 0
        last_snapshot_time = 0

        for epoch in tqdm(range(epochs)):

            gen_images = self.generator.forward(self.features)
            gen_images = gen_images.reshape(self.batch_size, -1)

            loss = F.mse_loss(gen_images, self.samples)

            self.generator.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

            cur_time = time.time()
            if cur_time - last_print_time >= .5:
                last_print_time = cur_time

                print(
                    f"loss {loss:.3f}"
                )

            if cur_time - last_snapshot_time >= 3:
                last_snapshot_time = cur_time

                image = (
                    torch.clamp(gen_images[:6*6], 0, 1)
                        .reshape(-1, self.channels, self.height, self.width)
                )
                image = make_grid(image, nrow=6)
                image = VF.resize(image, [image.shape[-2]*4, image.shape[-1]*4], PIL.Image.NEAREST)
                image = VF.to_pil_image(image)
                image.save("./snapshot.png")


class Trainer:

    def __init__(
            self,
            data: ImageDataset,
            device: str = "cuda",
    ):
        self.data = data
        self.device = device
        self.width, self.height = self.data.width, self.data.height
        self.channels = self.data.channels
        self.n_generator_in = 100
        self.batch_size = 20

        self.generator = Generator(self.n_generator_in, self.width, self.height, self.channels).to(self.device)
        self.discriminator = Discriminator(self.width, self.height, self.channels).to(self.device)

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
            lr=0.0001,
            weight_decay=0.001,
        )

    def generate_images(self, count: int, random_transform: bool = False) -> torch.Tensor:
        #generator_noise = torch.randn(count, self.n_generator_in)
        generator_noise = torch.randn(count, self.n_generator_in) * .2
        for row in generator_noise:
            row[random.randrange(row.shape[0])] += 1.

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

        mutual_inhibit = 5.
        last_discriminator_loss = 0.

        last_print_time = 0
        last_snapshot_time = time.time()
        for epoch in tqdm(range(50000)):

            # -- generate batch of images --

            generator_image_batch = self.generate_images(self.batch_size * 2)

            # -- discriminate --

            discriminator_result = self.discriminate(generator_image_batch)

            # -- train generator on discriminator result --

            generator_loss = F.binary_cross_entropy(
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
                    .reshape(-1, self.channels, self.height, self.width)
                )
                image = make_grid(image, nrow=6)
                image = VF.resize(image, [image.shape[-2]*4, image.shape[-1]*4], PIL.Image.NEAREST)
                image = VF.to_pil_image(image)
                image.save("./snapshot.png")



def main():

    data = ImageDataset(
        #"MNIST"
        #"FashionMNIST"
        "CIFAR10"
        #"CIFAR100"
        #"STL10"
    )
    # print(data.targets.shape)

    t = Trainer(data)
    #t = ReproTestTrainer(data)
    t.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

