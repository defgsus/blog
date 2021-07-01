import time
import random
from pathlib import Path
from typing import Tuple, Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchvision import datasets
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
from torchvision.utils import make_grid
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tqdm import tqdm

from datasets import ImageDataset
from models import get_module
from little_server import LittleServer


class ReproTestTrainer:

    def __init__(
            self,
            gen_model: nn.Module,
            data: ImageDataset,
            device: str = "cuda",
            server: Optional[LittleServer] = None,
            batch_size: int = 50,
            n_generator_in: int = 32,
    ):
        assert n_generator_in <= batch_size
        self.server = server
        self.data = data
        self.device = device
        self.width, self.height = self.data.width, self.data.height
        self.channels = self.data.channels
        self.n_generator_in = n_generator_in
        self.batch_size = batch_size
        self.label_features = dict()
        self.loss_history = []

        self.samples = self.data.random_samples(self.batch_size).to(self.device)
        self.features = None

        self.generator = gen_model(self.n_generator_in, self.width, self.height, self.channels).to(self.device)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            weight_decay=0.001,
        )

    def plot_features(self):
        if self.server:
            features = self.features[:,:2].cpu().numpy()
            fig = plt.figure(figsize=(2, 2))
            plt.title(f"{self.batch_size} features (first 2 of {self.n_generator_in} coords)")
            plt.scatter(features[:,0], features[:,1], s=1)
            self.server.set_cell("features", image=fig)

    def plot_loss_history(self):
        if self.server:
            fig = plt.figure(figsize=(4, 2))
            plt.title(f"training loss")
            plt.plot(self.loss_history)
            self.server.set_cell("loss", image=fig)

    def save_model(self, prefix: Optional[str] = None):
        filename = str(Path(__file__).resolve().parent / "checkpoints" / self.generator.__class__.__name__)
        if prefix:
            filename += prefix
        filename += ".pt"
        torch.save(
            self.generator.state_dict(),
            filename
        )

    def train(self, epochs: int = 5000):
        if self.features is None:
            if 0:
                reduced_size = [8, 8]
                reduction = PCA(self.n_generator_in)
                #reduction = TSNE(self.n_generator_in, method="exact", verbose=2)

                samples = self.samples.reshape(-1, self.channels, self.height, self.width)
                samples = VF.resize(samples, reduced_size, PIL.Image.BICUBIC)
                samples = samples.reshape(-1, self.channels * reduced_size[1] * reduced_size[0])
                features = reduction.fit_transform(samples.cpu())
                features /= np.max(np.abs(features))
                self.features = torch.Tensor(features).to(self.device)
            else:
                self.features = torch.randn(self.batch_size, self.n_generator_in, device=self.device)
        self.plot_features()

        last_print_time = 0
        last_snapshot_time = 0
        last_big_snapshot_time = 0

        for epoch in tqdm(range(epochs)):
            gen_images = self.generator.forward(self.features)
            gen_images = gen_images.reshape(self.batch_size, -1)

            loss = F.mse_loss(gen_images, self.samples)
            self.loss_history.append(float(loss))

            self.generator.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

            cur_time = time.time()
            if cur_time - last_print_time >= .5:
                last_print_time = cur_time

                print(
                    f"loss {loss:.3f}"
                )

            if self.server:
                self.server.set_cell("stats", text=f"epoch {epoch}, loss {loss:.3f}")

                if cur_time - last_snapshot_time >= 2:
                    last_snapshot_time = cur_time

                    image = (
                        torch.clamp(gen_images[:6*6], 0, 1)
                            .reshape(-1, self.channels, self.height, self.width)
                    )
                    image = make_grid(image, nrow=6)
                    #image = VF.resize(image, [image.shape[-2]*4, image.shape[-1]*4], PIL.Image.NEAREST)
                    image = VF.to_pil_image(image)
                    self.server.set_cell("samples", image=image)
                    #image.save("./snapshot.png")

                if cur_time - last_big_snapshot_time >= 5:
                    res = (128, 128)
                    last_big_snapshot_time = cur_time

                    with torch.no_grad():
                        self.generator.train(False)
                        gen_images = self.generator.forward(self.features[:9], width=res[0], height=res[1])
                        self.generator.train(True)
                    image = (
                        torch.clamp(gen_images, 0, 1)
                            .reshape(-1, self.channels, res[1], res[0])
                    )
                    image = make_grid(image, nrow=3)
                    image = VF.to_pil_image(image)
                    self.server.set_cell("big-samples", image=image)
                    self.plot_loss_history()

            if epoch != 0 and (epoch % 1000 == 0 or epoch+1 == epochs):
                self.save_model(f"-{epoch}-{loss:.3f}")


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "generator", type=str,
        help="Name of the Generator class"
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

    t = ReproTestTrainer(
        get_module(args.generator),
        data,
        args.device,
        server,
        batch_size=args.batch_size,
        n_generator_in=args.features,
    )
    t.train()


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        pass

