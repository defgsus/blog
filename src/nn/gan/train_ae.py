import sys
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

sys.path.insert(0, "../../")
from models import get_module, store_module, load_module
from little_server import LittleServer
from datasets import ImageDataset


class Trainer:

    def __init__(
            self,
            Encoder: nn.Module,
            Decoder: nn.Module,
            data: ImageDataset,
            device: str = "cuda",
            server: Optional[LittleServer] = None,
            batch_size: int = 50,
            n_features: int = 16,
            checkpoint_encoder: bool = False,
            checkpoint_decoder: bool = False,
    ):
        self.data = data
        self.device = device
        self.server = server
        self.width, self.height = self.data.width, self.data.height
        self.channels = self.data.channels
        self.n_features = n_features
        self.batch_size = batch_size

        self.encoder = Encoder(self.n_features, self.width, self.height, self.channels).to(self.device)
        self.decoder = Decoder(self.n_features, self.width, self.height, self.channels).to(self.device)

        self.criterion = F.mse_loss
        #self.criterion = F.l1_loss

        self.checkpoint_path = f"ae/{self.channels}x{self.width}x{self.height}"

        if checkpoint_encoder:
            load_module(self.encoder, self.checkpoint_path)
        if checkpoint_decoder:
            load_module(self.decoder, self.checkpoint_path)

        self.stats = {
            "loss": [],
            "features_std": [],
        }

        self.info_text = ""
        for key in ("encoder", "decoder"):
            model = getattr(self, key)
            num_params = 0
            for p in model.parameters():
                num_params += p.shape.numel()
            print(key, "params:", num_params)
            self.info_text += f"{key} params: {num_params}\n"

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001,
            weight_decay=0.001,
        )

        if self.server:
            self.server.set_cell_layout("stats", [1, 4], [1, 4])
            self.server.set_cell_layout("features", [4, 6], [1, 4], fit=True)
            self.server.set_cell_layout("real_samples", [1, 6], [4, 7])
            self.server.set_cell_layout("samples", [1, 6], [7, 10])
            self.server.set_cell_layout("fixed_samples", [1, 6], [10, 13])
            self.server.set_cell_layout("stats", [1, 6], [1, 4])
            self.server.set_cell_layout("loss", [6, 13], [1, 12])
            self.server.set_cell("actions", [6, 13], [12, 13], actions=[
                "new_samples", "store_weights"
            ])

    def store_weights(self):
        store_module(self.encoder, self.checkpoint_path)
        store_module(self.decoder, self.checkpoint_path)

    def plot_loss_history(self):
        if self.server:
            df = pd.DataFrame(self.stats)

            fig, ax = plt.subplots(figsize=(6, 2))
            df.plot(title="training loss", ax=ax)

            fig2, ax = plt.subplots(figsize=(6, 2))
            df.rolling(10000 // self.batch_size).mean().clip(0, 1.5).plot(title="training loss (ma)", ax=ax)

            self.server.set_cell("loss", images=[fig, fig2])

    def generate_images(self, count: int, random_transform: bool = False) -> torch.Tensor:
        if False and hasattr(self, "_gen_noise"):
            generator_noise = self._gen_noise
        else:
            generator_noise = torch.randn(count, self.n_features)
            self._gen_noise = generator_noise

        # generator_noise = generator_noise / generator_noise.norm(dim=-1, keepdim=True)

        generator_image_batch = self.generator.forward(generator_noise.to(self.device))

        if random_transform:
            if not hasattr(self, "_random_transforms"):
                self._random_transforms = VT.RandomRotation(7, center=[1./3, 2./3])

            generator_image_batch = torch.cat([
                self._random_transforms(generator_image_batch[i]).unsqueeze(0)
                for i in range(generator_image_batch.shape[0])
            ])

        return generator_image_batch

    def random_real_images(self, count: int) -> torch.Tensor:
        batch = self.data.random_samples(count).to(self.device)
        return batch

    def render_image_grid(self, batch: torch.Tensor, nrow: int = 6, min_width: int = 48) -> PIL.Image.Image:
        batch = torch.clamp(batch[:nrow*nrow], 0, 1)

        image = make_grid(batch, nrow=nrow)
        if self.width < min_width:
            factor = 1 + min_width // self.width
            image = VF.resize(image, [image.shape[-2]*factor, image.shape[-1]*factor], PIL.Image.NEAREST)
        image = VF.to_pil_image(image)
        return image

    def train(self):
        if self.server:
            self.server.set_cell("real_samples", image=self.render_image_grid(
                self.random_real_images(36)
            ))

        last_print_time = 0
        last_snapshot_time = time.time()

        sample_noise = torch.randn(36, self.n_features).to(self.device)

        for frame in tqdm(range(50000)):

            if self.server:
                action = self.server.get_action()
                if action:
                    name = action["name"]
                    if name == "new_samples":
                        sample_noise = torch.randn(36, self.n_features).to(self.device)
                    elif name == "store_weights":
                        self.store_weights()

            # -- build image batch --

            train_image_batch = self.random_real_images(self.batch_size)

            # -- train encoder/decoder --

            features = self.encoder.forward(train_image_batch)
            repro_image_batch = self.decoder(features)

            loss = self.criterion(
                repro_image_batch,
                train_image_batch,
            )

            self.decoder.zero_grad()
            loss.backward()
            self.optimizer.step()

            features_std = features.std()

            self.stats["loss"].append(float(loss))
            self.stats["features_std"].append(float(features_std))

            cur_time = time.time()
            if cur_time - last_print_time >= .5:
                last_print_time = cur_time

                print(
                    f"loss {float(loss):.3f}"
                )
                if self.server:
                    self.server.set_cell(
                        "stats",
                        text=(
                            self.info_text +
                            f"frame {frame}"
                            f", loss {loss:.3f}"
                            f", feat-std {features_std:.3f}"
                        )
                    )

            if self.server:
                if frame == 0 or cur_time - last_snapshot_time >= 3:
                    last_snapshot_time = cur_time

                    self.plot_loss_history()

                    df = pd.DataFrame(features[:10].detach().cpu().numpy())
                    fig, ax = plt.subplots(figsize=(6, 2))
                    df.transpose().plot(title="features", ax=ax)
                    self.server.set_cell("features", image=fig)

                    image_grid = self.render_image_grid(repro_image_batch)
                    self.server.set_cell("samples", image=image_grid)

                    with torch.no_grad():
                        sample_batch = self.decoder.forward(
                            sample_noise * features_std
                        )
                        image_grid = self.render_image_grid(sample_batch)
                        self.server.set_cell("fixed_samples", image=image_grid)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder", type=str,
        help="Name of the Encoder class"
    )
    parser.add_argument(
        "decoder", type=str,
        help="Name of the Decoder class"
    )
    parser.add_argument(
        "-ds", "--dataset", type=str, nargs="?", default="MNIST",
        help="Name of the torchvision dataset"
    )
    parser.add_argument(
        "-r", "--resize", type=int, nargs="?", default=None,
        help="Resize the dataset"
    )
    parser.add_argument(
        "-bw", "--black-white", type=bool, nargs="?", default=False, const=True,
        help="Convert dataset to grayscale"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, nargs="?", default=50,
        help="Batch-size: Number of images to be able to reproduce simultaneously"
    )
    parser.add_argument(
        "-f", "--features", type=int, nargs="?", default=32,
        help="Number of auto-encoded features"
    )
    parser.add_argument(
        "-ce", "--checkpoint-encoder", type=bool, nargs="?", default=False, const=True,
        help="load state_dict for encoder"
    )
    parser.add_argument(
        "-cd", "--checkpoint-decoder", type=bool, nargs="?", default=False, const=True,
        help="load state_dict for decoder"
    )
    parser.add_argument(
        "-d", "--device", type=str, nargs="?", default="cuda",
        help="Device to run on (cpu, cuda, cuda:0, ..)"
    )

    return parser.parse_args()


def main(args):

    data = ImageDataset(
        args.dataset,
        #"MNIST"
        #"FashionMNIST"
        #"CIFAR10"
        #"CIFAR100"
        #"STL10"
        resize=[args.resize, args.resize] if args.resize else None,
        bw=args.black_white,
    )
    # print(data.targets.shape)

    server = LittleServer(
        title="auto-encoder training"
    )
    server.start()

    t = Trainer(
        Encoder=get_module(args.encoder),
        Decoder=get_module(args.decoder),
        checkpoint_encoder=args.checkpoint_encoder,
        checkpoint_decoder=args.checkpoint_decoder,
        data=data,
        device=args.device,
        server=server,
        batch_size=args.batch_size,
        n_features=args.features,
    )
    try:
        t.train()
    finally:
        server.stop()


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        pass

