import time
import random
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchvision import datasets
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
import PIL.Image


class ImageDataset:

    ROOT = str(Path("~").expanduser() / "prog" / "data" / "datasets")

    def __init__(self, name: str, resize: Optional[List[int]] = None, bw: bool = False):
        self.name = name

        def _transform(p: PIL.Image.Image) -> torch.Tensor:
            t = VF.to_tensor(p)
            if bw and t.shape[-3] == 3:
                t = VF.rgb_to_grayscale(t)
            if resize:
                t = VF.resize(t, [resize[1], resize[0]])
            return t

        self.data = getattr(datasets, self.name)(
            root=self.ROOT,
            download=True,
            transform=_transform,
        )
        self.shape = self.data.data.shape
        self.num = self.shape[0]

        if self.name.startswith("CIFAR"):
            self.height, self.width = self.shape[-3:-1]
            self.channels = self.shape[-1] if len(self.shape) == 4 else 1
        else:
            self.height, self.width = self.shape[-2:]
            self.channels = self.shape[-3] if len(self.shape) == 4 else 1

        if resize:
            self.width, self.height = resize
        if bw:
            self.channels = 1

        print(f"dataset: {self.num} x {self.width}x{self.height}x{self.channels}")

    def random_samples(self, count: int) -> torch.Tensor:
        return torch.cat([
            self.data[random.randrange(len(self.data))][0].unsqueeze(0)
            for i in range(count)
        ], dim=0)

    def random_samples_with_labels(self, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = [
            self.data[random.randrange(len(self.data))]
            for i in range(count)
        ]
        return (
            torch.cat([s[0].unsqueeze(0) for s in samples], dim=0),
            torch.Tensor([s[1] for s in samples]),
        )
