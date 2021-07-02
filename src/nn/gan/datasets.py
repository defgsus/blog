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

    def __init__(self, name: str, resize: Optional[List[int]] = [16, 16]):
        self.name = name

        transform = VF.to_tensor
        if resize:
            def _transform(p: PIL.Image.Image) -> torch.Tensor:
                return VF.resize(VF.to_tensor(p), [resize[1], resize[0]])
            transform = _transform

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

        if resize:
            self.width, self.height = resize

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
