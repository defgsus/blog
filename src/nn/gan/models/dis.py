from .layers import *


class DiscriminatorBase(nn.Module):

    def __init__(self, width: int, height: int, channels: int):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.n_in = width * height * channels


class DiscriminatorLinear(DiscriminatorBase):

    def __init__(self, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(width, height, channels)

        self.layers = nn.Sequential(
            LinearLayer(self.n_in, 1024, F.relu, batch_norm=batch_norm),
            LinearLayer(1024, 512, F.relu, batch_norm=batch_norm),
            LinearLayer(512, 1, torch.sigmoid, batch_norm=batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_in)
        return self.layers(y)


class DiscriminatorLinear1(nn.Module):

    def __init__(self, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__()
        self.n_in = width * height * channels

        self.layers = nn.Sequential(
            LinearLayer(self.n_in, min(2000, self.n_in//2), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, self.n_in//2), min(2000, self.n_in//4), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, self.n_in//4), 1, torch.sigmoid, batch_norm=batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_in)
        return self.layers(y)
