from .layers import *


class DiscriminatorLinear(nn.Module):

    def __init__(self, width: int, height: int, channels: int, batch_norm: bool = False):
        super().__init__()
        self.n_in = width * height * channels

        self.layers = nn.Sequential(
            LinearLayer(self.n_in, min(2000, self.n_in//2), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, self.n_in//2), min(2000, self.n_in//4), F.relu, batch_norm=batch_norm),
            LinearLayer(min(2000, self.n_in//4), 1, torch.sigmoid, batch_norm=batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
