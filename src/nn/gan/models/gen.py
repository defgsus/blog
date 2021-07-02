from .layers import *


class GeneratorBase(nn.Module):

    def __init__(self, n_in: int, width: int, height: int, channels: int):
        super().__init__()
        self.n_in = n_in
        self.n_out = width * height * channels
        self.width = width
        self.height = height
        self.channels = channels


class GeneratorLinear(GeneratorBase):

    def __init__(self, n_in: int, width: int, height: int, channels: int):
        super().__init__(n_in, width, height, channels)

        self.layers = nn.Sequential(
            LinearLayer(n_in, 512, F.leaky_relu, batch_norm=True, bias=False),
            # nn.Dropout(0.5),
            LinearLayer(512, 1024, F.leaky_relu, batch_norm=True, bias=False),
            # nn.Dropout(0.5),
            LinearLayer(1024, self.n_out, "tanh", batch_norm=True, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return y.view(-1, self.channels, self.height, self.width)
        #return torch.clamp(y, 0, 1)
