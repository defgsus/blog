from .layers import *


class AutoEncoderBase(nn.Module):

    def __init__(self, n_features: int, width: int, height: int, channels: int):
        super().__init__()
        self.n_features = n_features
        self.n_pixels = width * height * channels
        self.width = width
        self.height = height
        self.channels = channels


class EncoderLinear(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int):
        super().__init__(n_features, width, height, channels)

        self.layers = nn.Sequential(
            LinearLayer(self.n_pixels, 1024, F.leaky_relu, batch_norm=True),
            # nn.Dropout(0.5),
            LinearLayer(1024, 512, F.leaky_relu, batch_norm=True),
            # nn.Dropout(0.5),
            LinearLayer(512, self.n_features, "tanh", batch_norm=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_pixels)
        return self.layers(y)

    def interesting_weights(self) -> Generator[torch.Tensor, None, None]:
        yield self.layers[0].weight.view(self.channels, self.height, self.width, 1024)


class DecoderLinear(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int):
        super().__init__(n_features, width, height, channels)

        self.layers = nn.Sequential(
            LinearLayer(self.n_features, 512, F.leaky_relu, batch_norm=True),
            # nn.Dropout(0.5),
            LinearLayer(512, 1024, F.leaky_relu, batch_norm=True),
            # nn.Dropout(0.5),
            LinearLayer(1024, self.n_pixels, "tanh", batch_norm=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return y.view(-1, self.channels, self.height, self.width)
