from .layers import *


class DiscriminatorConv(nn.Module):

    def __init__(self, width: int, height: int, channels: int, batch_norm: bool = False):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels

        self.layers = nn.Sequential(
            ConvLayer(chan_in=self.channels, chan_out=32, kernel_size=3, padding=1, act="leaky_relu", batch_norm=batch_norm),
            #nn.MaxPool2d(kernel_size=2, padding=0),
            ConvLayer(chan_in=32, chan_out=32, kernel_size=5, padding=2, act="leaky_relu", batch_norm=batch_norm),
            nn.MaxPool2d(kernel_size=2, padding=0),
            ConvLayer(chan_in=32, chan_out=32, kernel_size=7, padding=3, act="leaky_relu", batch_norm=batch_norm),
            nn.MaxPool2d(kernel_size=2, padding=0),
        )

        self.out_size = 32 * self.width * self.height // 4 // 4
        self.l_out = LinearLayer(self.out_size, 1, torch.sigmoid, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1, self.channels, self.height, self.width)
        y = self.layers(y)
        y = y.reshape(-1, self.out_size)
        y = self.l_out(y)
        return y

