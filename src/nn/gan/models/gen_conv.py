from .layers import *


class GeneratorConv(nn.Module):

    def __init__(self, n_in: int, width: int, height: int, channels: int, batch_norm: bool = False):
        super().__init__()
        self.n_in = n_in
        self.width = width
        self.height = height
        self.channels = channels

        self.l_in = LinearLayer(self.n_in, 1*8*8, act="relu", bias=False, batch_norm=batch_norm)
        self.layers = nn.Sequential(
            ConvLayer(
                chan_in=1, chan_out=32, kernel_size=3, padding=0,
                act="leaky_relu", batch_norm=batch_norm, reverse=True
            ),  # 8+2
            ConvLayer(
                chan_in=32, chan_out=32, kernel_size=5, padding=0,
                act="leaky_relu", batch_norm=batch_norm, reverse=True
            ),  # 8+2+4
            ConvLayer(
                chan_in=32, chan_out=self.channels, kernel_size=[self.height-14+1, self.width-14+1], padding=0,
                act="tanh", batch_norm=batch_norm, reverse=True
            ),
        )

        self.out_size = self.channels * self.width * self.height
        print("generator:", self.layers, f"\n num_in={self.n_in} num_out={self.out_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.l_in(x)
        y = y.reshape(-1, 1, 8, 8)
        y = self.layers(y)
        y = y.reshape(-1, self.out_size)
        return y


class GeneratorConv32(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    def __init__(self, n_in: int, width: int, height: int, channels: int, batch_norm: bool = False):
        super().__init__()
        self.n_in = n_in
        self.width = width
        self.height = height
        self.channels = channels

        ngf = 64

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.n_in, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, self.channels, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.out_size = self.channels * self.width * self.height
        print("generator:", self.layers, f"\n num_in={self.n_in} num_out={self.out_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1, self.n_in, 1, 1)
        y = self.layers(y)
        y = y.reshape(-1, self.out_size)
        return y


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
        print("discriminator:", self.layers, f"\n num_out={self.out_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1, self.channels, self.height, self.width)
        y = self.layers(y)
        y = y.reshape(-1, self.out_size)
        y = self.l_out(y)
        return y
