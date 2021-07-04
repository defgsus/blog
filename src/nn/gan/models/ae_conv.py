from .layers import *
from .ae import AutoEncoderBase


class EncoderConv(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(n_features, width, height, channels)

        n_feat = 64
        act = F.leaky_relu

        self.layers = nn.Sequential(
            ConvLayer(self.channels, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),
            ConvLayer(n_feat, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),
            ConvLayer(n_feat, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),
            ConvLayer(n_feat, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),
            ConvLayer(n_feat, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),

            nn.Flatten(),
            LinearLayer(n_feat*(width-10)*(height-10), self.n_features, act="tanh", batch_norm=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        #y = y / torch.norm(y, dim=-1, keepdim=True)
        return y

    def interesting_weights(self) -> Generator[torch.Tensor, None, None]:
        yield self.layers[0].conv.weight
        yield self.layers[1].conv.weight
        yield self.layers[2].conv.weight
        yield self.layers[3].conv.weight
        yield self.layers[4].conv.weight


class EncoderConv2(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(n_features, width, height, channels)

        n_feat = 64
        act = F.relu

        self.layers = nn.Sequential(
            # suppose 32
            ConvLayer(self.channels, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),
            # 30
            ConvLayer(n_feat, n_feat, 3, 1, 0, act=act, batch_norm=batch_norm),
            # 28
            ConvLayer(n_feat, n_feat, 5, 1, 0, act=act, batch_norm=batch_norm),
            # 24
            ConvLayer(n_feat, n_feat, 9, 1, 0, act=act, batch_norm=batch_norm),
            # 16
            ConvLayer(n_feat, n_feat, 9, 1, 0, act=act, batch_norm=batch_norm),
            # 8
            nn.Flatten(),
            LinearLayer(n_feat*(width-24)*(height-24), self.n_features, act="tanh", batch_norm=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        #y = y / torch.norm(y, dim=-1, keepdim=True)
        return y

    def interesting_weights(self) -> Generator[torch.Tensor, None, None]:
        yield self.layers[-3].conv.weight
        yield self.layers[-4].conv.weight


class DecoderConv(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int):
        super().__init__(n_features, width, height, channels)

        n_feat = 64

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.n_features, n_feat*2, 2, 1, 0, bias=False),
            nn.BatchNorm2d(n_feat*2),
            nn.LeakyReLU(inplace=True),
            # shape [N, n_feat, 2, 2]

            nn.ConvTranspose2d(n_feat*2, n_feat, 3, 1, 0, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(inplace=True),
            # shape [N, n_feat, 4, 4]

            nn.ConvTranspose2d(n_feat, n_feat, 5, 1, 0, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.LeakyReLU(inplace=True),
            # shape [N, channels, 8, 8]

            nn.ConvTranspose2d(n_feat, self.channels, 9, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.Tanh()
            # shape [N, channels, 16, 16]

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_features, 1, 1)
        y = self.layers(y)
        return y


class DecoderConv16(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(n_features, width, height, channels)

        n_feat = 64
        act = F.relu

        self.layers = nn.Sequential(
            ConvLayer(self.n_features, n_feat*2, 2, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape [N, n_feat, 2, 2]

            ConvLayer(n_feat*2, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape [N, n_feat, 4, 4]

            ConvLayer(n_feat, n_feat, 5, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape [N, n_feat, 8, 8]

            ConvLayer(n_feat, self.channels, 9, 1, 0, transpose=True, act="tanh", batch_norm=batch_norm),
            # shape [N, channels, 16, 16]

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_features, 1, 1)
        y = self.layers(y)
        return y


class DecoderConv16d(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(n_features, width, height, channels)

        n_feat = 256
        act = F.relu

        self.layers = nn.Sequential(
            ConvLayer(self.n_features, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 3x3
            ConvLayer(n_feat, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 5x5
            ConvLayer(n_feat, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 7x7
            ConvLayer(n_feat, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 9x9
            ConvLayer(n_feat, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 11x11
            ConvLayer(n_feat, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 13x13
            ConvLayer(n_feat, self.channels, 4, 1, 0, transpose=True, act="tanh", batch_norm=batch_norm),
            # shape 16x16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_features, 1, 1)
        y = self.layers(y)
        return y


class DecoderConv32(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(n_features, width, height, channels)

        n_feat = 32
        act = F.relu

        self.layers = nn.Sequential(
            ConvLayer(self.n_features, n_feat, 16, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 16x16
            ConvLayer(n_feat, n_feat, 9, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 24
            ConvLayer(n_feat, n_feat, 5, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 28
            ConvLayer(n_feat, n_feat, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 30
            ConvLayer(n_feat, self.channels, 3, 1, 0, transpose=True, act="tanh", batch_norm=batch_norm, drop_out=False),
            # shape 32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_features, 1, 1)
        y = self.layers(y)
        return y

    def interesting_weights(self) -> torch.Tensor:
        yield self.layers[0].conv.weight
        yield self.layers[1].conv.weight
        yield self.layers[2].conv.weight


class DecoderConv32R(AutoEncoderBase):

    def __init__(self, n_features: int, width: int, height: int, channels: int, batch_norm: bool = True):
        super().__init__(n_features, width, height, channels)

        n_feat = 16
        act = F.leaky_relu

        self.layers = nn.Sequential(
            ConvLayer(self.n_features, n_feat*4, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 3x3
            ConvLayer(n_feat*4, n_feat*3, 3, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 5x5
            ConvLayer(n_feat*3, n_feat*2, 5, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 9x9
            ConvLayer(n_feat*2, n_feat, 9, 1, 0, transpose=True, act=act, batch_norm=batch_norm),
            # shape 17
            ConvLayer(n_feat, self.channels, 16, 1, 0, transpose=True, act="tanh", batch_norm=batch_norm, drop_out=False),
            # shape 32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view(-1, self.n_features, 1, 1)
        y = self.layers(y)
        return y

    def interesting_weights(self) -> Generator[torch.Tensor, None, None]:
        yield self.layers[-1].conv.weight
        yield self.layers[-2].conv.weight
        yield self.layers[-3].conv.weight
