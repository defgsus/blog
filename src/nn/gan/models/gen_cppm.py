from .layers import *


class GeneratorCPPMBase(nn.Module):

    def __init__(self, n_in: int, width: int, height: int, channels: int):
        super().__init__()
        self.n_in = n_in
        self.width = width
        self.height = height
        self.channels = channels
        self._coordinate_cache = None
        self._coordinate_cache_size = None

    def forward(
            self,
            x: torch.Tensor,
            width: Optional[int] = None,
            height: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate image from feature vector x.

        :param x: Tensor of shape [n_in] or [batch_size, n_in]
        :param width: optional width overrides the default setting
        :param height: optional height overrides the default setting
        :return: Tensor of shape [batch_size, channels, height, width]
        """
        width = self.width if width is None else width
        height = self.height if height is None else height

        if x.ndim == 1:
            x = x.unsqueeze(0)
            batch_size = 1
        elif x.ndim == 2:
            batch_size = x.shape[0]
        else:
            raise ValueError(f"noise expected to be of shape [batch_size, n_in], got {x}")

        if x.shape[-1] != self.n_in:
            raise ValueError(f"noise expected to be of shape [batch_size, n_in({self.n_in})], got {x}")

        input = torch.cat(
            [
                self.coordinates(width, height, x.device).reshape(1, -1, 2).expand(batch_size, -1, -1),
                # shape=[batch_size, width*height, 2]
                x.reshape(batch_size, 1, self.n_in).expand(-1, width*height, -1),
                # shape=[batch_size, width*height, n_in]
            ],
            dim=-1,
        )

        # shape=[batch_size, width*height, n_in+2]

        output = (
            self.network(input.reshape(-1, self.n_in+2))
            .reshape(batch_size, width*height, self.channels)
        )
        # shape=[batch_size, width*height, channels]

        return output.transpose(1, 2).reshape(batch_size, self.channels, height, width)

    def forward_pil(self, *args, **kwargs) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        out = self.forward(*args, **kwargs)
        images = [VF.to_pil_image(img) for img in out]
        return images[0] if len(images) == 1 else images

    def coordinates(self, width: int, height: int, device=None) -> torch.Tensor:
        if self._coordinate_cache_size != (width, height):
            xs = torch.linspace(-1, 1, width)
            ys = torch.linspace(-1, 1, height)
            if device is not None:
                xs = xs.to(device)
                ys = ys.to(device)
            self._coordinate_cache = torch.cat(
                [
                    xs.expand(height, -1).reshape(height, width, 1),
                    ys.reshape(-1, 1).expand(-1, width).reshape(-1, width, 1)
                ],
                dim=-1
            )
            self._coordinate_cache_size = (width, height)

        if self.training:
            return self._coordinate_cache + torch.randn(height, width, 2).to(self._coordinate_cache.device) / width
        else:
            return self._coordinate_cache


class GeneratorCPPM1(GeneratorCPPMBase):

    def __init__(self, n_in: int, width: int, height: int, channels: int):
        super().__init__(n_in=n_in, width=width, height=height, channels=channels)

        self.network = nn.Sequential(
            nn.Linear(n_in + 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Tanh(),

            nn.Linear(512, self.channels, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.Tanh(),
        )


class GeneratorCPPM2(GeneratorCPPMBase):

    def __init__(self, n_in: int, width: int, height: int, channels: int):
        super().__init__(n_in=n_in, width=width, height=height, channels=channels)

    #def network(self, torch.Tensor) -> torch.Tensor:
        self.network = nn.Sequential(
            nn.Linear(n_in + 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            SinusLayer(256, 10, learn_freq=False, learn_amp=False),
            nn.BatchNorm1d(256),
#            nn.Tanh(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),

            SinusLayer(256, 100, learn_freq=False, learn_amp=False),
            nn.BatchNorm1d(256),
#            nn.Tanh(),

            nn.Linear(256, self.channels),
            nn.BatchNorm1d(self.channels),
            nn.Tanh(),
        )
