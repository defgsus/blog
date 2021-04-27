import math
import traceback
import argparse
import time
from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
from torchvision.transforms import Resize, RandomAffine, GaussianBlur, RandomCrop
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import save_image
import clip

import PIL.Image
from tqdm import tqdm


torch.autograd.set_detect_anomaly(True)

device = "cuda"


class Pixels(torch.nn.Module):

    def __init__(
            self,
            resolution: Sequence[int],
            hsv: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.hsv = hsv
        self.pixels = torch.nn.Parameter(
            torch.rand((3, resolution[1], resolution[0]))
        )

    def forward(self):
        pixels = self.pixels
        if self.hsv:
            pixels = hsv_to_rgb(pixels)
        return pixels

    def blur(
            self,
            kernel_size: int = 3,
            sigma: Union[float, Tuple[float]] = 0.35,
    ):
        with torch.no_grad():
            blurred_pixels = gaussian_blur(self.pixels, [kernel_size, kernel_size], [sigma, sigma])
            self.pixels[:, :, :] = blurred_pixels

    def save_image(self, filename: str):
        save_image(self.forward(), filename)


def train_image_clip(
        clip_model: torch.nn.Module,
        text: str,
        text_detail: Optional[str] = None,
        learnrate_scale: float = 1.,
        learnrate_scale_details: float = 1.,
        resolution: Sequence = (512, 512),
        num_epochs: int = 1000,
):
    clip_resolution = (224, 224)

    # --- generate desired features ---

    if not text_detail:
        text_detail = text

    with torch.no_grad():
        text_tokens = clip.tokenize([text, text_detail]).to(device)
        expected_features = clip_model.encode_text(text_tokens)

    expected_features /= expected_features.norm(dim=-1, keepdim=True)

    # --- setup pixel area ---

    pixel_model = Pixels(resolution).to(device)

    # --- setup transformations ---

    full_transform = torch.nn.Sequential(
        RandomAffine(
            degrees=0,
            translate=(1. / resolution[0], 1. / resolution[1]),
        ),
        Resize(clip_resolution),
        RandomAffine(
            degrees=25,
        ),
        GaussianBlur(127, .8),
    )

    detail_transform = torch.nn.Sequential(
        RandomAffine(
            degrees=25,
        ),
        RandomCrop(
            size=clip_resolution,
        ),
        #GaussianBlur(127, .8),
    )

    # --- setup optimizer ---

    learnrate = .001
    optimizer = torch.optim.Adam(
        pixel_model.parameters(),
        lr=1,  # will be adjusted per epoch
        weight_decay=0.000001,
        #momentum=0.9,
    )

    loss_function = (
        #torch.nn.L1Loss()
        torch.nn.MSELoss()
        #torch.nn.SmoothL1Loss()
    )

    # --- start training (breakable with CTRL-C) --

    try:
        print("training:")
        print(f"  learnrate: {learnrate} (scale full={learnrate_scale}, details={learnrate_scale_details})")

        num_params = sum(
            sum(len(p) for p in g["params"])
            for g in optimizer.param_groups
        )
        print("  trainable params:", num_params)

        last_print_time = time.time()
        last_snapshot_time = time.time()
        for epoch in tqdm(range(num_epochs)):

            detail_mode = epoch % 2 == 1
            final_phase = epoch >= num_epochs * 0.99

            # --- update learnrate ---

            epoch_f = np.power(epoch / num_epochs, .5)
            actual_learnrate = learnrate * min(1, epoch / 30. + .01) * (1. - 0.98 * epoch_f)
            actual_learnrate *= learnrate_scale_details if detail_mode else learnrate_scale
            for g in optimizer.param_groups:
                g['lr'] = actual_learnrate

            # --- feed pixels to CLIP ---

            pixels = pixel_model.forward()

            if detail_mode:
                pixels = detail_transform(pixels)
            else:
                pixels = full_transform(pixels)

            pixels = pixels + .1 * torch.randn(pixels.shape).to(device)

            image_features = clip_model.encode_image(pixels.unsqueeze(0))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # -- get loss and train ---

            similarity = 100. * expected_features @ image_features.T

            if detail_mode:
                expected_loss_features = expected_features[:1, :]
            else:
                expected_loss_features = expected_features[1:, :]

            loss = 100. * loss_function(image_features, expected_loss_features)

            #if detail_mode:
            #    loss = loss * .1
            #image_mean = output.mean(dim=1).mean(-1)
            #loss += 0.03 * loss_function(image_mean, torch.Tensor([.45, .45, .45]).to(device))

            pixel_model.zero_grad()
            loss.backward()
            optimizer.step()

            # --- post-proc image --

            if not final_phase:
                pixel_model.blur()

            # --- print info ---

            cur_time = time.time()
            if epoch == 0 or cur_time - last_print_time > 3:
                last_print_time = cur_time
                print(
                    "lr", round(actual_learnrate, 5),
                    "loss", round(float(loss), 3),
                    "sim", [round(float(s), 3) for s in similarity],
                )

            if epoch == 30 or cur_time - last_snapshot_time > 20:
                last_snapshot_time = cur_time

                print("writing snapshot.png")
                pixel_model.save_image("img-new/snapshot.png")

    except KeyboardInterrupt:
        pass
    except RuntimeError:
        traceback.print_exc()

    print("writing snapshot.png")
    pixel_model.save_image("img-new/snapshot.png")


# from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html
def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :]  # / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.).to(image.device)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((
        v, q, p, p, t, v,
        t, v, v, q, p, p,
        p, p, t, v, v, q,
    ), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lr", "--learnrate", type=float, default=1.,
        help="Learnrate scaling factor",
    )
    parser.add_argument(
        "-lrd", "--learnrate-detail", type=float, default=None,
        help="Learnrate scaling factor for details, defaults to --learnrate",
    )

    args = parser.parse_args()

    clip_model, preprocess = clip.load("ViT-B/32")

    train_image_clip(
        clip_model=clip_model,
        learnrate_scale=args.learnrate,
        learnrate_scale_details=args.learnrate_detail if args.learnrate_detail is not None else args.learnrate,
        text=(
            #"a white wall"
            #"the face of a happy cat"
            #"a lot of creepy spiders"
            #"close-up of a huge spider"
            #"a street full of cars"
            #"the american flag"
            #"a blue sky"
            #"porn"
            #"a fish underwater"
            #"the word love written on a wall"
            #"the letter f"
            #"a drawing of a house"
            #"a drawing of Bob Dobbs"
            #"a photo of a sunflower"
            #"a photo of a rose"
            #"a photo of a v2 rocket standing on the ground on a meadow.
            #"Trees are visible in the background. "
            #"The rocket is on the edge of the photo"
            #"A photo of a beautiful meadow. The sun is shining and two tentacles are passing by."
            #" Skyscrapers are visible in the background."
            #" The sky is blue and full of flying tentacles."
            "A photo of a wizard standing on a rock and casting a secret spell."
            " Some giant birds fly by in amazement."
            #None
        ),
        text_detail=(
            #"Sky and landscape"
        )
    )
