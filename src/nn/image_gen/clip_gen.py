import math
import random
import traceback
import argparse
import time
from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
from torchvision.transforms import Resize, RandomAffine, GaussianBlur, RandomCrop, RandomPerspective
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import save_image
import clip

import PIL.Image
from tqdm import tqdm


torch.autograd.set_detect_anomaly(True)

device = "cuda"


class PixelsBase(torch.nn.Module):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__()
        self.resolution = resolution

    def save_image(self, filename: str):
        save_image(self.forward(), filename)

    def info_str(self) -> str:
        raise NotImplementedError


class PixelsRGB(PixelsBase):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__(resolution)
        self.pixels = torch.nn.Parameter(
            torch.rand((3, resolution[1], resolution[0])) * .05 + .475
        )
        self.saturation = torch.nn.Parameter(
            torch.rand((1, resolution[1], resolution[0])) * .05 + .475
        )

    def info_str(self) -> str:
        return f"mean rgbs " \
               f"{float(self.pixels[0].mean()):.3f}, " \
               f"{float(self.pixels[1].mean()):.3f}, " \
               f"{float(self.pixels[2].mean()):.3f}, " \
               f"{float(self.saturation[0].mean()):.3f}"

    def forward(self):
        pixels = self.pixels
        pixel_means = pixels.mean(0).repeat(3, 1, 1)
        pixels = pixels * self.saturation + (1. - self.saturation) * pixel_means
        return torch.clip(pixels, 0, 1)

    def blur(
            self,
            kernel_size: int = 3,
            sigma: Union[float, Tuple[float]] = 0.35,
    ):
        with torch.no_grad():
            pixels = self.pixels
            blurred_pixels = gaussian_blur(pixels, [kernel_size, kernel_size], [sigma, sigma])
            self.pixels[...] = blurred_pixels


class PixelsHSV(PixelsBase):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__(resolution)
        self.pixels = torch.nn.Parameter(
            torch.rand((3, resolution[1], resolution[0])) * .05 + .475
        )

    def info_str(self) -> str:
        return f"mean hsv " \
               f"{float(self.pixels[0].mean()):.3f}, " \
               f"{float(self.pixels[1].mean()):.3f}, " \
               f"{float(self.pixels[2].mean()):.3f}, "

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
            pixels = self.pixels
            #    pixels = hsv_to_rgb(pixels)
            blurred_pixels = gaussian_blur(pixels, [kernel_size, kernel_size], [sigma, sigma])
            blurred_pixels[:1, ...] = gaussian_blur(pixels[:1, ...], [kernel_size, kernel_size], [.1, .1])
            #blurred_pixels = rgb_to_hsv(torch.clamp(blurred_pixels, 0, 1))
            # blurred_pixels = torch.nan_to_num(blurred_pixels)
            self.pixels[...] = blurred_pixels


def train_image_clip(
        clip_model: torch.nn.Module,
        text_main: str,
        text_detail: Optional[Sequence[str]] = None,
        learnrate_scale: float = 1.,
        learnrate_scale_details: float = 1.,
        resolution: Sequence = (512, 512),
        num_epochs: int = 1000,
        detail_batch_size: int = 5,
):
    clip_resolution = (224, 224)

    # --- generate desired features ---

    if not text_detail:
        text_detail = [text_main]

    text_detail_matches = {
        i: 0 for i, t in enumerate(text_detail)
    }

    with torch.no_grad():
        text_tokens = clip.tokenize([text_main] + list(text_detail)).to(device)
        expected_features = clip_model.encode_text(text_tokens)

    expected_features /= expected_features.norm(dim=-1, keepdim=True)

    # --- setup pixel area ---

    pixel_model = PixelsRGB(resolution).to(device)

    # --- setup transformations ---

    full_transform = torch.nn.Sequential(
        RandomAffine(
            degrees=20,
            #scale=[.25, 1.],
            translate=(
                resolution[0] / clip_resolution[0] / resolution[0],
                resolution[1] / clip_resolution[1] / resolution[1],
            ),
        ),
        #RandomPerspective(distortion_scale=1., p=1.),
        Resize(clip_resolution),
        RandomCrop(size=clip_resolution),
        #RandomAffine(
        #    degrees=25,
        #),
        GaussianBlur(33, .8),
    )

    detail_transform = torch.nn.Sequential(
        RandomAffine(
            degrees=20.,
            scale=[.25, 1.],
            #shear=2.,
        ),
        #RandomPerspective(distortion_scale=1., p=1.),
        RandomCrop(size=clip_resolution),
        #GaussianBlur(101, .8),
    )

    # --- setup optimizer ---

    learnrate = .01
    optimizer = torch.optim.Adam(
        pixel_model.parameters(),
        lr=1,  # will be adjusted per epoch
        # weight_decay=0.01,
    )

    loss_function = (
        #torch.nn.L1Loss()
        torch.nn.MSELoss()
        #torch.nn.SmoothL1Loss()
    )

    # --- start training (breakable with CTRL-C) --

    main_similarity = 0.
    num_detail_frames = 0

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

            allow_detail_mode = True#detail_batch_size > 0 and epoch > num_epochs * .2
            detail_mode = epoch % 2 == 1 and allow_detail_mode
            #XXX
            if epoch > num_epochs * .0:
                detail_mode = True

            if detail_mode:
                num_detail_frames += 1

            # --- update learnrate ---

            epoch_f = epoch / num_epochs
            learnrate_factor = np.power(1. - 0.98 * epoch_f, .5)
            actual_learnrate = learnrate * learnrate_factor

            if detail_mode:
                actual_learnrate *= learnrate_scale_details  # / detail_batch_size
                actual_learnrate *= min(1, epoch_f / .2 + .01)
            else:
                actual_learnrate *= learnrate_scale

            for g in optimizer.param_groups:
                g['lr'] = actual_learnrate

            # --- feed pixels to CLIP ---

            current_pixels = pixel_model.forward()

            if not detail_mode:
                pixels = full_transform(current_pixels).unsqueeze(0)
            else:
                pixels = detail_transform(
                    current_pixels.unsqueeze(0).repeat((detail_batch_size, 1, 1, 1))
                )

            pixels = pixels + .1 * torch.randn(pixels.shape).to(device)

            image_features = clip_model.encode_image(pixels)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # -- get loss and train ---

            # shape = num_batch x num_features
            similarity = 100. * (expected_features @ image_features.T).T

            if not detail_mode:
                loss_features = expected_features[0].unsqueeze(0)
                loss = 100. * loss_function(image_features, loss_features)
                main_similarity = float(similarity[0][0])
            else:
                if 1:
                    loss_features = []
                    loss_feature_weights = []
                    for i in range(detail_batch_size):

                        if len(text_detail) > 1:
                            # choose detail-feature randomly by probability of match
                            if random.uniform(0, 1) >= .6:
                                best_match_indices = torch.argsort(similarity[i][1:], descending=True)
                                best_match_idx = len(best_match_indices) - 1
                                for j in range(2):
                                    if not best_match_idx:
                                        break
                                    best_match_idx = random.randint(0, best_match_idx)
                                best_match_idx = best_match_indices[best_match_idx]

                            # or choose best match
                            else:
                                best_match_idx = torch.argmax(similarity[i][1:])
                        else:
                            best_match_idx = 0

                        text_detail_matches[int(best_match_idx)] += 1

                        loss_features.append(
                            expected_features[best_match_idx+1].unsqueeze(0)
                        )
                        max_used = max(text_detail_matches.values())
                        loss_feature_weights.append(
                            max_used - text_detail_matches[int(best_match_idx)]
                        )

                    loss_features = torch.cat(loss_features, dim=0)
                    # raise impact by lower overall use
                    loss_feature_weights = torch.Tensor(loss_feature_weights).to(device)
                    loss_feature_weights = 1. + .1 * loss_feature_weights / (.0001 + loss_feature_weights.max())
                    loss_feature_weights = loss_feature_weights.reshape(-1, 1)

                    loss = 100. * loss_function(
                        image_features * loss_feature_weights,
                        loss_features * loss_feature_weights,
                    )

                    #if len(text_detail) > 1:
                    #    std_of_similarity = similarity[:, 1:].mean(0).std()
                    #    loss = loss + std_of_similarity

            # reduce mean saturation
            if isinstance(pixel_model, PixelsRGB):
                loss = loss + .01 * torch.abs(pixel_model.saturation[0].mean() - .2)

            # compress batch
            similarity = similarity.mean(0)

            pixel_model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # --- post-proc image --

            blur_amount = .28 + .07 * learnrate_factor
            if epoch % 3 == 0:
                pixel_model.blur(sigma=blur_amount)

            # --- print info ---

            cur_time = time.time()
            if epoch == 0 or cur_time - last_print_time > 3:
                last_print_time = cur_time
                print(
                    "lr", round(actual_learnrate, 5),
                    "loss", round(float(loss), 3),
                    "blur", round(blur_amount, 3),
                    "img", pixel_model.info_str(),
                )
                print(f"{('main: ' + text_main)[:40]:40} : sim {main_similarity:.3f}")
                if allow_detail_mode:
                    for i, text in enumerate(text_detail):
                        s = float(similarity[i+1])
                        count = text_detail_matches[i]
                        count_p = count / max(1, num_detail_frames * detail_batch_size) * 100.
                        print(f"{('detail: ' + text)[:40]:40} : sim {s:.3f} matches {count:4} ({count_p:.2f}%)"
                    )

            if epoch == 30 or cur_time - last_snapshot_time > 15:
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


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps (float, optional): scalar to enforce numarical stability. Default: 1e-6.

    Returns:
        torch.Tensor: HSV version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (v + eps)

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
        ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


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
    parser.add_argument(
        "-e", "--epochs", type=int, default=1000,
        help="Number of training steps, default = 1000",
    )
    parser.add_argument(
        "-r", "--resolution", type=int, default=[512], nargs="+",
        help="Resolution in pixels, can be one or two numbers, defaults to 512",
    )
    parser.add_argument(
        "-bsd", "--batch-size-detail", type=int, default=5,
        help="Number of detail optimizations per epoch",
    )

    args = parser.parse_args()

    if len(args.resolution) == 1:
        resolution = args.resolution * 2
    elif len(args.resolution) == 2:
        resolution = args.resolution
    else:
        print(f"Expecting one or two numbers for resolution, got {len(args.resolution)}")
        exit(0)

    print("loading CLIP")
    clip_model, preprocess = clip.load("ViT-B/32")

    train_image_clip(
        clip_model=clip_model,
        learnrate_scale=args.learnrate,
        learnrate_scale_details=args.learnrate_detail if args.learnrate_detail is not None else args.learnrate,
        num_epochs=args.epochs,
        resolution=resolution,
        detail_batch_size=args.batch_size_detail,
        text_main=(
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
            #"A photo of a wizard standing on a rock and casting a secret spell."
            #" Some giant birds fly by in amazement."
            #"a static background"
            #"a screw"
            #"a photo of two androids walking through a futuristic machine city. Some sky is visible."
            #"a low ceiling"
            #"front of abandoned building"
            #"front of abandoned building",
            #"cables and wires"
            #"forest landscape"
            #"a photo of a wooden table with little spiders crawling everywhere"
            #"malazan landscape"
            #"cthulhu"
            "x"
        ),
        text_detail=(
            "bob howard", "the laundry files",
            "cthulhu",
            #"malazan", #"lord of the rings",
            #"malazan landscape",
            #"cthulhu", #"stairs in a dungeon",
            #"rocks",
            #"leather armour", "metal armour",
            #"mountains",
            #"warrior", "sword"
            #"sky",
            #"sunflower leaves",
            #"close-up of sunflower leaves",
            #"forest",
            #"pine trees",
            #"garbage",
            #"litter",
            #"close-up of two androids looking up",
            #"close-up of various machinery parts",
            #"old store front",
            #"close-up of a spider on a wooden table",
            #"macro photography of a spider on a wooden table",
            #"macro photography a spider leg",
            #"macro photography of a spider head",
            #"close-up of wood furniture",
            #"macro photography of wood furniture",
            #"close-up of cables and wires",
            #"cables and wires",
            #"enlarged cables and wires",
            #"transistor between cables and wires",
            #"resistor between cables and wires",
            #"close-up of resistor",
            #"close-up of transistor",
            #"pipelines",
            #"tubings",
            # "thick wire",
            #"front of abandoned building",
            #"metallic wall",
            #"close-up of a sad robot",
            #"smoke and steam",
            #"splintered glass",
            #"skyscraper skyline",
            #"Drones are flying in the sky!",
            #"a cloudy sky",
            #"clouds",
            #"dystopian sky",
            #"underwater",
            #"fish scales",
            #"algae",
            #"a fish swarm",
            #"evil eyes",
            #"A close-up photo of a wizard standing on a rock and casting a secret spell.",
            #"a magical sky",
            #"a doomsday sky behind gigantic mountains",
            #"big mushrooms",
            #"rocks and grass",
            #"rocks",
            #"macro-photography of rocky surface",
            #"Some giant birds fly by in amazement.",
            #"rocky surface with a blue sky",
            #"a photo of a wizard",
            #"a photo of a wizard face",
            #"a magical face",
            # "close-up of a face",
            #"a huge nose",
            #"a huge mustache",
            #"a long wizard beard",
            #"a long flowing purple coat",
            #"a golden bell",
            #"a brightly glowing sphere",
            #"magic sparks",
            #"a red ball",
            #"a green ball",
            #"a blue ball",
        )
    )
