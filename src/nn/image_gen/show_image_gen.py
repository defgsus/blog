import argparse
import importlib
import time
from typing import Union, Sequence, Tuple, Type

import numpy as np
import torch
import torch.nn
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

import PIL.Image
from tqdm import tqdm

try:
    from .models.base import ImageGenBase, get_uv, to_image
except ImportError:
    from models.base import ImageGenBase, get_uv, to_image


device = "cuda"


def get_expected_image(shape: Sequence[int]) -> torch.Tensor:
    image = PIL.Image.open(
        #"/home/bergi/Pictures/bob/Bobdobbs.png"
        #"/home/bergi/Pictures/__diverse/Annual-Sunflower.jpg"
        #"/home/bergi/Pictures/__diverse/Murdock.jpg"
        "/home/bergi/Pictures/__diverse/honecker.jpg"
        #"/home/bergi/Pictures/__diverse/World_Wildlife_Fund_Logo.gif"
    )
    expected_image = PIL.Image.new("RGB", image.size)
    expected_image.paste(image)
    expected_image = expected_image.resize(shape, PIL.Image.LANCZOS)
    expected_image = np.asarray(expected_image).astype(np.float) / 255.
    expected_image = torch.Tensor(expected_image).to(device)
    return expected_image


def render_image_gen(
        Model: Type[ImageGenBase],
        output_name: str,
        number: Tuple[int, int] = (9, 6),
        resolution: Tuple[int, int] = (64, 64),
):
    """
    Render a number of freshly initialized ImageGen pictures
    :param Model:
        torch.nn.Model that has
            - 2 inputs: position [-1, 1]
            - 3 outputs: color channels [0, 1]
    :param number:
        tuple of number-x and number-y
    """
    input_positions = get_uv((resolution[1], resolution[0]), 2).reshape(-1, 2).to(device)

    images = []
    for epoch in tqdm(range(number[0] * number[1])):

        model = Model().to(device)
        if epoch == 0:
            print(Model.__name__, "params:", sum(len(p.flatten()) for p in model.parameters()))

        with torch.no_grad():
            output = model(input_positions)

            # convert to torchvision-style [3, H, W]
            image = output.reshape((resolution[1], resolution[0], 3)).permute(2, 0, 1)
            images.append(image)

        del model

    image_grid = make_grid(
        images,
        nrow=number[0],
    )

    filename = f"img/init-grid-{output_name}.png"
    print("writing", filename)
    ToPILImage()(image_grid).save(filename)


def get_model_class(path: str) -> Type[torch.nn.Module]:
    path = path.split(".")
    module_name = ".".join(path[:-1])
    model_name = path[-1]

    module = importlib.import_module(module_name)

    Model = getattr(module, model_name)
    return Model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str,
        help="dotted path to torch.nn.Module, e.g 'package.file.Model'",
    )

    args = parser.parse_args()

    Model = get_model_class(args.model)

    render_image_gen(
        Model=Model,
        output_name=args.model.split(".")[-1],
    )
