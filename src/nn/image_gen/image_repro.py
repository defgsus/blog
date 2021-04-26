import argparse
import importlib
import time
from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

import PIL.Image
from tqdm import tqdm

try:
    from .models.base import ImageGenBase, FixedBase, get_uv, to_image
except ImportError:
    from models.base import ImageGenBase, FixedBase, get_uv, to_image


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


def train_image_reproduce(
        model: ImageGenBase,
        expected_image: torch.Tensor,
        model_shape: Sequence[int] = (128, 128)
):
    """
    Trains the model to reproduce the image.
    :param model:
        torch.nn.Model that has
            - 2 inputs: position [-1, 1]
            - 3 outputs: color channels [0, 1]
    :param expected_image:
        A torch.Tensor of shape [H, W, 3] to reproduce
    """
    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=1.,
        weight_decay=0.0001,
        #momentum=0.9,
    )

    try:
        expected_output = expected_image#.reshape(-1, 3)
        print("training:")
        print("  output:", expected_output.shape)
        print("  trainable params:", sum(
            sum(len(p) for p in g["params"])
            for g in optimizer.param_groups
        ))

        loss_function = (
            #torch.nn.L1Loss()
            torch.nn.MSELoss()
            #torch.nn.SmoothL1Loss()
        )
        last_print_time = time.time()
        for epoch in tqdm(range(10000)):
            #for g in optimizer.param_groups:
            #    g['lr'] = epoch / 10000.

            output = model.render_tensor(model_shape)

            image_loss = loss_function(output, expected_output)

            loss = image_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

            cur_time = time.time()
            if epoch % 150 == 0 or cur_time - last_print_time > 3:
                last_print_time = cur_time
                print(
                    "loss", round(float(loss), 3),
                    "weights", model.weight_info() if hasattr(model, "weight_info") else "-",
                    #f"(img {round(float(image_loss), 3)}"
                    #f" param {round(float(parameter_loss), 3)})"
                )

    except KeyboardInterrupt:
        pass


def train_and_render(
        model: ImageGenBase,
        expected_image: torch.Tensor,
        output_name: str,
):
    train_image_reproduce(
        model=model, expected_image=expected_image
    )

    filename = f"./img/{output_name}.png"
    print(f"writing", filename)
    model.render_image((512, 512)).save(filename)


def load_model(path: str) -> torch.nn.Module:
    path = path.split(".")
    module_name = ".".join(path[:-1])
    model_name = path[-1]

    module = importlib.import_module(module_name)

    Model = getattr(module, model_name)
    return Model().to(device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str,
        help="dotted path to torch.nn.Module, e.g 'package.file.Model'",
    )

    args = parser.parse_args()

    print("loading image")
    expected_image = get_expected_image((128, 128))
    print(" ", expected_image.shape)
    print("loading model")
    model = load_model(args.model)
    print("  params:", sum(len(p.flatten()) for p in model.parameters()))

    train_and_render(
        model=model,
        expected_image=expected_image,
        output_name=args.model.split(".")[-1],
    )
