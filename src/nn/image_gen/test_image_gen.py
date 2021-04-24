import argparse
import importlib
from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

import PIL.Image
from tqdm import tqdm


def get_uv(shape: Sequence[int], dimensions: int) -> torch.Tensor:
    """
    Create a 2d matrix of vectors of size `dimensions`,
    where first 2 dims are filled with numbers in the range [-1, 1]
    according to their positions.
    """
    space = torch.zeros((shape[0], shape[1], dimensions))
    for x in range(shape[1]):
        for y in range(shape[0]):
            space[y, x] = torch.Tensor(
                [x / (shape[1]-1), y / (shape[0]-1)] + [.5] * (dimensions - 2)
            )

    return (space - .5) * 2.


device = "cuda"


def to_image(tensor: Union[np.ndarray, torch.Tensor]) -> PIL.Image:
    """
    Convert [H, W, RGB] matrix to Pillow Image.
    Color-channels are clipped to the range [0, 1]
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    img = np.clip(tensor, 0, 1) * 255
    img = img.astype(np.uint8)
    return PIL.Image.fromarray(img)


def get_expected_image(shape: Sequence[int]) -> torch.Tensor:
    image = PIL.Image.open(
        "/home/bergi/Pictures/bob/Bobdobbs.png"
        #"/home/bergi/Pictures/__diverse/Annual-Sunflower.jpg"
        #"/home/bergi/Pictures/__diverse/Murdock.jpg"
        #"/home/bergi/Pictures/__diverse/honecker.jpg"
        #"/home/bergi/Pictures/__diverse/World_Wildlife_Fund_Logo.gif"
    )
    expected_image = PIL.Image.new("RGB", image.size)
    expected_image.paste(image)
    expected_image = expected_image.resize(shape, PIL.Image.LANCZOS)
    expected_image = np.asarray(expected_image).astype(np.float) / 255.
    expected_image = torch.Tensor(expected_image).to(device)
    return expected_image


def render_image(
        model: torch.nn.Module,
        shape: Sequence,
) -> torch.Tensor:
    with torch.no_grad():
        uv = get_uv(shape, 2).reshape(-1, 2).to(device)
        output = model(uv).reshape(shape[0], shape[1], 3)

    return output


def train_image_reproduce(
        model: torch.nn.Module,
        expected_image: torch.Tensor,
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
        expected_output = expected_image.reshape(-1, 3)
        input_positions = get_uv(expected_image.shape, 2).reshape(-1, 2).to(device)
        print("training batch:")
        print("  input:", input_positions.shape)
        print("  output:", expected_output.shape)

        loss_function = (
            #torch.nn.L1Loss()
            torch.nn.MSELoss()
            #torch.nn.SmoothL1Loss()
        )
        for epoch in tqdm(range(10000)):
            #for g in optimizer.param_groups:
            #    g['lr'] = epoch / 10000.

            output = model(input_positions)

            image_loss = loss_function(output, expected_output)

            loss = image_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 150 == 0:
                print(
                    "loss", round(float(loss), 3),
                    "weights", model.weight_info() if hasattr(model, "weight_info") else "-",
                    #f"(img {round(float(image_loss), 3)}"
                    #f" param {round(float(parameter_loss), 3)})"
                )

    except KeyboardInterrupt:
        pass


def train_and_render(
        model: torch.nn.Module,
        expected_image: torch.Tensor,
        output_name: str,
):
    train_image_reproduce(
        model=model, expected_image=expected_image
    )

    filename = f"./img/{output_name}.png"
    print(f"writing", filename)
    image = render_image(model, (512, 512))
    to_image(image).save(filename)


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
        help="dotted path to torch.nn.Module, e.g 'file.Model'",
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
        output_name=args.model.replace(".", "-"),
    )
