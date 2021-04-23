from typing import Union, Sequence

import numpy as np
import torch
import torch.nn

import PIL.Image
from tqdm import tqdm

try:
    from .kaliset import get_uv, KaliSetModule
    from .model1 import Model1
except ImportError:
    from kaliset import get_uv, KaliSetModule
    from model1 import Model1


device = "cuda"


def to_image(tensor: Union[np.ndarray, torch.Tensor]) -> PIL.Image:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    img = np.clip(tensor, 0, 1) * 255
    img = img.astype(np.uint8)
    return PIL.Image.fromarray(img)


def get_expected_image(resolution: int) -> torch.Tensor:
    image = PIL.Image.open(
        "/home/bergi/Pictures/bob/Bobdobbs.png"
        #"/home/bergi/Pictures/__diverse/Annual-Sunflower.jpg"
        #"/home/bergi/Pictures/__diverse/honecker.jpg"
        #"/home/bergi/Pictures/__diverse/World_Wildlife_Fund_Logo.gif"
    )
    expected_image = PIL.Image.new("RGB", image.size)
    expected_image.paste(image)
    expected_image = expected_image.resize((resolution, resolution), PIL.Image.LANCZOS)
    expected_image = np.asarray(expected_image).astype(np.float) / 255.
    expected_image = torch.Tensor(expected_image).to(device)
    return expected_image


def train_image(
        model: torch.nn.Module,
        expected_image: torch.Tensor,
        output_name: str,
):
    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=0.92,
        weight_decay=0.01,
        #momentum=0.9,
    )

    try:
        for epoch in tqdm(range(100000)):
            output = model()

            image_loss = torch.abs(expected_image - output).flatten().sum() / (model._resolution * model._resolution)
            parameter_loss = torch.abs(model.kali_parameters - .8).flatten().sum() / model._dimensions

            loss = image_loss + .01 * parameter_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 150 == 0:
                print(
                    "loss", round(float(loss), 3),
                    f"(img {round(float(image_loss), 3)}" 
                    f" param {round(float(parameter_loss), 3)})"
                )

    except KeyboardInterrupt:
        pass

    with torch.no_grad():
        output = model()

    to_image(output).save(f"./{output_name}.png")

    if 0:
        params = model.start_space.detach().cpu().numpy()
        params = (params - np.min(params)) / (np.max(params) - np.min(params))
        to_image(params).save(f"./{output_name}-params.png")

        #model.start_space[:,:] = get_uv(model._resolution, model._dimensions)
        with torch.no_grad():
            model.start_space[:,:] = model.start_space * .5 + .5 * get_uv(model._resolution, model._dimensions).to(device)
            output = model()
            to_image(output).save("./kaliset-unspaced.png")

    for p in model.parameters():
        if p.shape[0] < 50:
            print(p)


def test_kaliset():
    SIZE = 128
    expected_image = get_expected_image(SIZE)

    model = KaliSetModule(
        resolution=SIZE,
        num_iterations=17,
        #zoom=.1,
        position=[1., 1., .5],
        #learn_zoom=False,
        #learn_position=False,
        #learn_parameters=False,
    ).to(device)

    train_image(model, expected_image, "kaliset2")


def test_model1():
    SIZE = 128
    expected_image = get_expected_image(SIZE)

    model = Model1().to(device)

    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=.5,
        weight_decay=0.001,
        #momentum=0.9,
    )

    try:
        pos = get_uv(SIZE, 2).reshape(SIZE*SIZE, 2).to(device)
        expected_output = expected_image.reshape(SIZE*SIZE, 3)
        for epoch in tqdm(range(10000)):

            #pos = torch.rand(2)
            #pix_pos = torch.minimum(pos * SIZE, torch.Tensor([SIZE-1, SIZE-1])).to(torch.int)
            #expected_pixel = expected_image[pix_pos[0], pix_pos[1]]

            output = model(pos)

            image_loss = torch.abs(expected_output - output).sum() / (SIZE * SIZE * 3)

            loss = image_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 150 == 0:
                print(
                    "loss", round(float(loss), 3),
                    "weights", model.weight_info(),
                    #f"(img {round(float(image_loss), 3)}"
                    #f" param {round(float(parameter_loss), 3)})"
                )

    except KeyboardInterrupt:
        pass

    with torch.no_grad():
        uv = get_uv(SIZE, 2).to(device)
        output = model(uv)

    to_image(output).save(f"./model1.png")


if __name__ == "__main__":

    #test_kaliset()
    test_model1()
