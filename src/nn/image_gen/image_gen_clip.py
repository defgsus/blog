import argparse
import importlib
from typing import Union, Sequence

import numpy as np
import torch
import torch.nn
import clip

import PIL.Image
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)


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


def render_image(
        model: torch.nn.Module,
        shape: Sequence,
) -> torch.Tensor:
    with torch.no_grad():
        uv = get_uv(shape, 2).reshape(-1, 2).to(device)
        output = model(uv).reshape(shape[0], shape[1], 3)

    return output


def train_image_clip(
        model: torch.nn.Module,
        clip_model: torch.nn.Module,
        clip_preprocessor: torch.nn.Module,
        expected_features: torch.Tensor,
):
    """
    Trains the model to reproduce the image.
    :param model:
        torch.nn.Model that has
            - 2 inputs: position [-1, 1]
            - 3 outputs: color channels [0, 1]
    :param expected_features:
        A 512 dim CLIP feature vector
    """
    expected_features /= expected_features.norm(dim=-1, keepdim=True)

    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=1.,
        weight_decay=0.0001,
        #momentum=0.9,
    )

    try:
        expected_similarity = torch.Tensor([1]).to(device)
        input_positions = get_uv((224, 224), 2).reshape(-1, 2).to(device)
        print("training batch:")
        print("  input:", input_positions.shape)

        loss_function = (
            torch.nn.L1Loss()
            #torch.nn.MSELoss()
            #torch.nn.SmoothL1Loss()
        )
        for epoch in tqdm(range(2000)):
            #for g in optimizer.param_groups:
            #    g['lr'] = epoch / 10000.

            output = model(input_positions)
            image = output.reshape(224, 224, 3).permute(2, 1, 0).unsqueeze(0)

            #with torch.no_grad():
            image_features = clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = expected_features @ image_features.T
            loss = loss_function(similarity[0], expected_similarity)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 15 == 0:
                print(
                    "loss", round(float(loss), 3),
                    "weights", model.weight_info() if hasattr(model, "weight_info") else "-",
                    #f"(img {round(float(image_loss), 3)}"
                    #f" param {round(float(parameter_loss), 3)})"
                )

            if epoch % 200 == 1:
                print("writing snapshot.png")
                snapshot = render_image(model, (128, 128))
                to_image(snapshot).save("snapshot.png")

    except (KeyboardInterrupt, RuntimeError):
        pass


def train_and_render(
        model: torch.nn.Module,
        text: str,
        output_name: str,
):
    print("loading clip model")
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model = clip_model.to(device)

    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_features = clip_model.encode_text(text_tokens)

    print("searching for CLIP-features", text_features)

    train_image_clip(
        model=model,
        clip_model=clip_model,
        clip_preprocessor=preprocess,
        expected_features=text_features,
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

    print("loading model")
    model = load_model(args.model)
    print("  params:", sum(len(p.flatten()) for p in model.parameters()))

    output_name = args.model.split(".")[-1]
    output_name = f"clip-{output_name}"

    train_and_render(
        model=model,
        text="a yellow sphere",
        output_name=output_name,
    )
