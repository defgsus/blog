import traceback
import argparse
import importlib
import time
from typing import Union, Sequence

import numpy as np
import torch
import torch.nn
from torchvision.transforms import Resize
import clip

import PIL.Image
from tqdm import tqdm

try:
    from .models.base import ImageGenBase, FixedBase, get_uv, to_image
except ImportError:
    from models.base import ImageGenBase, FixedBase, get_uv, to_image


torch.autograd.set_detect_anomaly(True)

device = "cuda"


def train_image_clip(
        model: ImageGenBase,
        clip_model: torch.nn.Module,
        expected_features: torch.Tensor,
        learnrate_scale: float = 1.,
        model_shape: Sequence = (48, 48),
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

    learnrate = 10.
    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=learnrate,
        weight_decay=0.000001,
        #momentum=0.9,
    )

    learnrate = .001
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learnrate,
        weight_decay=0.000001,
        #momentum=0.9,
    )

    learnrate *= learnrate_scale

    clip_shape = (224, 224)
    if model_shape is None:
        model_shape = clip_shape

    try:

        print("training:")
        print(f"  learnrate: {learnrate} (scale {learnrate_scale})")

        num_params = sum(
            sum(len(p) for p in g["params"])
            for g in optimizer.param_groups
        )
        print("  trainable params:", num_params)

        loss_function = (
            #torch.nn.L1Loss()
            torch.nn.MSELoss()
            #torch.nn.SmoothL1Loss()
        )
        last_print_time = time.time()
        last_snapshot_time = time.time()
        num_iter = 1000
        for epoch in tqdm(range(num_iter)):
            actual_learnrate = learnrate * min(1, epoch / 30. + .01) * (1. - 0.7 * epoch / num_iter)
            for g in optimizer.param_groups:
                g['lr'] = actual_learnrate

            output = model.render_tensor(model_shape)
            output = output + .1 * torch.randn(output.shape).to(device)

            output = output.permute(2, 0, 1)
            if model_shape != clip_shape:
                output = Resize(clip_shape)(output)

            image_features = clip_model.encode_image(output.unsqueeze(0))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = 100. * expected_features @ image_features.T
            #loss = loss_function(similarity[0], expected_similarity)
            loss = 100. * loss_function(image_features, expected_features)

            image_mean = output.mean(dim=1).mean(-1)
            loss += 0.03 * loss_function(image_mean, torch.Tensor([.45, .45, .45]).to(device))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            cur_time = time.time()
            if epoch % 150 == 0 or cur_time - last_print_time > 3:
                last_print_time = cur_time
                print(
                    "lr", round(actual_learnrate, 5),
                    "loss", round(float(loss), 3),
                    "sim", round(float(similarity), 3),
                    "mean", [round(float(t), 2) for t in image_mean],
                    "weights", model.weight_info() if hasattr(model, "weight_info") else "-",
                    #f"(img {round(float(image_loss), 3)}"
                    #f" param {round(float(parameter_loss), 3)})"
                )

            if epoch % 200 == 0 or cur_time - last_snapshot_time > 30:
                last_snapshot_time = cur_time
                print("writing snapshot.png")
                model.render_image(clip_shape).save("snapshot.png")

    except KeyboardInterrupt:
        pass
    except RuntimeError:
        traceback.print_exc()


def train_and_render(
        model: ImageGenBase,
        output_name: str,
        text: str = None,
        image_filename: str = None,
        learnrate_scale: float = 1.,
):
    print("loading clip model")
    clip_model, preprocess = clip.load("ViT-B/32")
    clip_model = clip_model.to(device)

    if text:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            clip_features = clip_model.encode_text(text_tokens)

    elif image_filename:
        image = PIL.Image.open(image_filename)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(image)

    else:
        raise ValueError("Need to supply 'text' or 'image_filename'")

    print("searching for these CLIP-features",
          (clip_features * 100).round().to(np.int))

    train_image_clip(
        model=model,
        clip_model=clip_model,
        expected_features=clip_features,
        learnrate_scale=learnrate_scale,
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
        help="dotted path to torch.nn.Module, e.g 'file.Model'",
    )
    parser.add_argument(
        "-lr", "--learnrate", type=float, default=1.,
        help="Learnrate scaling factor",
    )

    args = parser.parse_args()

    print("loading model")
    model = load_model(args.model)
    print("  params:", sum(len(p.flatten()) for p in model.parameters()))

    output_name = args.model.split(".")[-1]
    output_name = f"clip-{output_name}"

    train_and_render(
        model=model,
        learnrate_scale=args.learnrate,
        text=(
            #"a white wall"
            #"the face of a happy cat"
            #"a lot of creepy spiders"
            #"close-up of a huge spider"
            #"a street full of cars"
            #"the american flag"
            #"a blue sky"
            #"a fish underwater"
            #"the word love written on a wall"
            #"the letter f"
            #"a drawing of a house"
            #"a drawing of Bob Dobbs"
            #"a photo of a sunflower"
            #"a photo of a rose"
            #None
        ),
        image_filename=(
            #None
            #"/home/bergi/Pictures/__diverse/Annual-Sunflower.jpg"
            #"/home/bergi/Pictures/__diverse/MANSON14.JPG"
            #"/home/bergi/Pictures/__diverse/v2_at_peenemuende-usedom.jpg"
            #"/home/bergi/Pictures/__diverse/1139662681_f.jpg"  # Cartman
            #"/home/bergi/Pictures/__diverse/superman-superman-returns-1206769.jpg"
            "/home/bergi/Pictures/__diverse/marxsE80728FED671E8226833AF91A8B67.jpg"
        ),
        output_name=output_name,
    )
