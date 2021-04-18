import os
import glob
import json
from typing import List
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "../../har_research")

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import clip
from PIL import Image

from har_research.image_scan import iter_image_entries


device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading model to", device)
model, preprocess = clip.load("ViT-B/32")


def store_label_features(labels: List[str], label_map: dict, feature_filename: str):
    label_tokens = clip.tokenize([
        l.replace("/", " ").replace("  ", " ").strip()
        for l in labels
    ]).to(device)
    with torch.no_grad():
        label_features = []
        while len(label_tokens):
            print(f"{min(1000, len(label_tokens))} labels...")
            label_features.append(
                model.encode_text(label_tokens[:1000]).cpu().numpy()
            )
            label_tokens = label_tokens[1000:]

    label_features = np.concatenate(label_features)
    df = pd.DataFrame(label_features)
    df.index = labels
    df.to_csv(f"{feature_filename}.csv")

    with open(f"{feature_filename}-map.json", "w") as fp:
        json.dump(label_map, fp)

    return df


def generate_labels():
    things = sorted([
        ("James Joyce", "James Joyces"),
        ("Charles Manson", "Charles Mansons"),
        ("John F. Kennedy", "John F. Kennedies"),
        ("George Bush", "George Bushes"),
        ("Barack Obama","Barack Obamas"),
        ("Donald Trump", "Donald Trumps"),
        ("Angela Merkel", "Angela Merkels"),
        ("cat", "cats"),
        ("dog", "dogs"),
        ("whale", "whales"),
        ("spider", "spiders"),
        ("penguin", "penguins"),
        ("tree", "trees"),
        ("car", "cars"),
        ("people", "people"),
        ("woman", "women"),
        ("man", "men"),
        ("kid", "kids"),
        ("radio", "radios"),
        ("computer", "computers"),
        ("sign", "signs"),
        ("trumpet", "trumpets"),
        ("food", "food"),
        ("death", "deaths"),
        ("alien", "aliens"),
    ], key=lambda l: (l[0].lower(), l[1].lower()))#[:5]
    adjectives = sorted([
        "red", "green", "blue", "yellow",
        "serious", "happy", "sad",
        "gigantic", "small",
        "warm", "cold", "wet",
        "psychedelic", "naked",
    ])
    numbers = sorted([
        ("a", 0),
        ("two", 1),
    ])
    prefixes = sorted([
        "photo of", "painting of", "drawing of", "3d-graphic of",
        "statue of", "anatomy of", "factual description of",
        "dream of", "surrealistic version of",
        "close-up of", "rear-view of", "top-view of",
        "sphere of",
    ])#[:3]

    labels = []
    for thing in things:
        for adjective in adjectives:
            for number in numbers:
                for prefix in prefixes:
                    labels.append(f"{prefix}/{number[0]}/{adjective}/{thing[number[1]]}")

    return labels, {
        "things": things,
        "adjectives": adjectives,
        "numbers": numbers,
        "prefixes": prefixes,
        "index": labels,
    }


if __name__ == "__main__":

    labels, label_map = generate_labels()
    for l in labels:
        print(l)
    print(len(labels), "labels")

    print("exporting labels...")
    df = store_label_features(
        labels, label_map,
        "label-features-4"
    )
    print(df)
