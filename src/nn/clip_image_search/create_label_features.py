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


def store_label_features(labels: List[str], feature_filename: str):
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
    return df


def generate_labels():
    things = [
        "CIA", "LSD",
        "Charles Manson", "Timothy Leary", "Bob Dobbs",
        "James Joyce", "Thomas Pynchon",
        "John F. Kennedy", "George Bush", "Barack Obama", "Donald Trump", "Angela Merkel",
        "cat", "dog", "whale", "spider", "penguin", "tree", "car", "people", "woman", "man", "kid",
        "radio", "computer", "super-human super-intelligent computer",
        "sign", "trumpet", "piano",
    ]
    adjectives = [
        "",
        "a huge gigantic", "a tiny little",
        "a scared", "a curious", "a happy", "a sad",
        "a naked", "a warm", "a cold", "a red", "a green", "a blue",
        "one", "two", "many",
    ]
    prefixes = [
        "",
        "photo of", "painting of", "drawing of", "sketch of", "voxel-graphic of",
        "statue of", "anatomy of", "description of",
        "dream of", "surrealistic version of",
        "close-up of", "rear-view of", "top-view of",
        "sphere of",
        "many",
    ]

    labels = []
    for thing in things:
        for adjective in adjectives:
            for prefix in prefixes:
                labels.append(f"{prefix}/{adjective}/{thing}")

    return labels


if __name__ == "__main__":

    print("exporting labels...")
    df = store_label_features(
        generate_labels(),
        "label-features-3"
    )
    print(df)
