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
    labels_tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        label_features = model.encode_text(labels_tokens).cpu().numpy()

    df = pd.DataFrame(label_features)
    df.index = labels
    df.to_csv(f"{feature_filename}.csv")
    return df


if __name__ == "__main__":

    print("exporting labels...")
    df = store_label_features(
        ["man", "woman", "dog", "face", "text", "people", "car", "hero"],
        "label-features"
    )
    print(df)
