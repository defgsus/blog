import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode", type=str,
        help="Type of source 'har', or 'path'"
    )

    return parser.parse_args()



device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading model to", device)
model, preprocess = clip.load("ViT-B/32")


def store_har_image_features(
        glob_pattern: str,
        feature_filename: str,
        batch_size: int = 50,
):
    filename_keys = dict()

    def _process_batch(info_batch, image_batch):
        image_batch = torch.stack(image_batch).to(device)

        with torch.no_grad():
            feature_batch = model.encode_image(image_batch).cpu().numpy()

        for (filename, e), features in zip(info_batch, feature_batch):
            if filename not in filename_keys:
                filename_keys[filename] = len(filename_keys)
                fp.write(json.dumps({
                    "filename_key": filename_keys[filename],
                    "filename": filename,
                }) + "\n")

            line = '{"f":%s,"url":"%s","features":[%s]}\n' % (
                filename_keys[filename],
                e["request"]["url"],
                ",".join(str(f) for f in features)
            )
            fp.write(line)

    image_batch = []
    info_batch = []
    urls_set = set()
    num_exported = 0
    with open(f"{feature_filename}.ndjson", "w") as fp:
        for filename, e, image in iter_image_entries(glob_pattern):
            if image.size[1] < 50:
                continue
            if image.size[0] > 3 * image.size[1] or image.size[1] > 3 * image.size[0]:
                continue

            if e["request"]["url"] in urls_set:
                continue
            urls_set.add(e["request"]["url"])

            try:
                image = preprocess(image)
            except Exception as ex:
                print(e["request"]["url"], ":", ex)
                continue

            image_batch.append(image)
            info_batch.append((filename, e))

            if len(info_batch) >= batch_size:
                _process_batch(info_batch, image_batch)
                num_exported += len(info_batch)
                image_batch = []
                info_batch = []

        if info_batch:
            _process_batch(info_batch, image_batch)
            num_exported += len(info_batch)

    return num_exported



if __name__ == "__main__":

    print("scanning images...")
    num_exported = store_har_image_features(
        "../../har_research/automatic/recordings/*/*.json",
        "web-image-features"
    )
    print("num images exported:", num_exported)

