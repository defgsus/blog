import PIL.Image
import base64
from io import BytesIO
import json
from typing import Tuple, Generator
import sys
sys.path.insert(0, "..")

import numpy as np
from scipy import stats

from har import *


def load_image_base64(data: str):
    try:
        byte_data = base64.b64decode(data)
    except Exception as e:
        print("DEC ERR", e, data[:50])
        return
    try:
        file = BytesIO(byte_data)
        img = PIL.Image.open(file)
        return img
    except Exception as e:
        print("IMG ERR", e, data[:50])
        return


def iter_image_entries(
        glob_pattern: str,
        min_size: int = 0,
        max_size: int = 1e10,
) -> Generator[Tuple[dict, PIL.Image.Image], None, None]:
    for filename, first_entry, e in tqdm(iter_har_entries(glob_pattern), total=900000):
        mime = e["response"]["content"].get("mimeType")
        if mime and "image" in mime and "svg" not in mime:
            text = e["response"]["content"].get("text")
            encoding = e["response"]["content"].get("encoding")
            size = e["response"]["content"].get("size")
            if text and size and min_size <= size < max_size:
                if encoding != "base64":
                    print(e["response"])
                    raise ValueError("Does this really exist?")

                image = load_image_base64(text)
                if image:
                    yield e, image


def image_to_numpy(image: PIL.Image) -> Optional[np.ndarray]:
    try:
        pixels = np.asarray(image, dtype=np.float)
        if pixels.max() > 255:
            pixels //= 256
        return pixels
    except TypeError:
        return None


def get_image_entropy(image: np.ndarray):
    pixels = image.flatten()
    hist, _ = np.histogram(pixels.flatten(), bins=256)
    return stats.entropy(hist)


def build_size_histograms(glob_pattern: str):
    histograms = dict()
    try:
        for e, image in iter_image_entries(glob_pattern):

            key = e["request"]["host"]

            pixels = image_to_numpy(image)

            if key not in histograms:
                histograms[key] = {
                    "count": 0,
                    "error_count": 0,
                    "entropy": 0.,
                    "paths": set(),
                    "width": dict(),
                    "height": dict(),
                    "channels": dict(),
                    "mean_r": dict(),
                    "mean_g": dict(),
                    "mean_b": dict(),
                    "mean_a": dict(),
                }

            if pixels is None:
                histograms[key]["error_count"] += 1
                continue

            histograms[key]["count"] += 1
            histograms[key]["entropy"] += get_image_entropy(pixels)

            if len(histograms[key]["paths"]) < 10:
                path = e["request"]["path"]
                if e["request"]["params"]:
                    path = path + "?" + e["request"]["params"]
                histograms[key]["paths"].add(path)

            channels = 1
            mean_g, mean_b, mean_a = 0, 0, 0
            if pixels.ndim == 2:
                mean_r = mean_g = mean_b = int(pixels.mean())
            else:
                means = [int(m) for m in pixels.mean(axis=1).mean(axis=0)]
                mean_r = means[0]
                channels = len(means)
                try:
                    mean_g = means[1]
                    mean_b = means[2]
                    mean_a = means[3]
                except IndexError:
                    pass

            for hist_name, value in (
                    ("width", pixels.shape[1]),
                    ("height", pixels.shape[0]),
                    ("channels", channels),
                    ("mean_r", mean_r),
                    ("mean_g", mean_g),
                    ("mean_b", mean_b),
                    ("mean_a", mean_a),
            ):
                histogram = histograms[key]
                histogram[hist_name][value] = histogram[hist_name].get(value, 0) + 1

            #print(image.size, str(image.info)[:100], e["request"]["url"])
    except KeyboardInterrupt:
        pass

    for h in histograms.values():
        h["paths"] = sorted(h["paths"])
        if h["count"]:
            h["entropy"] /= h["count"]

    return histograms


if __name__ == "__main__":

    histograms_per_key = build_size_histograms("automatic/recordings/*/*.json")

    with open("image-histograms.json", "w") as fp:
        json.dump(histograms_per_key, fp)

    for key, histograms in histograms_per_key.items():
        print(key)
        for name, histo in histograms.items():
            print(f"  {name:10} {histo}")

