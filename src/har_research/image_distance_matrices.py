import math
import json
from typing import List
from multiprocessing import Pool
import sys
sys.path.insert(0, "..")

from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE

from lib.cache import FileCache


def log(*args, file=None, **kwargs):
    print(*args, **kwargs, file=file or sys.stderr)


def filter_histograms(histograms: dict, min_count: int = 0):
    f_histograms = dict()
    for key, histos in histograms.items():
        if histos["count"] < min_count:
            continue
        f_histograms[key] = histos
    return f_histograms


def bergi_value_histogram_distance(h1: dict, h2: dict) -> float:
    h1_max = max(h1.values())
    h2_max = max(h2.values())
    keys1 = sorted(h1.keys(), key=lambda k: -h1[k])
    keys2 = sorted(h2.keys(), key=lambda k: -h2[k])
    dist = 0.
    for k1, k2 in zip(keys1, keys2):
        dist += abs(k1 - k2) * (h1[k1] / h1_max + h2[k2] / h2_max)
    for k in set(keys1) - set(keys2):
        dist += k * h1[k] / h1_max
    for k in set(keys2) - set(keys1):
        dist += k * h2[k] / h2_max
    return dist


def get_distance_matrix(
        histograms: dict,
        labels: List[str],
        field: str,
        read_cache: bool = True,
        write_cache: bool = True,
        cache_suffix: str = "",
) -> np.ndarray:
    cache_name = f"distance-matrix/{FileCache.repr_hash(labels + [field])}{cache_suffix}.npy"

    if read_cache:
        if FileCache.exists(cache_name):
            log("loading", FileCache.filename(cache_name))
            try:
                return np.load(FileCache.filename(cache_name))
            except Exception as e:
                log("cache load error:", e)
                pass
        else:
            log("no cache file:", FileCache.filename(cache_name))

    distances = np.ndarray((len(labels), len(labels)))
    for i in tqdm(range(len(labels))):
        h1 = histograms[labels[i]][field]
        for j in range(i+1, len(labels)):
            h2 = histograms[labels[j]][field]
            if isinstance(h1, dict):
                dist = bergi_value_histogram_distance(h1, h2)
            else:
                dist = abs(h1 - h2)
            distances[i][j] = distances[j][i] = dist

    if write_cache:
        log("writing", FileCache.filename(cache_name))
        FileCache.make_path(cache_name)
        np.save(FileCache.filename(cache_name), distances)

    return distances


if __name__ == "__main__":
    with open("image-histograms.json") as fp:
        histograms_raw = json.load(fp)
    log(len(histograms_raw), "hosts")

    # convert histogram string indices back to int
    for key in histograms_raw:
        for hkey, hist in histograms_raw[key].items():
            if isinstance(hist, dict):
                histograms_raw[key][hkey] = {int(k): v for k, v in histograms_raw[key][hkey].items()}

    histograms = filter_histograms(histograms_raw, min_count=5)
    log(len(histograms), "hosts after filter")

    fields = ["width", "height", "mean_r", "mean_g", "mean_b", "mean_a", "channels", "entropy"]

    def _get_distance_matrix(field: str) -> np.ndarray:
        return get_distance_matrix(histograms, sorted(histograms), field, cache_suffix="-5")

    pool = Pool()
    distances = pool.map(_get_distance_matrix, fields)

    # print(distances[0])
