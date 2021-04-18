import json
import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
from tqdm import tqdm


# stuff that does not work when requesting images online
IGNORE_LIST = [
    "aclu.org",
    "styles.redditmedia.com",
]

def load_image_features(filename: str):
    urls = []
    features = []
    filename_keys = dict()
    url_set = set()
    with open(f"{filename}.ndjson") as fp:
        for line in tqdm(fp.readlines(), desc="loading image features"):
            line = json.loads(line)
            if "filename_key" in line:
                filename_keys[line["filename_key"]] = line["filename"]
            else:
                ignore = False
                for ig in IGNORE_LIST:
                    if ig in line["url"]:
                        ignore = True
                        break
                if ignore:
                    continue

                #if "?" in line["url"]:
                #    continue
                #if len(line["url"]) > 100:
                #    continue

                if line["url"] in url_set:
                    continue
                url_set.add(line["url"])

                features.append(line["features"])
                urls.append((line["f"], line["url"]))

            # if len(urls) > 1000: break

    features = np.asarray(features)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return filename_keys, urls, features


def load_label_features(filename: str):
    df = pd.read_csv(f"{filename}.csv", index_col=0)
    labels = list(df.index)
    features = df.to_numpy()
    features = features / np.linalg.norm(features, axis=-1, keepdims=True)
    return labels, features


if __name__ == "__main__":
    filename_keys, urls, image_features = load_image_features(
        "web-image-features-180k-1h15m"
    )
    print(len(urls), "images")
    labels, label_features = load_label_features("label-features-4")

    sim = 100. * label_features @ image_features.T

    labels_top = []
    for i, label in enumerate(tqdm(labels, desc="sorting")):
        top_idxs = np.argsort(sim[i])[::-1][:5]
        top_urls = [
            (
                sim[i, idx],
                filename_keys[urls[idx][0]].split("/")[-2],
                urls[idx][1]
            )
            for idx in top_idxs
        ]
        labels_top.append(top_urls)

    with open("top-matches-4.json", "w") as fp:
        json.dump(labels_top, fp)
