import json
import html
import urllib.parse
import base64
import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
from tqdm import tqdm
from flask import Flask, request
import torch
import clip
from PIL import Image

from har_research.har import HarFile

app = Flask(__name__)


def load_image_features(filename: str):
    urls = []
    features = []
    filename_keys = dict()
    with open(f"{filename}.ndjson") as fp:
        for line in tqdm(fp.readlines(), desc="loading image features"):
            line = json.loads(line)
            if "filename_key" in line:
                filename_keys[line["filename_key"]] = line["filename"]
            else:
                features.append(line["features"])
                urls.append((line["f"], line["url"]))

            # if len(urls) > 10000: break

    features = np.asarray(features)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return filename_keys, urls, features


class ImageSearch:

    def __init__(self):
        # setup html stuff
        with open("index.html") as fp:
            self.index_html = fp.read()
        self.query = ""
        self.query_not = ""
        self.reverse_order = False

        # load image-refs and features
        self.filename_keys, self.urls, self.image_features = \
            load_image_features("web-image-features-180k-1h15m")

        # load CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("loading model to", self.device)
        self.model, self.preprocess = clip.load("ViT-B/32")

        # setup query
        self.query_features = None
        self.query_not_features = None
        self.set_query("", "", False)

    def html(self):
        top_url_indices = self.get_top_url_indices()

        markup = "\n".join(
            f"""<div>
                <img src="img/{file_idx}/{url_idx}" style="max-width: 256px">
                <p>
                    <b>{sim:.2f}</b>
                    <b>{self.filename_keys[file_idx].split("/")[-2]}</b>
                    <code>{self.urls[url_idx][1]}</code>
                </p>
            </div>"""
            for sim, (file_idx, url_idx) in top_url_indices
        )

        return self.index_html % {
            "num_images": len(self.urls),
            "query": self.query,
            "query_not": self.query_not,
            "image_markup": markup,
            "reverse_checked": "checked=\"\"" if self.reverse_order else "",
        }

    def set_query(self, q: str, q_not: str, reverse: bool):
        self.query = q
        self.query_not = q_not
        self.reverse_order = reverse

        if self.query:
            if not self.query_not:
                tokens = clip.tokenize([self.query]).to(self.device)
            else:
                tokens = clip.tokenize([self.query, self.query_not]).to(self.device)
        else:
            if not self.query_not:
                self.query_features = np.ones(512, dtype=np.float).reshape(1, -1)
                return
            else:
                tokens = clip.tokenize([self.query_not]).to(self.device)

        with torch.no_grad():
            self.query_features = self.model.encode_text(tokens).cpu().numpy()

        self.query_features /= np.linalg.norm(self.query_features, axis=-1, keepdims=True)

    def get_top_url_indices(self):
        sim = (100. * self.image_features @ self.query_features.T).T

        if self.query:
            if not self.query_not:
                sim = sim[0]
            else:
                sim = sim[0] - sim[1]
        else:
            if not self.query_not:
                sim = sim[0]
            else:
                sim = -sim[0]

        top_idxs = np.argsort(sim)[::1 if self.reverse_order else -1][:30]
        top_urls = [(sim[idx], (self.urls[idx][0], idx)) for idx in top_idxs]
        return top_urls

    def get_image(self, file_id, url_id):
        har_filename = self.filename_keys[file_id]
        url = self.urls[url_id][1]
        return self._load_image(har_filename, url)

    def _load_image(self, filename: str, url: str):
        for e in HarFile(filename):
            if e["request"]["url"] == url:
                try:
                    b64 = e["response"]["content"]["text"]
                    return base64.b64decode(b64)
                except:
                    pass


image_search = ImageSearch()


@app.route("/")
def index():
    params = urllib.parse.parse_qs(request.query_string)
    query = (params.get(b"q") or [b""])[0].decode("utf-8")
    query_not = (params.get(b"qnot") or [b""])[0].decode("utf-8")
    image_search.set_query(query, query_not, reverse=b"reverse" in params)
    return image_search.html()


@app.route("/img/<int:file_id>/<int:url_id>")
def get_image(file_id, url_id):
    return image_search.get_image(file_id, url_id)