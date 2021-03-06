"""
A simple script to accumulate connection data
from all recorded HAR files.
"""
import argparse
import time
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "..")

import numpy as np
from sklearn.cluster import SpectralClustering

from har import *
from clusters import LiveClusterer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "what", type=str, nargs="+",
        help="'index', 'bh-long', 'bh-mime-long', 'bh-cluster'"
    )

    return parser.parse_args()


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            o = sorted(o)
        return o


def build_index(generator, host_field: str = "host"):
    per_host = dict()
    for filename, first_entry, e in tqdm(generator):
        key = parse_url(e["request"]["url"])[host_field]
        if key not in per_host:
            per_host[key] = set()
        per_host[key].add(filename)

        #if len(per_host) > 1000:
        #    break

    return per_host


def get_connections(
        generator: Generator[Tuple[str, dict, dict], None, None],
):
    connections = dict()
    old_filename = None
    self_url = None
    host = None
    for filename, first_entry, e in tqdm(generator):
        if not e:
            continue

        if old_filename != filename:
            self_url = parse_url(e["request"]["url"])
            host = self_url["host"]

            connections[host] = {
                "count": 0,
                "count_third": 0,

                "requests": dict(),
                "origins": dict(),
                "referrers": dict(),
            }
        old_filename = filename

        con = connections[host]

        url = parse_url(e["request"]["url"])

        con["count"] += 1
        if url["short_host"] != self_url["short_host"]:
            con["count_third"] += 1

        other_host = url["short_host"]
        con["requests"][other_host] = con["requests"].get(other_host, 0) + 1

        for h in e["request"]["headers"]:
            if h["name"].lower() == "origin":
                other_host = parse_url(h["value"])["short_host"]
                con["origins"][other_host] = con["origins"].get(other_host, 0) + 1

            elif h["name"].lower() == "referer":
                other_host = parse_url(h["value"])["short_host"]
                con["referrers"][other_host] = con["referrers"].get(other_host, 0) + 1

    return connections


def get_character_histogram_per_host(
        generator,
        host_field: str = "short_host",
        as_bytes: bool = False,
        with_mime: bool = False,
):
    def _count(data: dict, text: str):
        if as_bytes:
            text = text.encode("utf-8")
        for c in text:
            data[c] = data.get(c, 0) + 1

    per_host = dict()
    for filename, first_entry, entry in tqdm(generator):
        key = entry["request"][host_field]
        if with_mime:
            # print(entry["response"]["content"])
            mime = entry["response"]["content"].get("mimeType")
            if not mime:
                continue
            if mime:
                for match, value in MIME_TYPES.items():
                    if match in mime:
                        mime = value
                        break
                key += " | " + mime

        if key not in per_host:
            per_host[key] = {
                "count": 0,
                "caller": set(),
                "histogram": {
                    "query_name": dict(),
                    "query_value": dict(),
                    "request_body": dict(),
                    "response_body": dict(),
                    "response_header_name": dict(),
                    "response_header_value": dict(),
                },
            }
        data = per_host[key]
        data["count"] += 1
        data["caller"].add(parse_url(first_entry["request"]["url"])["host"])
        data = data["histogram"]

        for p in entry["request"]["queryString"]:
            _count(data["query_name"], p["name"])
            _count(data["query_value"], p["value"])

        for h in entry["response"]["headers"]:
            _count(data["response_header_name"], h["name"])
            _count(data["response_header_value"], h["value"])

        if entry["request"].get("postData") and entry["request"]["postData"].get("text"):
            text = entry["request"]["postData"]["text"]
            if isinstance(text, dict):
                text = json.dumps(text)
            _count(data["request_body"], text)

        if entry["response"]["content"].get("text"):
            text = entry["response"]["content"]["text"]
            if isinstance(text, dict):
                text = json.dumps(text)
            _count(data["response_body"], text[:10000])

        #if len(per_host) > 20:
        #    break

    per_host = {
        key: data
        for key, data in per_host.items()
        if any(data.values())
    }

    return per_host


def get_histogram_clusters(generator, threshold=.4) -> dict:
    def byte_histogram(text: str):
        text = text.encode("utf-8")
        histogram = np.zeros((256, ))
        for c in text:
            histogram[c] += 1
        if text:
            histogram = np.round(histogram / len(text) * 100, 3)

        return histogram

    clusterer = LiveClusterer()
    filename_index = dict()

    entry_index = 0
    last_verbose_time = 0
    for filename, first_entry, entry in tqdm(generator):
        if filename not in filename_index:
            filename_index[filename] = len(filename_index)
            entry_index = 0

        if entry["request"].get("postData") and entry["request"]["postData"].get("text"):
            text = entry["request"]["postData"]["text"]
            if isinstance(text, dict):
                text = json.dumps(text)

            # url = parse_url(entry["request"]["url"])
            key = (filename_index[filename], entry_index)

            clusterer.add(
                key,
                byte_histogram(text),
                threshold=threshold,
            )

            cur_time = time.time()
            if cur_time - last_verbose_time > 1:
                last_verbose_time = cur_time
                print(f"{len(clusterer.clusters)} clusters")

        entry_index += 1

        #if len(clusterer.clusters) > 10:
        #    break

    clusterer = clusterer.recluster_below_count(2)

    data = {
        "clusters": [
            {
                "histogram": list(np.round(cl["data"], 3)),
                "keys": cl["keys"],
                "count": cl["count"],
            }
            for cl in clusterer.clusters
        ],
        "index": {
            value: key
            for key, value in filename_index.items()
        }
    }
    return data


if __name__ == "__main__":

    args = parse_args()

    for what in args.what:

        generator = iter_har_entries("./recordings/*/*.json")

        if what == "index":
            data = build_index(generator, "host")
            filename = "index"

        elif what == "bh-long":
            data = get_character_histogram_per_host(generator, as_bytes=True, host_field="host")
            filename = "bytes-histogram-longhost"

        elif what == "bh-mime-long":
            data = get_character_histogram_per_host(generator, as_bytes=True, with_mime=True, host_field="host")
            filename = "bytes-histogram-mime-longhost"

        elif what == "bh-cluster":
            data = get_histogram_clusters(generator, threshold=.3)
            filename = "bytes-histogram-clusters-03"

        else:
            raise ValueError(f"Unrecognized '{what}'")

        #get_connections(generator, "./data/all-connections.json")

        if filename:
            with open(f"./data/{filename}.json", "w") as fp:
                json.dump(data, fp, indent=None, cls=Encoder)

import PIL.Image
PIL.Image.frombytes()
