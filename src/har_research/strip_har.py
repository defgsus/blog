import os
import shutil
import time
import urllib.parse
import sys
import datetime
import json
import argparse
sys.path.insert(0, "..")


PATH = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file", type=str, nargs="+",
        help="The HAR files to strip data away"
    )

    return parser.parse_args()


def printe(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def strip_har(filename: str):
    with open(filename) as fp:
        data = json.load(fp)

    def include_mime(mime: str):
        return "javascript" in mime

    for e in data["log"]["entries"]:
        do_strip = False
        mime = e["response"]["content"].get("mimeType")

        response_size = len(e["response"]["content"].get("text") or "")
        response_size += len(e["response"]["content"].get("data") or "")

        if response_size > 1024:
            if mime and not include_mime(mime):
                do_strip = True

        if do_strip:
            url = urllib.parse.urlparse(e['request']['url'])
            printe(f"Stripping {filename}: {mime} {url.netloc}/{url.path}")
            if e["response"]["content"].get("text"):
                e["response"]["content"]["text"] = "STRIPPED!"
            if e["response"]["content"].get("data"):
                e["response"]["content"]["data"] = "STRIPPED!"

    if "." not in filename:
        stripped_filename = filename + "-stripped"
    else:
        stripped_filename = filename.split(".")
        ext = stripped_filename.pop(-1)
        stripped_filename = ".".join(stripped_filename) + "-stripped" + "." + ext

    printe(f"Writing {stripped_filename}")
    with open(stripped_filename, "w") as fp:
        json.dump(data, fp)


if __name__ == "__main__":

    args = parse_args()

    for file in args.file:
        strip_har(file)
