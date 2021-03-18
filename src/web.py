import os
import requests
import sys

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def get_web_file(url: str, filename: str):
    """
    Download from url, return absolute filename

    :param url: str, the place in the wab
    :param filename: filename with optional additional path
    :return: str, filename with path
    """
    full_path = CACHE_DIR
    filename_path = os.path.dirname(filename)
    if filename_path:
        full_path = os.path.join(CACHE_DIR, filename_path)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    cache_filename = os.path.join(
        CACHE_DIR,
        filename,
    )
    if os.path.exists(cache_filename):
        return cache_filename

    print(f"downloading {url} to {cache_filename}", file=sys.stderr)
    response = requests.get(url)

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    with open(cache_filename, "wb") as fp:
        fp.write(response.content)

    return cache_filename
