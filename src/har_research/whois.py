"""
No no! I do not build a database!
I'm just caching results
"""
import os
import subprocess
from typing import Optional
import argparse


class Cache:

    PATH = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "whois-cache",
    ))

    __instance = None

    @classmethod
    def instance(cls):
        if not cls.__instance:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        self._objects = dict()

    def get(self, type: str, id: str) -> Optional[str]:
        if type not in self._objects:
            self._objects[type] = dict()

        if id not in self._objects[type]:
            filename = os.path.join(self.PATH, type, f"{id}.txt")
            if os.path.exists(filename):
                with open(filename) as fp:
                    self._objects[type][id] = fp.read()

        if id in self._objects[type]:
            return self._objects[type][id]

    def store(self, type: str, id: str, content: str):
        if type not in self._objects:
            self._objects[type] = dict()

        self._objects[type][id] = content

        filepath = os.path.join(self.PATH, type)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = os.path.join(filepath, f"{id}.txt")

        with open(filename, "w") as fp:
            fp.write(content)


def run_whois(obj: str, read_cache: bool = True, store_cache: bool = True) -> str:
    if read_cache:
        content = Cache.instance().get("whois", obj)
        if content:
            return content

    try:
        content = subprocess.check_output(
            ["whois", obj]
        )

        content = content.decode("utf-8")
    except subprocess.CalledProcessError as e:
        content = f"ERROR {e}"

    if store_cache:
        Cache.instance().store("whois", obj, content)

    return content


WHOIS_KEYS = {
    "registrant": [
        "Registrant Organization",
    ],
    "registrant_country": [
        "Registrant Country",
    ],
    "registrar": [
        "Registrar",
    ],
    "name_server": [
        "Name Server",
        "Nserver",
    ],
}

WHOIS_IGNORE_RESPONSE = (
    "",
    "REDACTED FOR PRIVACY",
    "DATA REDACTED",
    "Not Disclosed",
    "Whois Privacy Service",
)


def whois_to_dict(text: str) -> dict:
    ret = {
        key: None
        for key in WHOIS_KEYS
    }
    for section_key, keys in WHOIS_KEYS.items():
        for key in keys:
            try:
                idx = text.index(key + ":")
                value = text[idx+len(key)+1:].split("\n")[0].strip()
                if value not in WHOIS_IGNORE_RESPONSE:
                    ret[section_key] = value
                    break
            except ValueError:
                pass

    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "object", type=str, nargs="+",
        help="The 'object' to search for, e.g. domain name or IP address",
    )

    args = parser.parse_args()

    for obj in args.object:
        print(f"\n------------------ whois {obj} ------------------\n")
        text = run_whois(obj)
        print(text)


