"""
A simple script to accumulate connection data
from all recorded HAR files.
"""
import sys
sys.path.insert(0, "../..")
sys.path.insert(0, "..")

from har import *


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            o = sorted(o)
        return o


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
):
    def _count(data: dict, text: str):
        if as_bytes:
            text = text.encode("utf-8")
        for c in text:
            data[c] = data.get(c, 0) + 1

    per_host = dict()
    for filename, first_entry, entry in tqdm(generator):
        key = entry["request"][host_field]
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


if __name__ == "__main__":

    generator = iter_har_entries("./recordings/*/*.json")

    #get_connections(generator, "./data/all-connections.json")
    data = get_character_histogram_per_host(generator, as_bytes=True, host_field="host")

    with open("./data/bytes-histogram-longhost.json", "w") as fp:
        json.dump(data, fp, indent=None, cls=Encoder)

    #print(data)
