"""

some related links
https://github.com/jmdugan/blocklists
https://pgl.yoyo.org/adservers/#
https://github.com/InteractiveAdvertisingBureau/GDPR-Transparency-and-Consent-Framework/blob/master/TCFv2/IAB%20Tech%20Lab%20-%20Consent%20string%20and%20vendor%20list%20formats%20v2.md#definitions

"""
import os
import re
import glob
import json
import urllib.parse
from copy import deepcopy
from typing import Optional, Union, Generator, Tuple

from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_rows', 1000)


# convert a mime-type string to short type
#   if key somewhere in mime-type-string
MIME_TYPES = {
    "html": "html",
    "xml": "xml",
    "javascript": "js",
    "ecmascript": "js",
    "wasm": "wasm",
    "font": "font",
    "woff": "font",
    "css": "css",
    "image": "img",
    "octet-stream": "data",
    "json": "json",
    "form-data": "form",
    "text": "text",
    "video": "video",
    "mpegurl": "video",
    "audio": "audio",
    "unknown": "unknown",
    "opendocument": "odf",
    "shockwave-flash": "flash",
}


def parse_url(url: str) -> dict:
    url = urllib.parse.urlparse(url)

    short_host = url.netloc.split(".")
    if len(short_host) > 2:
        if short_host[-2] in ("co", "com"):
            short_host = short_host[-3:]
        else:
            short_host = short_host[-2:]

    return {
        "protocol": url.scheme,
        "host": url.netloc,
        "short_host": ".".join(short_host),
        "path": url.path,
        "params": url.params,
    }


def enrich_entry(e: dict) -> None:
    """
    Add some convenience stuff to a HAR entry
    :param e: dict, will be updated with new data
    """
    url = parse_url(e["request"]["url"])
    e["request"].update(url)


def iter_har_entries(
        *glob_pattern: str,
        require: bool = True,
) -> Generator[Tuple[str, dict, dict], None, None]:
    """
    Iterate through all entries of all found HAR files.

    Yields a tuple of (filename, first-entry, current-entry)

    :param glob_pattern: str, wildcard pattern
    """
    num_files = 0
    for pattern in glob_pattern:
        for filename in glob.glob(pattern):
            num_files += 1
            with open(filename) as fp:
                data = json.load(fp)

            first_entry = None
            for entry in data["log"]["entries"]:
                enrich_entry(entry)
                if not first_entry:
                    first_entry = entry

                yield filename, first_entry, entry

    if require and not num_files:
        raise IOError(f"No HARs found at {glob_pattern}")


class HarFile:
    """
    Base class for loading one or several ``har`` files.
    """
    def __init__(
            self,
            filename: str,
            require: bool = True,
            remove_data: bool = False,
            verbose: bool = False,
    ):
        """
        Loads one or several har files

        :param filename: str, handles bash wildcards
        :param require: bool, If True raise IOError when nothing is found
        :param remove_data: bool, Will remove data via the HarFile._remove_data() method
            to save memory
        :param verbose: bool, Show loading progress
        """
        self.filename = filename
        self.data = {"entries": [], "pages": []}
        if self.filename:
            if not verbose:
                gen = glob.iglob(self.filename)
            else:
                gen = tqdm(glob.glob(self.filename))
            for fn in gen:
                with open(fn) as fp:
                    data = json.load(fp)["log"]

                for e in data["entries"]:
                    enrich_entry(e)

                if remove_data:
                    self._remove_data(data["entries"])

                self.data["entries"] += data["entries"]
                self.data["pages"] += data["pages"]

            if require and not self.data["entries"]:
                raise IOError(f"No HARs found at '{filename}'")

    def __len__(self):
        return len(self.data["entries"])

    def __getitem__(self, i):
        return self.data["entries"][i]

    def __iter__(self):
        return self.data["entries"].__iter__()

    def dump(self):
        print(json.dumps(self.data, indent=2))

    def dump_connections(self):
        print(pd.DataFrame(self.connections()))

    def filtered(self, filters: dict) -> 'HarFile':
        """
        Return a filtered HarFile

        :param filters: Dict[str, str]

            key is path to param
            value can be
                - str, will be used as regex
                - function with value argument

            {
                "request.url": r"123[a-z]+",
                "request.queryParam.name: r"cookie",
            }

        :return: new HarFile instance
        """
        har = self.__class__(None)
        har.filename = self.filename
        har.data = deepcopy(self.data)
        har.data["entries"] = list(self.filtered_iter(filters))
        return har

    def filtered_iter(self, filters: dict) -> Generator[dict, None, None]:
        yield from filter(
            lambda e: self._filter_entry(e, filters),
            self.data["entries"],
        )

    def _filter_entry(self, entry: dict, filters: dict = None) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if not self._filter_path(entry, key.split("."), value):
                return False

        return True

    def _filter_path(self, data, path: list, value) -> bool:
        if len(path) == 0:
            if isinstance(value, str):
                return bool(re.findall(value, data))
            elif callable(value):
                return value(data)
            else:
                return data == value
        else:
            if isinstance(data, dict):
                if path[0] not in data:
                    return False
                return self._filter_path(data[path[0]], path[1:], value)
            elif isinstance(data, list):
                for entry in data:
                    if self._filter_path(entry, path, value):
                        return True
                return False
            else:
                raise TypeError(
                    f"Can not handle type '{type(data).__name__}' for remaining path {path}"
                )

    @classmethod
    def _remove_data(
            cls,
            entries,
            mime_types: Tuple[str] = ("video", "image"),
            max_size: int = 10000,
    ):
        for e in entries:
            mime = e["response"]["content"].get("mimeType")
            remove = False
            if mime:
                for mt in mime_types:
                    if mt in mime:
                        remove = True
                        break
            if not remove:
                if e["response"]["content"].get("text"):
                    e["response"]["content"]["text"] = e["response"]["content"]["text"][:max_size]
            else:
                data = e["response"]["content"].pop("text", None)
                del data

    def dump_pretty(self, file=None, max_data=1024):
        width, _ = os.get_terminal_size()
        width = max(1, width - 3)
        for e in self:
            print("\n" + "-" * width, file=file)
            print("Url:", e["request"]["url"].split("?")[0], file=file)
            print("Method:", e["request"]["method"], file=file)
            print("Date:", e["startedDateTime"], file=file)
            print("Headers:", file=file)
            max_len = max(0, *(len(q["name"]) for q in e["request"]["headers"]))
            for q in e["request"]["headers"]:
                print(f"  {q['name']:{max_len}} : {q['value']}", file=file)

            if e["request"]["queryString"]:
                print("Query:", file=file)
                max_len = max(0, *(len(q["name"]) for q in e["request"]["queryString"]))
                for q in e["request"]["queryString"]:
                    print(f"  {q['name']:{max_len}} : {q['value']}", file=file)

            if e["request"].get("postData"):
                data = None
                if e["request"]["postData"].get("text"):
                    data = e["request"]["postData"].get("text")
                elif e["request"]["postData"].get("params"):
                    print(e)
                    raise NotImplementedError
                if data:
                    print(f"Post:\n{data}")

            if e["response"]["headers"]:
                print("Response headers:", file=file)
                max_len = max(0, *(len(q["name"]) for q in e["response"]["headers"]))
                for q in e["response"]["headers"]:
                    print(f"  {q['name']:{max_len}} : {q['value']}", file=file)

            content = None
            for key in ("text", "data"):
                if e["response"]["content"].get(key):
                    content = e["response"]["content"][key]
                    break
            if content:
                print(f"Content:\n{content[:max_data]}", file=file)

    def connections_df(self):
        df = pd.DataFrame(self.connections())
        df.index = df.pop("host")
        df.sort_values("strength", ascending=False, inplace=True)
        return df

    def connections(self):
        per_host = dict()
        max_strength = 0
        for e in self:
            host = e["request"]["short_host"]
            if host not in per_host:
                per_host[host] = {
                    "req": 0,
                    "req_cookie": 0,
                    "req_param": 0,
                    "req_param_len": 0,
                    "res": 0,
                    "res_cookie": 0,
                    "res_type": {},
                    "strength": 0,
                }
            info = per_host[host]
            info["req"] += 1
            info["req_cookie"] += len(e["request"]["cookies"])
            info["req_param"] += len(e["request"]["queryString"])
            info["req_param_len"] += sum(len(e["name"]) + len(e["value"]) for e in e["request"]["queryString"])
            info["res"] += 1
            info["res_cookie"] += len(e["response"]["cookies"])

            info["strength"] += info["req"] + info["req_cookie"] + \
                                info["req_param"] + info["res"] + info["res_cookie"]
            max_strength = max(max_strength, info["strength"])

            mime = e["response"]["content"].get("mimeType")
            #if not mime:
            #   print(json.dumps(e, indent=2))
            if not mime:
                info["res_type"]["-"] = info["res_type"].get("-", 0) + 1
            else:
                _matched = False
                for match, type in MIME_TYPES.items():
                    if match in mime:
                        # tracking pixels?
                        if type == "img" and e["response"]["content"]["size"] < 100:
                            type = "tp"

                        info["res_type"][type] = info["res_type"].get(type, 0) + 1
                        _matched = True
                        break
                if not _matched:
                    print("UNMATCHED ", mime)

        return [
            {
                "host": host,
                **info,
                "strength": info["strength"] / max(1, max_strength)
            }
            for host, info in per_host.items()
        ]

    def get_dependency_graph(
            self,
            key_path: str = "request.short_host",
            with_referer: bool = True,
            with_text_match: bool = False,
    ):
        by_key = {}
        for e in self:
            key = get_value_path(e, key_path)
            if key not in by_key:
                by_key[key] = {
                    "count": 0,
                    "to": set(),
                    "from": set(),
                }
            by_key[key]["count"] += 1

        for e in self:
            key = get_value_path(e, key_path)
            for header in e["request"]["headers"]:
                if header["name"].lower() == "referer" and with_referer:
                    for h in by_key:
                        if h in header["value"] and h != key:
                            by_key[h]["to"].add(key)
                            by_key[key]["from"].add(h)

            if with_text_match:
                text = e["response"]["content"].get("text")
                if text:
                    for h in by_key:
                        if h != key and h in text:
                            by_key[key]["to"].add(h)
                            by_key[h]["from"].add(key)

        for h in by_key.values():
            h["to"] = sorted(h["to"])
            h["from"] = sorted(h["from"])
        # print(json.dumps(hosts, indent=2))

        return by_key

    def get_dependency_graph_nx(self):
        import networkx as nx
        hosts = self.get_dependency_graph()
        graph = nx.DiGraph()
        for host, value in hosts.items():
            graph.add_node(host)
        for host, value in hosts.items():
            for to_host in value["to"]:
                graph.add_edge(host, to_host)
        return graph

    @classmethod
    def get_entry_size(cls, e):
        """
        Try to get real size of the request and response,
        because har-file might contain data but no size values.
        :returns: tuple of request-size, response-size
        """
        try:
            num_in = (e["request"].get("headersSize") or 0) + (e["request"].get("bodySize") or 0)
            if num_in == 0:
                num_in = sum(len(h["name"]) + len(h["value"]) for h in e["request"]["headers"])
                num_in += sum(len(h["name"]) + len(h["value"]) for h in e["request"]["queryString"])

            num_out = (e["response"].get("headersSize") or 0)
            if num_out == 0:
                num_out = sum(len(h["name"]) + len(h["value"]) for h in e["response"]["headers"])

            if (e["response"].get("bodySize") or 0) > 0:
                num_out += e["response"]["bodySize"]
            else:
                num_out += (e["response"]["content"].get("size") or 0)
            return num_in, num_out
        except KeyError:
            print(json.dumps(e, indent=2))
            raise

    def get_actions(
            self,
            key_path: str = "request.short_host",
            as_df: bool = False,
    ):
        """
        Return a couple of counters for various actions that are
        associated with each request/response.

        :param key_path: str, dotted path to the value that
            separates the entries, typically the host name.
        :param as_df: bool, return a pandas.DataFrame
        :return: dict: key -> dict of values, where ``key`` is each value
            behind the ``key_path``
        """
        actions = dict()
        for e in self:
            key = get_value_path(e, key_path)
            if key not in actions:
                actions[key] = {
                    "count": 0,
                    "in_out_ratio": 0,
                    "receive_params": 0,
                    "send_media": 0,
                    "send_cookie": 0,
                    "send_js": 0,
                    "send_tp": 0,
                    "send_js_canvas": 0,
                }
            actions[key]["count"] += 1
            actions[key]["send_cookie"] += len(e["response"]["cookies"])
            for p in e["request"]["queryString"]:
                actions[key]["receive_params"] += len(p["name"]) + len(p["value"])

            num_in, num_out = self.get_entry_size(e)
            ratio = 1
            if num_out:
                ratio = num_in / num_out
            actions[key]["in_out_ratio"] = ratio

            mime = e["response"]["content"].get("mimeType")
            text = e["response"]["content"].get("text")
            if mime and e["response"]["content"]["size"]:
                if "javascript" in mime:
                    if text:
                        actions[key]["send_js"] += 1
                        if re.findall("[cC]anvas", text):
                            actions[key]["send_js_canvas"] += 1
                elif "image" in mime:
                    if e["response"]["content"]["size"] < 100:
                        actions[key]["send_tp"] += 1
                    else:
                        actions[key]["send_media"] += 1

        if as_df:
            df = pd.DataFrame([{"key": key, **value} for key, value in actions.items()])
            df.index = df.pop("key")
            return df
        return actions


def get_value_path(data, path: Union[list, tuple, str]):
    if isinstance(path, str):
        path = path.split(".")

    if len(path) == 0:
        return data
    else:
        if isinstance(data, dict):
            return get_value_path(data.get(path[0]), path[1:])
        elif isinstance(data, (tuple, list)):
            return get_value_path(data[int(path[0])], path[1:])
        else:
            raise TypeError(f"Can not handle type '{type(data).__name__}' with subpath {'.'.join(path)}")


if __name__ == "__main__":
    har = HarFile("./automatic/recordings/*/*.json", verbose=True, remove_data=True)
    print(len(har))
    #har = har.filtered({"request.queryString.name": "tid"})
    har = har.filtered({"request.method": "POST"})
    print(len(har))
    #har = har.filtered({"request.url": "https://ib.adnxs.com/ut/v3/prebid"})
    #har.dump_pretty(max_data=10000)
    #
    #har = har.filtered({"request.queryString.name": "cookie"})
    #print(len(har))

    #df = pd.DataFrame(har.connections())
    #df.sort_values("req_param_len", ascending=False, inplace=True)
    #print(df)

    #har.filtered({"response.content.mimeType": "image", "request.url": r"reddit\.com"}).dump_connections()

    #har.get_dependency_graph()
