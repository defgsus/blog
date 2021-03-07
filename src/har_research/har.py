"""

some related links
https://github.com/jmdugan/blocklists
https://pgl.yoyo.org/adservers/#
https://github.com/InteractiveAdvertisingBureau/GDPR-Transparency-and-Consent-Framework/blob/master/TCFv2/IAB%20Tech%20Lab%20-%20Consent%20string%20and%20vendor%20list%20formats%20v2.md#definitions

"""
import re
import glob
import json
import urllib.parse
from copy import deepcopy
from typing import Optional, Union

import pandas as pd
pd.set_option('display.max_rows', 1000)


# convert a mime-type string to short type
#   if key somewhere in mime-type-string
MIME_TYPES = {
    "html": "html",
    "xml": "html",
    "javascript": "js",
    "font": "font",
    "css": "css",
    "image": "img",
    "octet-stream": "data",
    "json": "data",
    "text": "text",
    "video": "video",
    "mpegurl": "video",
}


class HarFile:
    """
    Base class for loading one or several ``har`` files.
    """
    def __init__(self, filename: str):
        """
        Loads one or several har files
        :param filename: str, handles bash wildcards
        """
        self.filename = filename
        self.data = {"entries": [], "pages": []}
        if self.filename:
            for fn in glob.iglob(self.filename):
                with open(fn) as fp:
                    data = json.load(fp)["log"]

                for e in data["entries"]:
                    url = urllib.parse.urlparse(e["request"]["url"])
                    e["request"].update({
                        "host": url.netloc,
                        "short_host": ".".join(url.netloc.split(".")[-2:]),
                        "path": url.path,
                    })

                self.data["entries"] += data["entries"]
                self.data["pages"] += data["pages"]

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
        har.data["entries"] = list(filter(
            lambda e: self._filter_entry(e, filters),
            har.data["entries"],
        ))
        return har

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
                        if h in text:
                            by_key[key]["to"].add(h)
                            by_key[h]["from"].add(key)

        for h in by_key.values():
            h["to"] = sorted(h["to"])
            h["from"] = sorted(h["from"])
        # print(json.dumps(hosts, indent=2))

        return by_key

    def get_connection_graph_nx(self):
        import networkx as nx
        hosts = self.get_connection_graph()
        graph = nx.DiGraph()
        for host, value in hosts.items():
            graph.add_node(host)
        for host, value in hosts.items():
            for to_host in value["to"]:
                graph.add_edge(host, to_host)
        return graph


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
    har = HarFile("./hars/*spiegel*")
    print(len(har))
    #har = har.filtered({"request.queryString.name": "cookie"})
    #print(len(har))

    #df = pd.DataFrame(har.connections())
    #df.sort_values("req_param_len", ascending=False, inplace=True)
    #print(df)

    #har.filtered({"response.content.mimeType": "image", "request.url": r"reddit\.com"}).dump_connections()

    har.get_connection_graph()
