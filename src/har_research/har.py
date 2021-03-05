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

import pandas as pd
pd.set_option('display.max_rows', 500)


# convert a mime-type string to short type
#   if key somewhere in mime-type-string
MIME_TYPES = {
    "html": "html",
    "javascript": "js",
    "font": "font",
    "css": "css",
    "image": "img",
    "octet-stream": "data",
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

                self.data["entries"] += data["entries"]
                self.data["pages"] += data["pages"]

    def __len__(self):
        return len(self.data["entries"])

    def __getitem__(self, i):
        return self.data["entries"][i]

    def dump(self):
        print(json.dumps(self.data, indent=2))

    def dump_connections(self):
        print(pd.DataFrame(self.connections()))

    def filtered(self, filters: dict) -> 'HarFile':
        """
        Return a filtered HarFile

        :param filters: Dict[str, str]

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
            if isinstance(data, str):
                return bool(re.findall(value, data))
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

    def connections(self):
        per_host = dict()
        for e in self.data["entries"]:
            url = urllib.parse.urlparse(e["request"]["url"])
            host = ".".join(url.netloc.split(".")[-2:])
            if host not in per_host:
                per_host[host] = {
                    "req": 0,
                    "req_cookie": 0,
                    "req_param": 0,
                    "req_param_len": 0,
                    "res": 0,
                    "res_cookie": 0,
                    "res_type": {},
                }
            info = per_host[host]
            info["req"] += 1
            info["req_cookie"] += len(e["request"]["cookies"])
            info["req_param"] += len(e["request"]["queryString"])
            info["req_param_len"] += sum(len(e["name"]) + len(e["value"]) for e in e["request"]["queryString"])
            info["res"] += 1
            info["res_cookie"] += len(e["response"]["cookies"])

            mime = e["response"]["content"].get("mimeType")
            #if not mime:
            #   print(json.dumps(e, indent=2))
            if not mime:
                info["res_type"]["-"] = info["res_type"].get("-", 0) + 1
            else:
                for match, type in MIME_TYPES.items():
                    if match in mime:
                        # tracking pixels?
                        if type == "img" and e["response"]["content"]["size"] < 100:
                            type = "tp"

                        info["res_type"][type] = info["res_type"].get(type, 0) + 1
                        break

        return [
            {
                "host": host,
                **info,
            }
            for host, info in per_host.items()
        ]


if __name__ == "__main__":
    har = HarFile("./hars/*reddit*")
    print(len(har))
    #har = har.filtered({"request.queryString.name": "cookie"})
    print(len(har))

    df = pd.DataFrame(har.connections())
    df.sort_values("req_param_len", ascending=False, inplace=True)
    print(df)

    #har.filtered({"response.content.mimeType": "image", "request.url": r"reddit\.com"}).dump_connections()
