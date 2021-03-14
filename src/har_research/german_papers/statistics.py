import csv
import sys
sys.path.insert(0, "../..")

from web import get_web_file
from har_research.har import *


class SiteStatistics:

    def __init__(self):
        with open("urls.json") as fp:
            self.websites = json.load(fp)
            self.requested_hosts = dict()

    def df(self):
        return pd.DataFrame(self.websites)

    def iter_website_hars(self):
        for ws in self.websites:
            files = list(glob.glob(f"../automatic/recordings/{ws['url']}/*.json"))
            files.sort(reverse=True)
            har = HarFile(files[0])
            print(ws)
            yield ws, har

    def parse_hars(self):
        for ws, har in self.iter_website_hars():
            self.requested_hosts[ws["url"]] = dict()
            hosts = self.requested_hosts[ws["url"]]

            for e in har:
                url = parse_url(e["request"]["url"])
                key = url["short_host"]

                hosts[key] = hosts.get(key, 0) + 1

            ws["requests"] = len(har)
            ws["servers"] = len(hosts)

    def save_json(self, filename: str):
        with open(filename, "w") as fp:
            json.dump({
                "websites": self.websites,
                "requested_hosts": self.requested_hosts,
            }, fp)


if __name__ == "__main__":

    stats = SiteStatistics()

    stats.parse_hars()
    stats.save_json("website-stats.json")
    print(stats.df())

