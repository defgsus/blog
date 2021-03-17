import csv
import sys
sys.path.insert(0, "../..")

from web import get_web_file
from har_research.har import *


EXCLUDE_REFERRER_PATHS = {
    "www.handelsblatt.com": [
        re.compile(r"^/downloads/.*"),
    ],
    "www.architekturzeitung.com": [
        re.compile(r"^/architekturgalerie-architektur.*"),
        re.compile(r".*datenschutz-architekturzeitung.*"),
    ],
}


class WebsiteStatistics:

    def __init__(self):
        with open("urls.json") as fp:
            websites = json.load(fp)
            self.websites = {
                ws["url"]: {
                    "url": ws["url"],
                    "title": ws["title"],
                    "publisher": ws["publisher"],
                    "requests": dict(),
                }
                for ws in websites
            }

    def iter_website_hars(self):
        for ws in tqdm(self.websites.values()):
            files = list(glob.glob(f"../automatic/recordings/{ws['url']}/*.json"))
            files.sort(reverse=True)
            har = HarFile(files[0])
            # print(ws)
            yield ws, har

    def get_whois(self):
        import random
        import time
        from har_research.whois import run_whois, whois_to_dict
        hosts = set()
        for ws, har in self.iter_website_hars():
            for e in har:
                hosts.add(e["request"]["short_host"])
                ip = e.get("serverIPAddress")
                if ip and e["request"]["short_host"] == e["request"]["host"]:
                    hosts.add(ip)
            #if len(hosts) > 100:
            #    break
        print("whoissing...")
        for host in tqdm(hosts):
            content = run_whois(host)
            d = whois_to_dict(content)
            print(f"{host:40} - {d}")
            #print(f"\n------------------ whois {host} ------------------\n")
            #print(content)
            # time.sleep(random.uniform(1, 2))

    def parse_hars(self):
        for ws, har in self.iter_website_hars():
            requests = dict()
            ws["requests"] = requests

            article_paths = set()
            for e in har:
                url = parse_url(e["request"]["url"])

                referrer = None
                for h in e["request"]["headers"]:
                    if h["name"].lower() == "referer":
                        referrer = h["value"]
                        break
                e["referrer"] = referrer

            for e in har:
                url = parse_url(e["request"]["url"])
                key = url["short_host"]
                is_third = not (ws["url"] in url["host"] or url["host"] in ws["url"])
                referrer = e["referrer"]

                if key not in requests:
                    requests[key] = {
                        "is_third": is_third,
                        "count": 0,
                        "article_referer": 0,
                        "bytes_sent": 0,
                        "bytes_received": 0,
                    }
                request = requests[key]
                request["count"] += 1
                request["bytes_sent"] = sum(len(e["name"]) + len(e["value"]) for e in e["request"]["queryString"])
                if "postData" in e["request"]:
                    request["bytes_sent"] += e["request"]["postData"].get("size", 0)

                request["bytes_received"] += e["response"]["content"].get("size", 0)

                if is_third and referrer:
                    ref_url = parse_url(referrer)
                    if ref_url["host"] == ws["url"]:
                        if len(ref_url["path"]) > 20 and not ref_url["path"].endswith(".css"):
                            #article_paths.add(ref_url["path"])
                            request["article_referer"] += 1

            ws["num_requests"] = sum(r["count"] for r in ws["requests"].values())
            #print(ws["url"])
            #print(" ", referrer)
            #for p in article_paths:
            #    print(" ", p)

    def save_json(self, filename: str):
        with open(filename, "w") as fp:
            json.dump({
                "websites": self.websites,
            }, fp)


if __name__ == "__main__":

    stats = WebsiteStatistics()

    #stats.get_whois()
    stats.parse_hars()
    stats.save_json("website-stats.json")

