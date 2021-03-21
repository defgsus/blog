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
                    "shareholders": [],
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
        from har_research.whois import WhoisCache
        hosts = dict()
        for ws, har in self.iter_website_hars():
            url = parse_url(ws["url"])
            hosts[url["short_host"]] = {url["short_host"], url["host"]}
            for e in har:
                url = parse_url(e["request"]["url"])
                if url["short_host"] not in hosts:
                    hosts[url["short_host"]] = {url["short_host"]}
                hosts[url["short_host"]].add(url["host"])

        self.whois = dict()
        print("whoissing...")
        missing = dict()
        for short_host, host_set in tqdm(hosts.items()):
            host_set = sorted(host_set, key=lambda h: -len(h))
            data = {}
            while host_set and not data.get("network"):
                host = host_set.pop(-1)
                #if data:
                #    print("GETTING", host)
                data2 = WhoisCache.instance().get_best_effort(host, ask_the_web=True)
                for key, v in data2.items():
                    data[key] = data.get(key) or v

            for key, value in data.items():
                if not value:
                    missing[key] = missing.get(key, 0) + 1

            self.whois[short_host] = data
            if "architektur" in short_host:
                print(f"{short_host:40} - {data}")

        print("MISSING:", missing)

    def get_ownership(self):
        from kek import Kek
        MAPPING = {
            "BILD-Zeitung": "Bild",
            "Generalanzeiger": "General-Anzeiger (Bonn)",
        }

        missing = 0
        for ws in self.websites.values():
            media = Kek.instance().find_media(name="*"+ws["url"])
            if not media:
                media = Kek.instance().find_media(name=MAPPING.get(ws["title"], ws["title"]))
            # print("%30s" % ws["title"], (media or {}).get("name", "---"))
            if not media:
                missing += 1
            else:
                ws["shareholders"] = [
                    {
                        "name": o.get("fullName") or o["name"],
                        "id": o["squuid"],
                        "value": round(perc, 4),
                    }
                    for o, perc in media.top_owners()
                ]
        if missing:
            print(missing, "missing ownerships")

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
                        "ip": "",
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

                ip = e.get("serverIPAddress")
                if ip and e["request"]["short_host"] == e["request"]["host"]:
                    request["ip"] = ip
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
                "whois": self.whois,
            }, fp)


if __name__ == "__main__":

    stats = WebsiteStatistics()

    stats.get_whois()
    stats.get_ownership()
    stats.parse_hars()
    stats.save_json("website-stats.json")

