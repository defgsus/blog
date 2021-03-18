import json
import fnmatch
from typing import Optional, List, Generator, Tuple

from web import get_web_file


class Kek:
    """
    Scraper for media-ownership-data from

    https://kek-online.de
    "Kommission zur Ermittlung der Konzentration im Medienbereich"

    There is an undocumented json API at ``medienvielfaltsmonitor.de/api/v1/``

    First call will bootstrap all data to a cache directory
    and consumes a few minutes..

    """

    __instance = None

    @classmethod
    def instance(cls):
        if not cls.__instance:
            cls.__instance = Kek()
        return cls.__instance

    def __init__(self):
        self._medias = dict()
        self._holders = dict()

    def get(self, squuid) -> Optional["KekObject"]:
        if squuid in self.medias:
            return self._medias[squuid]
        elif squuid in self.holders:
            return self._holders[squuid]

    def find_media(self, **kwargs) -> Optional["KekObject"]:
        for r in self.filter_media(**kwargs):
            return r

    def find_holder(self, **kwargs) -> Optional["KekObject"]:
        for r in self.filter_holder(**kwargs):
            return r

    def filter_media(self, **kwargs) -> Generator["KekObject", None, None]:
        yield from self._filter(self.medias, filters=kwargs)

    def filter_holder(self, **kwargs) -> Generator["KekObject", None, None]:
        yield from self._filter(self.holders, filters=kwargs)

    def _filter(self, data: dict, filters: dict) -> Generator["KekObject", None, None]:
        for record in data.values():
            matches = True
            for field, value in filters.items():
                # TODO: does only support strings and no dotted paths
                if not fnmatch.fnmatch(record.get(field) or "", value):
                    matches = False
                    break

            if matches:
                yield record

    @property
    def medias(self):
        if not self._medias:
            fn = get_web_file(
                "https://medienvielfaltsmonitor.de/api/v1/media/",
                "kek/media.json"
            )
            with open(fn) as fp:
                media_list = json.load(fp)
            #self._medias = {
            #    m["squuid"]: m
            #    for m in media_list
            #}
            for m in media_list:
                fn = get_web_file(
                    f"https://medienvielfaltsmonitor.de/api/v1/media/{m['squuid']}",
                    f"kek/media/{m['squuid']}.json",
                )
                with open(fn) as fp:
                    self._medias[m["squuid"]] = KekObject(self, json.load(fp))

        return self._medias

    @property
    def holders(self):
        if not self._holders:
            fn = get_web_file(
                "https://medienvielfaltsmonitor.de/api/v1/shareholders/",
                "kek/shareholders.json"
            )
            with open(fn) as fp:
                sh_list = json.load(fp)

            self._holders = dict()
            for m in sh_list:
                fn = get_web_file(
                    f"https://medienvielfaltsmonitor.de/api/v1/shareholders/{m['squuid']}",
                    f"kek/shareholders/{m['squuid']}.json",
                )
                with open(fn) as fp:
                    self._holders[m["squuid"]] = KekObject(self, json.load(fp))

        return self._holders


class KekObject(dict):

    def __init__(self, kek: Kek, data: dict):
        super().__init__(**data)
        self._kek = kek
        self["_hash"] = int(self["squuid"].replace("-", ""), base=16)

    def __str__(self):
        return json.dumps(self, indent=2)

    def __hash__(self):
        return self["_hash"]

    @property
    def name(self) -> str:
        return self.get("fullName") or self["name"]

    def is_media(self) -> bool:
        return "operatedBy" in self

    @property
    def operators(self) -> List["KekObject"]:
        if "operatedBy" not in self:
            return []
        return [
            #KekObject(self._kek, o)
            self._kek.get(o["holder"]["squuid"])
            for o in self["operatedBy"]
        ]

    @property
    def operates(self) -> List["KekObject"]:
        if "operates" not in self:
            return []
        return [
            self._kek.get(o["held"]["squuid"])
            for o in self["operates"]
        ]

    @property
    def owners(self) -> List[Tuple["KekObject", float]]:
        if "ownedBy" not in self:
            return []
        return [
            (self._kek.get(o["holder"]["squuid"]), o.get("capitalShares", 0))
            for o in self["ownedBy"]
        ]

    @property
    def owns(self) -> List[Tuple["KekObject", float]]:
        if "owns" not in self:
            return []
        return [
            (self._kek.get(o["held"]["squuid"]), o["capitalShares"])
            for o in self["owns"]
        ]

    def top_owners(self):
        if self.is_media():
            open_set = {op: 1. for op in self.operators}
        else:
            open_set = {owner: percent for owner, percent in self.owners}

        histogram = dict()
        done_set = set()
        while open_set:
            holder, share = open_set.popitem()
            histogram[holder] = histogram.get(holder, 0) + share

            for owner, percent in holder.owners:
                if (holder, owner) not in done_set:
                    # print(holder.name, ">", owner.name, "|", share, percent / 100)
                    open_set[owner] = open_set.get(owner, 0) + share * percent / 100.

                done_set.add((holder, owner))
        histogram = [
            (owner, histogram[owner] * 100)
            for owner, value in histogram.items()#sorted(histogram, key=lambda h: -histogram[h])
        ]
        histogram.sort(key=lambda h: h[0].name)
        histogram.sort(key=lambda h: -h[1])
        return histogram

    def dump_tree(self, direction: str = "up", prefix: str = "", prefix2: str = "", file=None, _cache=None):
        if _cache is None:
            _cache = set()

        s = self.name
        if self.get("type"):
            s = f"({self['type']}) {s}"
        print(f"{prefix}{prefix2}{s}", file=file)

        prefix = prefix.replace("└", " ").replace("─", " ").replace("├", "│")

        if direction == "up":
            branches = self.operators or self.owners
        elif direction == "down":
            branches = self.operates or self.owns
        else:
            raise ValueError(f"Try direction 'up' or 'down', not '{direction}'")

        if branches and isinstance(branches[0], tuple):
            branches.sort(key=lambda b: -b[1])
        else:
            branches.sort(key=lambda b: b.name)

        if self in _cache:
            if branches:
                print(f"{prefix}└─...")
                return
        _cache.add(self)

        for i, b in enumerate(branches):
            prefix2 = ""
            if isinstance(b, tuple):
                prefix2 = f"{b[1]} "
                b = b[0]

            if i == len(branches) - 1:
                next_prefix = "└─"
            else:
                next_prefix = "├─"
            b.dump_tree(direction=direction, prefix=prefix + next_prefix, prefix2=prefix2, file=file, _cache=_cache)


if __name__ == "__main__":

    kek = Kek()

    # print(json.dumps(kek.medias, indent=2))
    # print(json.dumps(kek.shareholders, indent=2))

    #m = kek.find_holder(name="Gruner + Jahr")
    #m.dump_tree("up")

    m = kek.find_media(name="*Zeitung*")
    #m = kek.find_media(name="www.waz.de")
    #m = kek.find_media(name="www.saarbruecker-zeitung.de")
    #m = kek.find_media(name="www.general-anzeiger-bonn.de")
    #m = kek.find_media(name="www.volksfreund.de")
    #m = kek.find_media(name="www.aachener-nachrichten.de")
    #m = kek.find_media(name="www.merkur.de")
    #m = kek.find_media(name="www.taz.de")
    #m = kek.find_media(name="www.spiegel.de")
    #m = kek.find_media(name="www.kreiszeitung.de")
    #m = kek.find_media(name="www.freitag.de")
    m = kek.find_media(name="www.swp.de")
    m.dump_tree("up")

    for o, perc in m.top_owners():
        print(f"{perc:0.2f} {o.name}")
