import sys
import json
import argparse
import pymongo
import datetime
import urllib.parse
from tqdm import tqdm
from elastipy import Exporter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command", type=str,
        help="'dump', 'export', 'delete-index'",
    )

    parser.add_argument(
        "--source", type=str, default=None, nargs="?",
        help="Instead of mongodb user this ndjson file as source",
    )

    parser.add_argument(
        "--pretty", type=bool, default=False, const=True, nargs="?",
        help="Pretty-print json (which is no valid ndjson)",
    )

    parser.add_argument(
        "-o", "--offset", type=int, default=0, nargs="?",
        help="Offset to start dumping/exporting",
    )

    return parser.parse_args()


class HackerNewsDb:

    def __init__(self, source: str = None):
        """
        :param source: str, optional,
            If None, the mongodb is used as source
            if a filename, a ndjson file is parsed
            If "-", ndjson lines are parsed from stdin
        """
        self.source = source
        if not self.source:
            self.mongo = pymongo.mongo_client.MongoClient()
            self.db = self.mongo.get_database("hackernews")
            self.db_items = self.db.get_collection("items")

    def num_items(self):
        if not self.source:
            return self.db_items.count()
        else:
            return None  # dont really can or want to count the lines

    def items(self, offset: int = 0):
        if not self.source:
            yield from self.db_items.find().sort([("time", 1)]).skip(offset)

        elif self.source == "-":
            offset = -offset
            for line in sys.stdin.readlines():
                if offset >= 0:
                    item = json.loads(line)
                    yield item
                offset += 1
        else:
            offset = -offset
            with open(self.source) as fp:
                for line in fp.readlines():
                    if offset >= 0:
                        item = json.loads(line)
                        yield item
                    offset += 1


class HackerNewsItemsExporter(Exporter):

    INDEX_NAME = "hackernews-items"
    MAPPINGS = {
        "properties": {
            "by": {"type": "keyword"},
            "dead": {"type": "integer"},
            "deleted": {"type": "integer"},
            "descendants": {"type": "integer"},
            "id": {"type": "long"},
            "kids": {"type": "long"},
            "parent": {"type": "integer"},
            "parts": {"type": "integer"},
            "poll": {"type": "integer"},
            "score": {"type": "integer"},
            "timestamp": {"type": "date"},
            "timestamp_hour": {"type": "integer"},
            "timestamp_minute": {"type": "integer"},
            "timestamp_month": {"type": "keyword"},
            "timestamp_weekday": {"type": "keyword"},
            "text": {
                "type": "text",
                "analyzer": "stop",
                "term_vector": "with_positions_offsets_payloads",
                "store": True,
                "fielddata": True,
            },
            "text_length": {"type": "integer"},
            "title": {
                "type": "text",
                "analyzer": "stop",
                "term_vector": "with_positions_offsets_payloads",
                "store": True,
                "fielddata": True,
            },
            "type": {"type": "keyword"},
            "url": {
                "properties": {
                    "url": {"type": "keyword"},
                    "protocol": {"type": "keyword"},
                    "host": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "ext": {"type": "keyword"},
                    "query": {"type": "keyword"},
                    "value": {"type": "keyword"},
                }
            }
        }
    }
    #def get_document_index(self, es_data: dict) -> str:
    #    timestamp = es_data.get("timestamp")
    #    year = timestamp.strftime("%Y") if timestamp else "0"
    #    return self.index_name().replace("*", year)

    def get_document_id(self, es_data: dict):
        return es_data["id"]

    def transform_document(self, data) -> dict:
        data = data.copy()
        data.pop("_id", None)

        timestamp = data.pop("time", None)
        if timestamp is not None:
            timestamp = datetime.datetime.fromtimestamp(timestamp)
            data.update({
                "timestamp": timestamp,
                "timestamp_hour": timestamp.hour,
                "timestamp_minute": timestamp.minute,
                "timestamp_month": timestamp.strftime("%m %b"),
                "timestamp_weekday": timestamp.strftime("%w %A"),
            })

        data.update({
            "deleted": 1 if data.get("deleted") else 0,
            "dead": 1 if data.get("dead") else 0,
            "text_length": len(data.get("text") or ""),
            "url": split_url(data.get("url")),
        })
        return data


def split_url(url):
    if not url:
        return None

    u = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(u.query)
    dot_path = u.path.split(".")
    return {
        "url": url,
        "protocol": u.scheme or None,
        "host": u.netloc or None,
        "path": u.path or None,
        "ext": dot_path[-1] if len(dot_path) > 1 else None,
        "query": sorted(query.keys()) or None,
        "value": sorted(set(sum(query.values(), []))) or None,
    }


if __name__ == "__main__":

    args = parse_args()

    db = HackerNewsDb(source=args.source)

    if args.command == "dump":
        indent = 2 if args.pretty else None
        offset = -args.offset
        for item in tqdm(db.items(), total=db.num_items()):
            if offset >= 0:
                print(json.dumps(item, indent=indent))
            offset += 1

    elif args.command == "delete-index":
        exporter = HackerNewsItemsExporter()
        exporter.delete_index()

    elif args.command == "export":
        exporter = HackerNewsItemsExporter()

        exporter.export_list(db.items(offset=args.offset), verbose=True, count=db.num_items())

    else:
        print(f"Invalid command '{args.command}'")
        exit(1)
