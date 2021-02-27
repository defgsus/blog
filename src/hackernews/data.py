import json
import argparse
import pymongo
import datetime
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

    return parser.parse_args()


class HackerNewsDb:

    def __init__(self, source: str = None):
        self.source = source
        if not self.source:
            self.mongo = pymongo.mongo_client.MongoClient()
            self.db = self.mongo.get_database("hackernews")
            self.db_items = self.db.get_collection("items")

    def num_items(self):
        if not self.source:
            return self.db_items.count()
        else:
            return None  # dont really want to count the lines

    def items(self):
        if not self.source:
            yield from self.db_items.find()
        else:
            with open(self.source) as fp:
                for line in fp.readlines():
                    item = json.loads(line)
                    yield item


class HackerNewsItemsExporter(Exporter):

    INDEX_NAME = "hackernews-items-*"
    MAPPINGS = {
        "properties": {
            "by": {"type": "keyword"},
            "dead": {"type": "integer"},
            "deleted": {"type": "integer"},
            "descandants": {"type": "integer"},
            "id": {"type": "long"},
            "kids": {"type": "long"},
            "parent": {"type": "integer"},
            "parts": {"type": "integer"},
            "poll": {"type": "integer"},
            "score": {"type": "integer"},
            "timestamp": {"type": "date"},
            "text": {
                "type": "text",
                "analyzer": "stop",
                "term_vector": "with_positions_offsets_payloads",
                "store": True,
                "fielddata": True,
            },
            "title": {
                "type": "text",
                "analyzer": "stop",
                "term_vector": "with_positions_offsets_payloads",
                "store": True,
                "fielddata": True,
            },
            "type": {"type": "keyword"},
            "url": {"type": "keyword"},
        }
    }
    def get_document_index(self, es_data: dict) -> str:
        timestamp = es_data["timestamp"]
        year = timestamp.strftime("%Y") if timestamp else "0"
        return self.index_name().replace("*", year)

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
            "deleted": 1 if data.get("deleted") else 0,
            "dead": 1 if data.get("dead") else 0,
        })
        return data


if __name__ == "__main__":

    args = parse_args()

    db = HackerNewsDb(source=args.source)

    if args.command == "dump":
        indent = 2 if args.pretty else None
        for item in tqdm(db.items(), total=db.num_items()):
            print(json.dumps(item, indent=indent))

    elif args.command == "delete-index":
        exporter = HackerNewsItemsExporter()
        exporter.delete_index()

    elif args.command == "export":
        exporter = HackerNewsItemsExporter()

        exporter.export_list(db.items(), verbose=True, count=db.num_items())

    else:
        print(f"Invalid command '{args.command}'")
        exit(1)
