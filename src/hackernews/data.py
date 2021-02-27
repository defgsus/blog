import json
import argparse
import pymongo
import datetime
from elastipy import Exporter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command", type=str,
        help="'dump', 'export'",
    )

    parser.add_argument(
        "--pretty", type=bool, default=False, const=True, nargs="?",
        help="Pretty-print json (which is no valid ndjson)",
    )

    return parser.parse_args()


class HackerNewsDb:

    def __init__(self):
        self.mongo = pymongo.mongo_client.MongoClient()
        self.db = self.mongo.get_database("hackernews")
        self.db_items = self.db.get_collection("items")

    def num_items(self):
        return self.db_items.count()

    def items(self):
        yield from self.db_items.find()


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
            "text": {"type": "text"},
            "title": {"type": "text"},
            "type": {"type": "keyword"},
            "url": {"type": "keyword"},
        }
    }
    def get_document_index(self, es_data: dict) -> str:
        return self.index_name().replace("*", es_data["timestamp"].strftime("%Y"))

    def get_document_id(self, es_data: dict):
        return es_data["id"]

    def transform_document(self, data) -> dict:
        data = data.copy()
        data.pop("_id", None)
        data.update({
            "timestamp": datetime.datetime.fromtimestamp(data.pop("time")),
            "deleted": 1 if data.get("deleted") else 0,
            "dead": 1 if data.get("dead") else 0,
        })
        return data


if __name__ == "__main__":

    args = parse_args()

    db = HackerNewsDb()

    if args.command == "dump":
        indent = 2 if args.pretty else None
        for item in db.items():
            print(json.dumps(item, indent=indent))

    elif args.command == "export":
        exporter = HackerNewsItemsExporter()

        exporter.export_list(db.items(), verbose=True, count=db.num_items())

    else:
        print(f"Invalid command '{args.command}'")
        exit(1)
