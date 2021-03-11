"""
Gets 'Tranco's top million list and splits it into chunks

https://tranco-list.eu
"""
import argparse
import csv
import sys
sys.path.insert(0, "../..")

from web import get_web_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "batches", type=int,
        help="Number of batches to create",
    )

    parser.add_argument(
        "--size", type=int, default=1000, nargs="?",
        help="Number of pages per batch, default = 1000",
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    tranco_csv_name = get_web_file(
        "https://tranco-list.eu/download/L6J4/full", "tranco_websites.csv",
    )

    with open(tranco_csv_name) as fp:
        reader = csv.reader(fp)
        urls = [row[1] for row in reader]

    for batch_idx in range(args.batches):
        start_idx = batch_idx * args.size
        end_idx = (batch_idx + 1) * args.size
        urls_batch = urls[start_idx:end_idx]
        with open(f"urls/tranco-list-{start_idx+1}-to-{end_idx}.txt", "w") as fp:
            fp.write("\n".join(urls_batch))



