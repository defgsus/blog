#!./env/bin/python
"""
Converts a jupyter notebook to a jekyll post
"""
import os
import sys
import shutil
import datetime
import subprocess
import argparse
import string


def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "notebook", type=str,
        help="path to the notebook file",
    )

    return parser.parse_args()


def export_notebook(args):
    today = datetime.date.today()

    notebook_filename = os.path.basename(args.notebook)
    output_filename = ".".join(notebook_filename.split(".")[:-1]).lower()
    output_filename = "".join(
        c if c in string.ascii_letters or c in string.digits else "-"
        for c in output_filename
    )
    output_dir = os.path.abspath(
        os.path.join("blog", "_posts", f"{today.year}")
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    commands = [
        "jupyter",
        "nbconvert",
        "--to=src.nbconv.JekyllExporter",
        "--post=src.nbconv.JekyllPostProcessor",
        f"--output-dir=blog/_posts/{today.year}/",
        f"--output={today}-{output_filename}",
        args.notebook,
    ]

    subprocess.check_call(commands)

    asset_path = f"{today}-{output_filename}_files"
    print("EX", os.path.join(output_dir, asset_path))
    if os.path.exists(os.path.join(output_dir, asset_path)):
        printe(f"moving {asset_path} to assets")
        if os.path.exists(os.path.join("blog", "assets", "nb", asset_path)):
            shutil.rmtree(os.path.join("blog", "assets", "nb", asset_path))
        shutil.move(
            os.path.join(output_dir, asset_path),
            os.path.join("blog", "assets", "nb")
        )


if __name__ == "__main__":

    args = parse_args()

    export_notebook(args)

