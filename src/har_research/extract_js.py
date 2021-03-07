import sys
import os
import hashlib
import jsbeautifier
from tqdm import tqdm
from har import *

SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__), "adscripts", "extract"
)


def collect_scripts(har: HarFile):
    scripts = {}
    for entry in har:
        text = entry["response"]["content"]["text"]
        scripts[text] = scripts.get(text, 0) + 1

    for entry in sorted(har, key=lambda e: e["request"]["host"]):
        text = entry["response"]["content"]["text"]
        path = entry["request"]["path"]
        name = path.split("/")[-1]
        name = name[-24:]
        print(f'{entry["request"]["host"]:40} {name:24} {scripts[text]}')

    exported = set()
    for entry in tqdm(sorted(har, key=lambda e: e["request"]["host"])):
        text = entry["response"]["content"]["text"]
        if text in exported:
            continue
        exported.add(text)

        if 1:
            path = os.path.join(SCRIPT_DIR, entry["request"]["host"])
            if not os.path.exists(path):
                os.makedirs(path)

            index = 1
            while os.path.exists(os.path.join(path, f"{index}.js")):
                index += 1

            with open(os.path.join(path, f"{index}.js"), "w") as fp:
                dump_script(entry, file=fp)

        else:
            dump_script(entry)
            input()


def dump_script(e: dict, file=None):
    comment = [
        e["request"]["url"].split("?")[0],
        "Date: " + e["startedDateTime"],
        "Headers:",
    ]
    max_len = max(0, *(len(q["name"]) for q in e["request"]["headers"]))
    for q in e["request"]["headers"]:
        comment.append(f"  {q['name']:{max_len}} : {q['value']}")

    if e["request"]["queryString"]:
        comment.append("Query:")
        max_len = max(0, *(len(q["name"]) for q in e["request"]["queryString"]))
        for q in e["request"]["queryString"]:
            comment.append(f"  {q['name']:{max_len}} : {q['value']}")

    print(
        "\n".join(
            f"// {line}"
            for line in comment
        ) + "\n",
        file=file,
    )

    script = e["response"]["content"]["text"]
    if len(script.splitlines()) < 50:
        script = jsbeautifier.beautify(script)
        print("// beautified\n", file=file)

    print(script, file=file)


if __name__ == "__main__":

    har = HarFile("./hars/*spiegel*")
    print(len(har), "requests")

    har = har.filtered({
        "response.content.mimeType": "javascript",
        "response.content.text": bool,
    })
    print(len(har), "javascript")

    #har = har.filtered({
    #    "response.content.text": "[cC]anvas",
    #})
    #print(len(har), "with canvas")

    collect_scripts(har)
