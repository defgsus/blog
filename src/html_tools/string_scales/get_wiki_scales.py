import json
import sys
from pathlib import Path
from bs4 import BeautifulSoup

soup = BeautifulSoup(Path("wikiscales.html").read_text(), features="lxml")

table = soup.find("table", {"class": "wikitable"})

scales = []

for tr in table.find_all("tr"):
    row = [td.text.strip() for td in tr.find_all("td")]
    if len(row) > 5 and row[5].startswith("("):
        intervals = row[5][1:].split(")", 1)[0]
        try:
            intervals = [int(i) for i in intervals.split(",")]
        except ValueError:
            #print("Can't read", row[5], row[0], file=sys.stderr)
            continue

        #print(intervals, row[0])
        #scales.append({
        #    "name": row[0],
        #    "intervals": intervals,
        #})

        print(f"""{{
    "name": "{row[0]}",
    "intervals": {intervals},
}},""")

#print(json.dumps(scales, indent=2))
