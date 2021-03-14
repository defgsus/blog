import csv
import sys
sys.path.insert(0, "../..")

from web import get_web_file
from har_research.har import *


class SiteStatistics:

    def __init__(self):
        with open("urls.json") as fp:
            self.websites = json.load(fp)

    def df(self):
        return pd.DataFrame(self.websites)


if __name__ == "__main__":

    stats = SiteStatistics()

    print(stats.df())

