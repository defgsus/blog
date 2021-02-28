# hackernews downloader/exporter

The `download.py` downloads all hackernews items from 
their API `https://hacker-news.firebaseio.com/v0` and stores the
results to a mongodb collection in `hackernews.items`.

The `data.py` will dump or export to elasticsearch.

If you download on a separate server, call:
```shell script
python download.py

# and once it's finished
python data.py dump | zip -9 | hn-items.zip
``` 

then on your elasticsearch system:

```shell script
unzip -p hn-items.zip | python data.py export --source -
```