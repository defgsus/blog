### automatic HAR capture using firefox/geckodriver/selenium

Download 'geckodriver' from https://github.com/mozilla/geckodriver/releases

Also need to get `har-export-trigger` plugin via
```shell script
git submodule update
```

Example call
```shell script
python capture.py urls/tranco-list-1001-to-2000.txt -a -s --wait 3 --headless
```