---
layout: post
title: Extract HAR files with Selenium / Firefox
tags: security data selenium
---

[Selenium](https://www.selenium.dev/) is a framework for automating 
browser interaction. Basically it starts a browser, surfs a website,
injects a script and exposes everything that can be done with javascript
to read or interact with the page. And a few things more.

[HAR](https://en.wikipedia.org/wiki/HAR_\(file_format\)) is a file 
format to store web traffic of a browser for later analysis. They can be
exported from all the major browsers built-in development tools.

There was always a way to retrieve those HAR files when running a browser
with selenium but in changed over time. 

In 2021, this is the 

- using Python
- using Firefox 
- not using [BrowserMob Proxy](https://github.com/lightbody/browsermob-proxy)

way:

Get a zip of the 
[har-export-trigger](https://github.com/firefox-devtools/har-export-trigger)
extension. You can either download a *signed* version from addons.mozilla or 
you package the contents of the repository yourself.

The necessity is just this: The browser's devtools can only be accessed 
through a background extension script but selenium exposes just an ordinary
page script. The har-export-trigger helps by defining an object in the 
page context that can request the HAR from the background.


```python
from selenium import webdriver

options = webdriver.FirefoxOptions()
# open the devtools right from the start
options.add_argument("--devtools")   

profile = webdriver.FirefoxProfile()
# switch to the netmonitor
profile.set_preference("devtools.toolbox.selectedTool", "netmonitor")
# keep the network log when changing pages
profile.set_preference("devtools.netmonitor.persistlog", True)

browser = webdriver.Firefox(
    firefox_options=options,
    firefox_profile=profile,
)

# record interesting stuff
...

# load extension
# (temporary is required when addon is not signed by mozilla)
browser.install_addon("har-export-trigger.zip", temporary=True)

# call the secret 'HAR.triggerExport()' function
#   and return the result of the promise to python
har_data = browser.execute_async_script(
    "HAR.triggerExport().then(arguments[0]);"
)
```

The data-flow is a bit over-the-top:

- background loads HAR data as js object
- passes it as message to the page context
- there it will be transmitted to selenium (probably as json)
- and python extracts it as a python dict

It's probably not good for capturing HD videos but quite helpful for
everything else.
 