---
layout: post
title: Operating System Required for Proper COVID-19 Vaccination
---

In Germany, there is this website 
[www.impfterminservice.de](https://www.impfterminservice.de/impftermine) 
by the *Kassenärztliche Bundesvereinigung*, a federal association
of physicians and health insurance companies, where people can 
centrally book an appointment for a COVID-19 vaccination in their
area. 

The website is implemented by 
[kv.digital GmbH - Digitalisierung im Gesundheitswesen](https://www.kv.digital/),
former [KV Telematik GmbH](http://www.kv-telematik.de/),
a company originally founded by the *Kassenärztliche Bundesvereinigung*
([ref](https://www.kv.digital/ueber-uns.html)).

Their [privacy agreement](https://www.impfterminservice.de/datenschutz)
states that, in order to visit the website, the following **required**
data is collected:

    - Operating system of the user
    - Browser type and version
    - Internet-Service-Provider
    - IP-Address
    - Date and time of access

It's a pretty *usual* statement nowadays, although it's already quite 
questionable why submitting the operating system is **required** for 
visiting a website. 

Let's assume, they just need screen resolution and
the knowledge if a touch device is present or something like that, 
to make the site look good on all systems. Still, why the **requirement**
to submit these values to the server?  

Here's a dump of the data that was collected in my browser:

```json
{
  "ap": "true",
  "bt": "0",
  "fonts": "6,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,42,43,44,45,63,65",
  "fh": "cd9349b7cb07e2746e452457b8469d0381b77609",
  "timing": {
    "1": 19,
    "2": 148,
    "3": 259,
    "4": 377,
    "5": 478,
    "6": 579,
    "profile": {
      "bp": 0,
      "sr": 0,
      "dp": 1,
      "lt": 0,
      "ps": 0,
      "cv": 9,
      "fp": 1,
      "sp": 0,
      "br": 0,
      "ieps": 0,
      "av": 0,
      "z1": 5,
      "jsv": 2,
      "nav": 0,
      "nap": 1,
      "crc": 0,
      "z2": 6,
      "z3": 0,
      "z4": 1,
      "z5": 0,
      "z6": 1,
      "fonts": 11
    },
    "main": 1040,
    "compute": 19,
    "send": 590
  },
  "bp": "",
  "sr": {
    "inner": [1248, 961],
    "outer": [1920, 1046],
    "screen": [0, 0],
    "pageOffset": [0, 0],
    "avail": [1920, 1046],
    "size": [1920, 1080],
    "client": [1248, 0],
    "colorDepth": 24,
    "pixelDepth": 24
  },
  "dp": {
    "XDomainRequest": 0,
    "createPopup": 0,
    "removeEventListener": 1,
    "globalStorage": 0,
    "openDatabase": 0,
    "indexedDB": 1,
    "attachEvent": 0,
    "ActiveXObject": 0,
    "dispatchEvent": 1,
    "addBehavior": 0,
    "addEventListener": 1,
    "detachEvent": 0,
    "fireEvent": 0,
    "MutationObserver": 1,
    "HTMLMenuItemElement": 0,
    "Int8Array": 1,
    "postMessage": 1,
    "querySelector": 1,
    "getElementsByClassName": 1,
    "images": 1,
    "compatMode": "CSS1Compat",
    "documentMode": 0,
    "all": 1,
    "now": 1,
    "contextMenu": 0
  },
  "lt": "1627044885148+2",
  "ps": "true,true",
  "cv": "951ea047f51144fadf5111e26144ca162f9918ea",
  "fp": "false",
  "sp": "false",
  "br": "Firefox",
  "ieps": "false",
  "av": "false",
  "z": {
    "a": 452728321,
    "b": 1,
    "c": 0
  },
  "zh": "",
  "jsv": "1.5",
  "nav": {
    "userAgent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "appName": "Netscape",
    "appCodeName": "Mozilla",
    "appVersion": "5.0 (X11)",
    "appMinorVersion": 0,
    "product": "Gecko",
    "productSub": "20100101",
    "vendor": "",
    "vendorSub": "",
    "buildID": "20181001000000",
    "platform": "Linux x86_64",
    "oscpu": "Linux x86_64",
    "hardwareConcurrency": 12,
    "language": "en-US",
    "languages": ["en-US", "en"],
    "systemLanguage": 0,
    "userLanguage": 0,
    "doNotTrack": "1",
    "msDoNotTrack": 0,
    "cookieEnabled": true,
    "geolocation": 1,
    "vibrate": 1,
    "maxTouchPoints": 0,
    "webdriver": false,
    "plugins": []
  },
  "crc": {
    "window.chrome": "-not-existent"
  },
  "t": "204d2b34aa0a124e6c1007e535ce31fd9d9f6575",
  "u": "8f37ae19f8d35ddc9f5373fa3df091d3",
  "nap": "11133333331333333333",
  "fc": "true"
}
```  

This data is POSTed to 

    https://www.impfterminservice.de/akam/11/pixel_1afc1679
 
It's a lot of stuff. A list of installed fonts, a couple of browser features,
the graphics card, number of available CPU threads, etc.. and a few cryptic 
numbers. From my point of view this is nothing other than a 
**fingerprint** to (re-)identify a user system without relying on cookies. 

Probably a shortened version of this is repeatedly POSTed 
to a cryptic url

    https://www.impfterminservice.de/mB2r7RnT/KLD88p2/M-GlWn_/0j/7EiDGzVVOSh1/bjl5/Yj/ILYhxYFSw
    
The responsible javascript is delivered with a 
[GET request to that url](https://www.impfterminservice.de/mB2r7RnT/KLD88p2/M-GlWn_/0j/7EiDGzVVOSh1/bjl5/Yj/ILYhxYFSw).
It's highly obfuscated but can be inspected with a little effort. The script
places it's obfuscated strings in a variable called `_ac` so you can actually
read them or filter by regular expressions in the web console:

```js
_ac.filter(i => RegExp("[pP]lugin").exec(i))
```

There's also an object called `bmak` which holds all the collected data and
the functions that gather and transmit them. 

The privacy agreement states that [Matomo](https://matomo.org/) (formerly Piwik) 
is used for collection of user statistics but when searching the 
[matomo repository](https://github.com/matomo-org/matomo/) it's clear that 
Matomo is not responsible for the finger-printing.

The main developer behind **impfterminservice.de** seems to be 
[Oleg Fedak](https://jwebdev.github.io/) who claims to made experience
throughout the project with the whole stack of 
[elastisearch](https://www.elastic.co/elasticsearch/)
and [kubernetes](https://kubernetes.io/) microservices. 

Well, is this just the way how we setup infrastructure today? Just because
it's possible?

**There is certainly no rightful reason to fingerprint visitors of a federal
agencies website, at least in the European Union.**

And i'm not even talking about the purpose of the website, yet! Getting
an appointment and being vaccinated (potentially as a reaction to 
overestimation in the media or because of requirements by job, state and
infrastructure) leads into a nowadays fully digitized territory of 
health surveillance. Vaccination passport on your smartphone and all that. 

To say it bluntly: **We are encouraged to fight the enemy** 
(a virus in this case) **with full dedication, while we are actually
just fighting a way into complete monitoring of ourselves.**