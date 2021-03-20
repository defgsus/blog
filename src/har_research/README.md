## HAR Research

It's mainly about analyzing web traffic but i find 
*[HAR](https://en.wikipedia.org/wiki/HAR_\(file_format\))s Research* a cool name.


#### some urchin info links

uMatrix will block some of the sites, but it's worth to temporarily allow
first-party CSS if you're interested in some ad tech points of view.

- https://clearcode.cc/blog/
- https://www.adexchanger.com/the-sell-sider/
- https://github.com/prebid/prebid-server
- https://github.com/prebid/headerbid-expert
- https://www.outbrain.com/blog/programmatic-advertising/



## random data

```
# sale on teads.tv
------------------------------------------------------------------------------------------------------------------------------------------------------------
Url: https://a.teads.tv/hb/bid-request
Method: POST
Date: 2021-03-08T22:28:24.538+01:00
Headers:
  Host            : a.teads.tv
  User-Agent      : Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0
  Accept          : */*
  Accept-Language : en-US,en;q=0.5
  Accept-Encoding : gzip, deflate, br
  Content-Type    : text/plain
  Content-Length  : 881
  Origin          : https://www.waz.de
  Connection      : keep-alive
  Referer         : https://www.waz.de/
Post:
{"referrer":"https://www.waz.de/","pageReferrer":"","networkBandwidth":"","timeToFirstByte":"274","data":[{"sizes":["300x250","600x800"],"bidId":"4d9b300511b1a4","bidderRequestId":"39a94113c642fe","placementId":104805,"pageId":96732,"adUnitCode":"qpb-ban1","auctionId":"3f5afa75-6c84-4792-b18c-df2ff83a2e75","transactionId":"ba879cbe-d016-4fb9-9033-7d4d7d0c8196"}],"deviceWidth":1920,"hb_version":"4.20.0-pre","gdpr_iab":{"consent":"CPCweWLPCweWLAfHkCDEBQCsAP_AAH_AAAYgHUtZ9DpGbXFCcXx9YMsUKYRf1tRXA2QiChSBg2AFSEOEsJwEkWAAAASgoCAAgQ4AolYBAAVEDEAEAAEAQAEVAAGsAwAEhAAIICJAEAEBCEAAAAgAAAAAABAAgEgZiGQImBBEA-PoRGAIiogwBgAAKIgAgIAFAoIHUtZ9DpGbXFCcXx9YMsUKYRf1tRXA2QiChSBg2AFSEOEsJwEkWAAAASgoCAAgQ4AolYBAAVEDEAEAAEAQAEVAAGsAwAEhAAIICJAEAEBCEAAAAgAAAAAABAAgEgZiGQImBBEA-PoRGAIiogwBgAAKIgAgIAFAoIBQMACAtoKABAW0HAAgLaEgAQFtCwAIC2hoAEBbQ8ACAtoiABAW0TAAgLaKgAQFtA","status":12,"apiVersion":2}}
Response headers:
  content-type                     : application/json
  access-control-allow-credentials : true
  access-control-allow-origin      : https://www.waz.de
  content-encoding                 : gzip
  content-length                   : 301
  vary                             : Accept-Encoding
  expires                          : Mon, 08 Mar 2021 21:28:25 GMT
  cache-control                    : max-age=0, no-cache, no-store
  pragma                           : no-cache
  date                             : Mon, 08 Mar 2021 21:28:25 GMT
  set-cookie                       : cs=1;Domain=.teads.tv;max-age=2592000;path=/;SameSite=None;Secure
  X-Firefox-Spdy                   : h2
Content:
{"responses":[{"placementId":104805,"transactionId":"ba879cbe-d016-4fb9-9033-7d4d7d0c8196","bidId":"4d9b300511b1a4","ttl":150,"creativeId":"554250","netRevenue":true,"currency":"EUR","cpm":0.57,"width":"1","height":"1","ad":"<script type=\"text/javascript\" class=\"teads\" async=\"true\" src=\"https://a.teads.tv/hb/ad/ba879cbe-d016-4fb9-9033-7d4d7d0c8196_b27e24f9-5f41-4bb0-be54-dd5f65387aef\"></script>"}]}
```

#### permutive 

```
Url: https://api.permutive.com/v2.0/batch/events
Method: POST
Date: 2021-03-08T22:28:23.489+01:00
Headers:
  Host            : api.permutive.com
  User-Agent      : Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0
  Accept          : */*
  Accept-Language : en-US,en;q=0.5
  Accept-Encoding : gzip, deflate, br
  Content-Type    : text/plain
  Content-Length  : 5260
  Origin          : https://www.waz.de
  Connection      : keep-alive
  Referer         : https://www.waz.de/
Query:
  enrich : false
  sdkp   : true
  k      : 7d66f832-46f2-4f67-9929-5ce2f7bb9f86
Post:
[
  {
    "user_id": "d599f282-bf48-4ae9-8865-7aa777bdb150",
    "name": "Pageview",
    "segments": [],
    "properties": {
      "isp_info": {
        "isp": "O2 Deutschland",
        "organization": "O2 Deutschland",
        "autonomous_system_number": 6805,
        "autonomous_system_organization": "Telefonica Germany"
      },
      "geo_info": {
        "continent": "Europe",
        "country": "Germany",
        "city": "Berlin",
        "province": "Land Berlin",
        "postal_code": null
      },
      "classifications_watson": {
        "categories": [
          {
            "label": "/food and drink",
            "score": 0.625519
          },
          {
            "label": "/law, govt and politics",
            "score": 0.600567
          },
          {
            "label": "/society/unrest and war",
            "score": 0.567638
          }
        ],
        "concepts": [
          {
            "text": "Vergaberecht",
            "relevance": 0.820531
          },
          {
            "text": "Bundesgesetz",
            "relevance": 0.586387
          },
          {
            "text": "Grundgesetz f\u00fcr die Bundesrepublik Deutschland",
            "relevance": 0.578236
          },
          {
            "text": "Sport",
            "relevance": 0.506776
          }
        ],
        "entities": [
          {
            "text": "100.000",
            "relevance": 0.94975
          },
          {
            "text": "wenige Tage",
            "relevance": 0.860468
          },
          {
            "text": "Favre",
            "relevance": 0.817573
          },
          {
            "text": "Union Berlin",
            "relevance": 0.78467
          },
          {
            "text": "Essen",
            "relevance": 0.762241
          },
          {
            "text": "10",
            "relevance": 0.627957
          },
          {
            "text": "Dortmund",
            "relevance": 0.440313
          },
          {
            "text": "Ruhrparlament",
            "relevance": 0.38717
          },
          {
            "text": "BVB",
            "relevance": 0.376314
          },
          {
            "text": "16",
            "relevance": 0.312096
          },
          {
            "text": "18",
            "relevance": 0.31154
          },
          {
            "text": "TV",
            "relevance": 0.305717
          },
          {
            "text": "Ermittlerinnen",
            "relevance": 0.259832
          },
          {
            "text": "erste",
            "relevance": 0.177701
          },
          {
            "text": "110",
            "relevance": 0.168659
          },
          {
            "text": "M\u00fcnchen",
            "relevance": 0.160599
          },
          {
            "text": "Bundesregierung",
            "relevance": 0.140473
          },
          {
            "text": "ersten",
            "relevance": 0.108008
          },
          {
            "text": "FFP",
            "relevance": 0.083258
          },
          {
            "text": "Berlin",
            "relevance": 0.074229
          }
        ],
        "keywords": [
          {
            "text": "L\u00f6schen der Browser-Cookies",
            "relevance": 0.619917
          },
          {
            "text": "Vergabe von FFP2-Masken",
            "relevance": 0.619917
          },
          {
            "text": "ganz gro\u00dfem Gl\u00fcck",
            "relevance": 0.615344
          },
          {
            "text": "Union Berlin",
            "relevance": 0.588999
          },
          {
            "text": "aktuellsten Nachrichten angezeigt",
            "relevance": 0.585938
          },
          {
            "text": "erste Bew\u00e4hrungsprobe",
            "relevance": 0.572283
          },
          {
            "text": "Bitte",
            "relevance": 0.55028
          },
          {
            "text": "Bedarf gro\u00df",
            "relevance": 0.533249
          },
          {
            "text": "neue Gewerbefl\u00e4chen",
            "relevance": 0.532434
          },
          {
            "text": "weiteren Gewinnen",
            "relevance": 0.529389
          },
          {
            "text": "Vorg\u00e4nger Favre",
            "relevance": 0.529389
          },
          {
            "text": "Free-TV",
            "relevance": 0.529389
          },
          {
            "text": "ersten Folgen",
            "relevance": 0.523348
          },
          {
            "text": "Bundesregierung",
            "relevance": 0.502016
          },
          {
            "text": "MillionenKracher",
            "relevance": 0.498335
          },
          {
            "text": "Terzic",
            "relevance": 0.487886
          },
          {
            "text": "R\u00fcckrundenauftakt",
            "relevance": 0.487886
          },
          {
            "text": "Tatort",
            "relevance": 0.487886
          },
          {
            "text": "Ruhrparlament",
            "relevance": 0.487327
          },
          {
            "text": "Partien",
            "relevance": 0.479944
          },
          {
            "text": "Revierst\u00e4dte",
            "relevance": 0.479944
          },
          {
            "text": "Verordnung",
            "relevance": 0.476238
          },
          {
            "text": "Verein",
            "relevance": 0.467863
          },
          {
            "text": "Tage",
            "relevance": 0.46709
          },
          {
            "text": "Polizeiruf",
            "relevance": 0.46709
          },
          {
            "text": "Spiel",
            "relevance": 0.462814
          },
          {
            "text": "Bundesligaspieltagen",
            "relevance": 0.462814
          },
          {
            "text": "Amt",
            "relevance": 0.459816
          },
          {
            "text": "BVB",
            "relevance": 0.459816
          },
          {
            "text": "Vereinsauswahl",
            "relevance": 0.456044
          },
          {
            "text": "Ermittlerinnen",
            "relevance": 0.456044
          },
          {
            "text": "Risikogruppen",
            "relevance": 0.456044
          },
          {
            "text": "Essen",
            "relevance": 0.453714
          },
          {
            "text": "Spielscheine",
            "relevance": 0.453714
          },
          {
            "text": "Dortmund",
            "relevance": 0.453714
          },
          {
            "text": "M\u00fcnchen",
            "relevance": 0.453714
          },
          {
            "text": "F\u00fchrung",
            "relevance": 0.453714
          },
          {
            "text": "Berlin",
            "relevance": 0.448238
          }
        ],
        "watson_sentiment": {
          "label": "positive",
          "score": 0.479765
        }
      },
      "sectionL1": "",
      "sectionL2": "",
      "sectionL3": "",
      "article": {
        "articleID": "",
        "contentType": ""
      },
      "user": {
        "login": ""
      },
      "client": {
        "type": "web",
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0",
        "url": "https://www.waz.de/",
        "domain": "www.waz.de",
        "title": "Deutschlands gr\u00f6\u00dfte Regionalzeitung - waz.de",
        "referrer": ""
      }
    },
    "session_id": "157fdb7f-31ec-4acf-9491-617d7817730e",
    "view_id": "ebe8dddb-c2a5-4bbd-8084-23cc2f0f3467"
  },
  {
    "user_id": "d599f282-bf48-4ae9-8865-7aa777bdb150",
    "name": "SegmentEntry",
    "segments": [
      50032,
      50431,
      61610
    ],
    "properties": {
      "segment_number": 50032,
      "client": {
        "type": "web",
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0",
        "url": "https://www.waz.de/",
        "domain": "www.waz.de",
        "title": "Deutschlands gr\u00f6\u00dfte Regionalzeitung - waz.de",
        "referrer": ""
      }
    },
    "session_id": "157fdb7f-31ec-4acf-9491-617d7817730e",
    "view_id": "ebe8dddb-c2a5-4bbd-8084-23cc2f0f3467"
  },
  {
    "user_id": "d599f282-bf48-4ae9-8865-7aa777bdb150",
    "name": "SegmentEntry",
    "segments": [
      50032,
      50431,
      61610
    ],
    "properties": {
      "segment_number": 50431,
      "client": {
        "type": "web",
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0",
        "url": "https://www.waz.de/",
        "domain": "www.waz.de",
        "title": "Deutschlands gr\u00f6\u00dfte Regionalzeitung - waz.de",
        "referrer": ""
      }
    },
    "session_id": "157fdb7f-31ec-4acf-9491-617d7817730e",
    "view_id": "ebe8dddb-c2a5-4bbd-8084-23cc2f0f3467"
  },
  {
    "user_id": "d599f282-bf48-4ae9-8865-7aa777bdb150",
    "name": "SegmentEntry",
    "segments": [
      50032,
      50431,
      61610
    ],
    "properties": {
      "segment_number": 61610,
      "client": {
        "type": "web",
        "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:86.0) Gecko/20100101 Firefox/86.0",
        "url": "https://www.waz.de/",
        "domain": "www.waz.de",
        "title": "Deutschlands gr\u00f6\u00dfte Regionalzeitung - waz.de",
        "referrer": ""
      }
    },
    "session_id": "157fdb7f-31ec-4acf-9491-617d7817730e",
    "view_id": "ebe8dddb-c2a5-4bbd-8084-23cc2f0f3467"
  }
]

Response headers:
  date                             : Mon, 08 Mar 2021 21:28:24 GMT
  content-type                     : application/json
  content-encoding                 : gzip
  access-control-expose-headers    : *
  vary                             : Origin,Access-Control-Request-Method
  access-control-allow-credentials : true
  access-control-allow-methods     : POST
  access-control-allow-origin      : https://www.waz.de
  access-control-max-age           : 86400
  server                           : Permutive
  content-length                   : 188
  via                              : 1.1 google
  alt-svc                          : clear
  X-Firefox-Spdy                   : h2
Content:
[{"code":200,"body":{"id":"b83aca9c-b546-4895-b28f-724d0d2d9176","time":"2021-03-08T21:28:24.243Z"}},{"code":200,"body":{"id":"a31d300e-b3f6-4783-b51c-8b46c162288f","time":"2021-03-08T21:28:24.243Z"}},{"code":200,"body":{"id":"a246d7bb-c4dd-40e4-affb-f860d36fee6f","time":"2021-03-08T21:28:24.243Z"}},{"code":200,"body":{"id":"5ab892cd-05cd-4532-b39a-edb8dffeeeca","time":"2021-03-08T21:28:24.243Z"}}]
```