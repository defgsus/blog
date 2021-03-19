"""
No no! I do not build a database!
I'm just caching results
"""
import os
import subprocess
from typing import Optional
import argparse


class WhoisCache:

    PATH = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "whois-cache",
    )

    __instance = None

    @classmethod
    def instance(cls):
        if not cls.__instance:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        self._objects = dict()

    def get_best_effort(self, name: str, ask_the_web: bool = False) -> dict:
        try:
            from .har import parse_url
        except ImportError:
            from har import parse_url

        content = WhoisCache.instance().get("whois", name)
        if not content and ask_the_web:
            if parse_url(name)["short_host"] == name:
                content = run_whois(name)

        data = whois_to_dict(content or "", from_ip=False)

        content = WhoisCache.instance().get("nslookup", name)
        if not content and ask_the_web:
            content = run_nslookup(name)

        if content:
            ip = nslookup_to_ip(content)
            if ip:
                content = WhoisCache.instance().get("whois", ip)
                if not content and ask_the_web:
                    content = run_whois(ip)

                if content:
                    ip_data = whois_to_dict(content, from_ip=True)
                    for k, v in ip_data.items():
                        if v:
                            data[k] = v

                #if not data["network_country"]:
                #    print(content)

        return data

    def get_dict(self, type: str, id: str) -> dict:
        text = self.get(type, id)
        if type == "whois":
            return whois_to_dict(text or "")
        elif type == "nslookup":
            return {"ip": nslookup_to_ip(text or "")}
        else:
            raise ValueError(f"Invalid type #{type}'")

    def get(self, type: str, id: str) -> Optional[str]:
        if type not in self._objects:
            self._objects[type] = dict()

        if id not in self._objects[type]:
            filename = os.path.join(self.PATH, type, f"{id}.txt")
            if os.path.exists(filename):
                with open(filename) as fp:
                    self._objects[type][id] = fp.read()

        if id in self._objects[type]:
            return self._objects[type][id]

    def store(self, type: str, id: str, content: str):
        if type not in self._objects:
            self._objects[type] = dict()

        self._objects[type][id] = content

        filepath = os.path.join(self.PATH, type)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = os.path.join(filepath, f"{id}.txt")

        with open(filename, "w") as fp:
            fp.write(content)


def run_whois(obj: str, read_cache: bool = True, store_cache: bool = True) -> str:
    return run_("whois", obj, read_cache, store_cache)


def run_nslookup(obj: str, read_cache: bool = True, store_cache: bool = True) -> str:
    return run_("nslookup", obj, read_cache, store_cache)


def run_(what: str, obj: str, read_cache: bool = True, store_cache: bool = True) -> str:
    if read_cache:
        content = WhoisCache.instance().get(what, obj)
        if content:
            return content

    try:
        content = subprocess.check_output([what, obj])

        for encoding in ("utf-8", "latin1"):
            try:
                content = content.decode(encoding)
                break
            except UnicodeDecodeError:
                pass

        if isinstance(content, bytes):
            print(f"DECODING ERROR for {what} '{obj}'")
            print(content)
            raise ValueError(f"Decoding error for '{obj}'")

    except subprocess.CalledProcessError as e:
        content = f"ERROR {e}"

    if store_cache:
        WhoisCache.instance().store(what, obj, content)

    return content


WHOIS_KEYS = {
    "registrant": [
        "Registrant Organization",
    ],
    "registrant_country": [
        "Registrant Country",
    ],
    "registrar": [
        "Registrar",
    ],
    "name_server": [
        "Name Server",
        "Nserver",
    ],
    "network": [
        "OrgName",
        "org-name",
        "descr",
        "role",
        "mnt-by",
        "netname",
    ],
    "network_country": [
        "country",
        "Country",
    ],
}

# couple of string parts to ignore
WHOIS_IGNORE_RESPONSE = (
    "REDACTED FOR PRIVACY",
    "DATA REDACTED",
    "Not Disclosed",
    "Whois Privacy Service",
    "WhoisGuard",
    "Domain Privacy Service FBO Registrant.",
    "Domain Protection Services",
    "Contact Privacy Inc. Customer",
    "PrivateName Services",
    "c/o whoisproxy.com",
    "Digital Privacy Corporation",
    "Transferred to the RIPE region",
    "Registrant of",
    "Redacted for Privacy",
    "Domains By Proxy",
    "Data Protected",
    "Domain Privacy Limited",
)

WHOIS_IGNORE_RESPONSE_REGISTRANT = (
    "Hetzner Online GmbH",
    "Boreus Rechenzentrum GmbH",
    "Infrastructure",
)


def whois_to_dict(text: str, from_ip: bool = False) -> dict:
    ret = {
        key: None
        for key in WHOIS_KEYS
    }
    for section_key, keys in WHOIS_KEYS.items():
        for key in keys:
            try:
                idx = text.index(key + ":")
                value = text[idx+len(key)+1:].split("\n")[0].strip()
                if not value:
                    continue
                do_ignore = False
                for ignore in WHOIS_IGNORE_RESPONSE:
                    if value in ignore or ignore in value:
                        do_ignore = True
                        break

                if section_key == "registrant":
                    for ignore in WHOIS_IGNORE_RESPONSE_REGISTRANT:
                        if value in ignore or ignore in value:
                            do_ignore = True
                            break

                if not do_ignore:
                    ret[section_key] = value
                    break
            except ValueError:
                pass

    if not from_ip:
        ret.update({"network": None, "network_country": None})
    return ret


def nslookup_to_ip(text: str) -> Optional[str]:
    while True:
        try:
            idx = text.index("Address:")
        except ValueError:
            return

        ip = text[idx+8:].split("\n")[0].strip()
        if not ip.startswith("127."):
            return ip

        text = text[idx+8:]



def run_test():
    host_names = ['1rx.io', '2mdn.net', '360yield.com', '3lift.com', '3qsdn.com', '71i.de', '_.rocks', 'aachener-zeitung.de', 'abendblatt.de', 'abendzeitung.de', 'ablida.net', 'abtasty.com', 'ad-production-stage.com', 'ad-server.eu', 'ad-srv.net', 'ad.gt', 'ad4m.at', 'ad4mat.net', 'adalliance.io', 'adcell.com', 'addthis.com', 'addthisedge.com', 'addtoany.com', 'adform.net', 'adition.com', 'adkernel.com', 'adlooxtracking.com', 'adnxs.com', 'adobedtm.com', 'adrtx.net', 'ads-twitter.com', 'adsafeprotected.com', 'adsafety.net', 'adscale.de', 'adspirit.de', 'adsrvr.org', 'adup-tech.com', 'advertising.com', 'agkn.com', 'akamaihd.net', 'akamaized.net', 'akstat.io', 'alexametrics.com', 'allesregional.de', 'allgemeine-zeitung.de', 'amazon-adsystem.com', 'amazonaws.com', 'aminopay.net', 'ampproject.org', 'aniview.com', 'app.link', 'appier.net', 'appspot.com', 'arcgis.com', 'architekturzeitung.net', 'artefact.com', 'artikelscore.de', 'asadcdn.com', 'aspnetcdn.com', 'atdmt.com', 'aticdn.net', 'atonato.de', 'audiencemanager.de', 'awin.com', 'awin1.com', 'aws-cbc.cloud', 'az-muenchen.de', 'b-cdn.net', 'bannersnack.com', 'batch.com', 'berliner-zeitung.de', 'bf-ad.net', 'bf-tools.net', 'bfops.io', 'biallo.de', 'biallo3.de', 'bidr.io', 'bidswitch.net', 'bildstatic.de', 'bing.com', 'bit.ly', 'bitmovin.com', 'blau.de', 'bluekai.com', 'bluesummit.de', 'boltdns.net', 'bootstrapcdn.com', 'bottalk.io', 'branch.io', 'brandmetrics.com', 'brealtime.com', 'brightcove.com', 'brightcove.net', 'brillen.de', 'bttrack.com', 'bundestag.de', 'c-i.as', 'casalemedia.com', 'cdn-solution.net', 'cdntrf.com', 'chartbeat.com', 'chartbeat.net', 'cheqzone.com', 'chimpstatic.com', 'clarium.io', 'cleverpush.com', 'clickagy.com', 'cloudflare.com', 'cloudfront.net', 'cloudfunctions.net', 'cloudimg.io', 'cmcdn.de', 'commander1.com', 'conative.de', 'congstar.de', 'conrad.com', 'conrad.de', 'consensu.org', 'consentric.de', 'content-garden.com', 'contentinsights.com', 'contentspread.net', 'contextweb.com', 'cookiebot.com', 'cookielaw.org', 'cookiepro.com', 'crazyegg.com', 'createjs.com', 'creative-serving.com', 'creativecdn.com', 'criteo.com', 'criteo.net', 'crwdcntrl.net', 'ctnsnet.com', 'cxense.com', 'cxpublic.com', 'datawrapper.de', 'de.com', 'demdex.net', 'derwesten.de', 'df-srv.de', 'disqus.com', 'disquscdn.com', 'districtm.io', 'dnacdn.net', 'dotomi.com', 'doubleclick.net', 'doubleverify.com', 'dreilaenderschmeck.de', 'dspx.tv', 'dumontnet.de', 'dumontnext.de', 'dwcdn.net', 'dwin1.com', 'dymatrix.cloud', 'e-pages.dk', 'ebayadservices.com', 'ebaystatic.com', 'emetriq.de', 'emsservice.de', 'emxdgt.com', 'eon.de', 'erne.co', 'ethinking.de', 'everesttech.net', 'exactag.com', 'exelator.com', 'f11-ads.com', 'f11-ads.net', 'facebook.com', 'facebook.net', 'fanmatics.com', 'fastly.net', 'fazcdn.net', 'fbcdn.net', 'finance.si', 'finanzen100.de', 'flashtalking.com', 'flourish.studio', 'fontawesome.com', 'fonts.net', 'freiepresse-display.de', 'fupa.net', 'futalis.de', 'ga.de', 'geoedge.be', 'getback.ch', 'ggpht.com', 'githubusercontent.com', 'glomex.cloud', 'glomex.com', 'go-mpulse.net', 'goo.gl', 'google-analytics.com', 'google.com', 'google.de', 'googleadservices.com', 'googleapis.com', 'googleoptimize.com', 'googlesyndication.com', 'googletagmanager.com', 'googletagservices.com', 'googleusercontent.com', 'googlevideo.com', 'gravatar.com', 'gscontxt.net', 'gstatic.com', 'h-cdn.com', 'hariken.co', 'haz.de', 'heilbronnerstimme.de', 'hotjar.com', 'hotjar.io', 'hs-data.com', 'hs-edge.net', 'hscta.net', 'hstrck.com', 'hubspot.com', 'hubspotusercontent20.net', 'ibillboard.com', 'ibytedtos.com', 'icony-hosting.de', 'icony.com', 'id5-sync.com', 'idcdn.de', 'igodigital.com', 'igstatic.com', 'ikz-online.de', 'imgix.net', 'imrworldwide.com', 'indexww.com', 'indivsurvey.de', 'infogram.com', 'inforsea.com', 'instagram.com', 'intellitxt.com', 'intercom.io', 'intercomcdn.com', 'ioam.de', 'ippen.space', 'iqdigital.de', 'ix.de', 'jifo.co', 'jobs-im-suedwesten.de', 'jquery.com', 'jsdelivr.net', 'justpremium.com', 'kaloo.ga', 'kaltura.com', 'kameleoon.eu', 'kaspersky.com', 'kobel.io', 'krxd.net', 'lead-alliance.net', 'leasewebultracdn.com', 'liadm.com', 'licdn.com', 'ligatus.com', 'lijit.com', 'linkedin.com', 'list-manage.com', 'lkqd.net', 'ln-online.de', 'localhost', 'loggly.com', 'lp4.io', 'lr-digital.de', 'm-t.io', 'm6r.eu', 'madsack-native.de', 'mailchimp.com', 'main-echo-cdn.de', 'mannheimer-morgen.de', 'marktjagd.com', 'marktjagd.de', 'mateti.net', 'mathtag.com', 'media-amazon.com', 'media01.eu', 'medialead.de', 'mediamathtag.com', 'medienhausaachen.de', 'meetrics.net', 'meine-vrm.de', 'meinsol.de', 'mfadsrvr.com', 'mgaz.de', 'ml314.com', 'mlsat02.de', 'moatads.com', 'mookie1.com', 'motoso.de', 'mpnrs.com', 'msgp.pl', 'mxcdn.net', 'mycleverpush.com', 'myfonts.net', 'nativendo.de', 'netdna-ssl.com', 'netpoint-media.de', 'nexx.cloud', 'nmrodam.com', 'noz-cdn.de', 'noz.de', 'npttech.com', 'nrz.de', 'nuggad.net', 'o2online.de', 'oadts.com', 'oberpfalzmedien.de', 'oecherdeal.de', 'offerista.com', 'office-partner.de', 'omnitagjs.com', 'omniv.io', 'omny.fm', 'omnycontent.com', 'omsnative.de', 'omtrdc.net', 'onaudience.com', 'onesignal.com', 'onetag-sys.com', 'onetrust.com', 'onthe.io', 'opecloud.com', 'opencmp.net', 'openx.net', 'opinary.com', 'otto.de', 'outbrain.com', 'outbrainimg.com', 'ovb24.de', 'parsely.com', 'paypal.com', 'paypalobjects.com', 'peiq.de', 'perfectmarket.com', 'permutive.app', 'permutive.com', 'piano.io', 'pingdom.net', 'pinpoll.com', 'plenigo.com', 'plista.com', 'plyr.io', 'pnp.de', 'podigee-cdn.net', 'podigee.com', 'podigee.io', 'polyfill.io', 'prebid.org', 'pressekompass.net', 'privacy-mgmt.com', 'prmutv.co', 'pubmatic.com', 'pubmine.com', 'purelocalmedia.de', 'pushengage.com', 'pushwoosh.com', 'quantcount.com', 'quantserve.com', 'rackcdn.com', 'ravenjs.com', 'rawgit.com', 'rawr.at', 'recognified.net', 'redintelligence.net', 'reisereporter.de', 'resetdigital.co', 'retailads.net', 'rfihub.com', 'richaudience.com', 'rlcdn.com', 'rmbl.ws', 'rnd.de', 'rndtech.de', 'rp-online.de', 'rqtrk.eu', 'rta-design.de', 'rtclx.com', 'rtmark.net', 'rubiconproject.com', 'rumble.com', 'rvty.net', 's-i-r.de', 's-onetag.com', 's-p-m.ch', 's4p-iapps.com', 'sascdn.com', 'scdn.co', 'scene7.com', 'scorecardresearch.com', 'selfcampaign.com', 'semasio.net', 'serving-sys.com', 'showheroes.com', 'shz.de', 'sitescout.com', 'slgnt.eu', 'smartadserver.com', 'smartclip.net', 'smartstream.tv', 'sonobi.com', 'sp-prod.net', 'sparwelt.click', 'speedcurve.com', 'sphere.com', 'sportbuzzer.de', 'spotify.com', 'spotxchange.com', 'springer.com', 'sqrt-5041.de', 'ssl-images-amazon.com', 'stackpathdns.com', 'stellenanzeigen.de', 'stickyadstv.com', 'stroeerdigital.de', 'stroeerdigitalgroup.de', 'stroeerdigitalmedia.de', 'stuttgarter-zeitung.de', 't.co', 'taboola.com', 'tchibo.de', 'teads.tv', 'technical-service.net', 'technoratimedia.com', 'telefonica-partner.de', 'telekom.de', 'theadex.com', 'theepochtimes.com', 'tickaroo.com', 'tiktok.com', 'tiktokcdn.com', 'tinypass.com', 'tiqcdn.com', 'transmatico.com', 'trauer-im-allgaeu.de', 'tremorhub.com', 'trmads.eu', 'trmcdn.eu', 'trustarc.com', 'trustcommander.net', 'truste.com', 'turn.com', 'twiago.com', 'twimg.com', 'twitter.com', 'typekit.net', 'typography.com', 'unbounce.com', 'unicef.de', 'unpkg.com', 'unrulymedia.com', 'uobsoe.com', 'upscore.com', 'urban-media.com', 'uri.sh', 'usabilla.com', 'usercentrics.eu', 'userreport.com', 'vgwort.de', 'vhb.de', 'vi-serve.com', 'vidazoo.com', 'videoreach.com', 'viralize.tv', 'visx.net', 'vlyby.com', 'vodafone.de', 'voltairenet.org', 'vtracy.de', 'vxcp.de', 'wallstreet-online.de', 'warenform.de', 'wbtrk.net', 'wcfbc.net', 'webgains.com', 'webgains.io', 'weekli.de', 'weekli.systems', 'welect.de', 'welt.de', 'wetterkontor.de', 'wfxtriggers.com', 'windows.net', 'wlct-one.de', 'wordlift.io', 'wordpress.com', 'wp.com', 'wp.de', 'wr.de', 'wrzmty.com', 'wt-safetag.com', 'wz-media.de', 'xiti.com', 'xplosion.de', 'yagiay.com', 'yahoo.com', 'yieldlab.net', 'yieldlove-ad-serving.net', 'yieldlove.com', 'yieldscale.com', 'yimg.com', 'yoochoose.net', 'youtube-nocookie.com', 'youtube.com', 'ytimg.com', 'yumpu.com', 'zemanta.com', 'zenaps.com', 'zencdn.net', 'zeotap.com']
    for host_name in host_names:
        data = WhoisCache().instance().get_best_effort(host_name, ask_the_web=True)
        print(f"{host_name:30} {data}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "object", type=str, nargs="+",
        help="The 'object' to search for, e.g. domain name or IP address",
    )

    args = parser.parse_args()

    if args.object == ["test"]:
        run_test()
        exit(0)

    for obj in args.object:
        print(f"\n------------------ whois {obj} ------------------\n")
        text = run_whois(obj)
        print(text)


