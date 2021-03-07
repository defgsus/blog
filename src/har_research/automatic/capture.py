import os
import shutil
import time
import urllib.parse
import sys
import datetime
import json
import argparse
sys.path.insert(0, "..")

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


PATH = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "url", type=str,
        help="Url of website to capture (including protocol)"
    )
    parser.add_argument(
        "-i", "--interactive", type=bool, nargs="?", default=False, const=True,
        help="Opens an interactive ipython session when page is loaded"
    )
    parser.add_argument(
        "-a", "--accept-consent", type=bool, nargs="?", default=False, const=True,
        help="Try to accept the cookie consent"
    )
    parser.add_argument(
        "-r", "--require-consent", type=bool, nargs="?", default=False, const=True,
        help="Require accepting the cookie consent"
    )
    parser.add_argument(
        "--wait", type=int, nargs="?", default=1,
        help="Wait number of seconds before exiting (default 1)"
    )
    parser.add_argument(
        "--bundle-extension", type=bool, nargs="?", default=False, const=True,
        help="Bundles the 'har-export-trigger' to a zip file, even if present"
    )

    return parser.parse_args()


def printe(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


class CaptureError(Exception):
    pass


class Extension:
    def __init__(self, short_path: str):
        self.short_path = short_path
        self.long_path = os.path.join(PATH, short_path)
        self.zip_path = self.long_path + ".zip"

    def get_zip(self, repackage=False):
        if repackage or not os.path.exists(self.zip_path):
            printe("creating extension package", self.short_path + ".zip")
            shutil.make_archive(self.long_path, "zip", self.long_path)
        return self.zip_path


class Capture:

    EXTENSION_PATH = os.path.join(PATH, )
    EXTENSION_ZIP = EXTENSION_PATH + ".zip"
    RECORDING_PATH = os.path.join(PATH, "recordings")

    def __init__(
            self,
            url: str,
            headless: bool = False,
            repackage_extensions: bool = False,
    ):
        self.url = url
        self.timestamp = datetime.datetime.now()
        self._script_helper = None
        self.repackage_extensions = repackage_extensions
        self.extension = Extension("extension")
        self.extension_har = Extension("har-export-trigger")

        options = webdriver.FirefoxOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--devtools")

        profile = webdriver.FirefoxProfile()
        profile.set_preference("devtools.toolbox.selectedTool", "netmonitor")
        profile.set_preference("devtools.netmonitor.persistlog", True)

        self.browser = webdriver.Firefox(
            executable_path="./geckodriver",
            firefox_options=options,
            firefox_profile=profile,
        )

    def install_extension(self, ext: Extension):
        self.browser.install_addon(
            ext.get_zip(repackage=self.repackage_extensions),
            temporary=True,
        )

    def recording_path(self, create=False):
        url = urllib.parse.urlparse(self.url)
        path = os.path.join(
            self.RECORDING_PATH,
            url.netloc,
        )
        if create and not os.path.exists(path):
            printe(f"creating {path}")
            os.makedirs(path)
        return path

    def run(
            self,
            accept_consent: bool = True,
            require_consent: bool = True,
            scroll_page_down: bool = True,
            stay_seconds: float = 1,
            interactive: bool = False,
    ):
        try:
            try:
                self.browser.get(self.url)
                self.install_extension(self.extension)
                time.sleep(1)

                if interactive:
                    self.interactive_shell()

                if accept_consent or require_consent:
                    time.sleep(3)  # give some time to display consent
                    result = self.accept_consent()
                    if not result and require_consent:
                        raise CaptureError(
                            f"Could not accept privacy consent in {self.browser.current_url}"
                        )

                if scroll_page_down:
                    self.scroll_page(down=True)

                time.sleep(stay_seconds)

            except KeyboardInterrupt:
                pass

            self.install_extension(self.extension_har)
            time.sleep(1)
            self.store_recording()

        finally:
            self.browser.quit()

    def interactive_shell(self):
        import IPython
        IPython.embed(header=f"""Hi there, you are visiting {self.url}

self == {self}
self.browser == {self.browser}

Press CTRL+D to exit. Good luck!       
        """)

    def actions(self):
        return ActionChains(self.browser)

    def body(self):
        return self.browser.find_element_by_tag_name("body")

    def store_recording(self):
        log = self.browser.execute_async_script(
            "HAR.triggerExport().then(arguments[0]);"
        )

        recording = {
            "log": log,
        }

        path = self.recording_path(create=True)
        filename = self.timestamp.strftime("%Y-%m-%d-%H-%M-%S.json")
        with open(os.path.join(path, filename), "w") as fp:
            json.dump(recording, fp)

        printe(f"stored {len(log['entries'])} HAR entries to {filename}")

    def sleep_after_consent_click(self):
        time.sleep(3)

    def accept_consent(self):
        from accept_consent import accept_consent
        return accept_consent(self)

    def scroll_page(
            self,
            down: bool = True,
            interval: float = .5,
            max_scroll: float = 20000.,
    ):
        height = self.body().size["height"]
        _, last_y = self.get_scroll_pos()
        init_y = last_y
        while True:
            self.body().send_keys(Keys.PAGE_DOWN if down else Keys.PAGE_UP)
            time.sleep(interval)
            x, y = self.get_scroll_pos()
            if y == last_y:
                printe(f"end of scrolling at {y} (height {round(height)})")
                break
            if abs(y - init_y) > max_scroll:
                printe(f"end of scrolling at {y} after {abs(y - init_y)} (height {round(height)})")
                break
            printe(f"scrolled from {last_y} to {y} (height {round(height)})")
            last_y = y

    def get_scroll_pos(self):
        return tuple(self.browser.execute_script('return [window.scrollX, window.scrollY]'))

    def run_script(self, script: str):
        """
        Run script with some helper functions
        """
        if not self._script_helper:
            with open(os.path.join(self.PATH, "helper.js")) as fp:
                helper = fp.read()

        return self.browser.execute_script(
            helper + "\n" + script,
        )

    def query_selector(self, selector: str, **filters):
        elems = []
        for elem in self.run_script(f"""
            return dom_elements_to_object(document.querySelectorAll("{selector}"))
        """):
            matches = True
            for key, value in filters.items():
                if key not in elem:
                    if value not in elem["key"]:
                        matches = False
                        break
            if matches:
                elems.append(elem)

        return elems


if __name__ == "__main__":

    args = parse_args()

    cap = Capture(
        url=args.url,
        repackage_extensions=args.bundle_extension,
    )

    cap.run(
        stay_seconds=args.wait,
        accept_consent=args.accept_consent,
        require_consent=args.require_consent,
        interactive=args.interactive,
    )
