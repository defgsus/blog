import sys
import os
import tempfile
import json as jsonlib
import hashlib
from pathlib import Path
from typing import Union, Optional, List

import requests


class CachedRequest:

    #BASE_CACHE_PATH: Path = Path(tempfile.gettempdir())
    BASE_CACHE_PATH: Path = Path(__file__).resolve().parent.parent.parent / "cache"

    def __init__(
            self,
            path: str,
            caching: Union[bool, str] = True,
            headers: Optional[dict] = None,
    ):
        self.caching = caching
        self.cache_path = self.BASE_CACHE_PATH / path
        self.headers = headers
        self.session = requests.Session()

    def request(self, url, json: bool = False, **kwargs) -> Union[bytes, dict, list]:
        filename = url.split("//", 1)[1]

        if self.headers:
            headers = self.headers
            if kwargs.get("headers"):
                headers = {
                    **self.headers,
                    **kwargs["headers"],
                }
            kwargs["headers"] = headers

        if kwargs:
            filename += "-" + hashlib.md5(repr(kwargs).encode("utf-8")).hexdigest()

        cache_filename = self.cache_path / filename

        content = None
        if self.caching in (True, "read"):
            if cache_filename.exists():
                content = cache_filename.read_bytes()

        if not content:
            print(f"requesting {url} {kwargs}", file=sys.stderr)
            response = self.session.get(
                url,
                **kwargs,
            )
            content = response.content

            if self.caching in (True, "write"):
                print(f"writing {cache_filename}", file=sys.stderr)
                os.makedirs(str(cache_filename.parent), exist_ok=True)
                cache_filename.write_bytes(content)

        if json:
            try:
                return jsonlib.loads(content.decode("utf-8"))
            except:
                print(content)
                raise

        return content
