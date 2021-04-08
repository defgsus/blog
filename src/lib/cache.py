import os
import hashlib
import json
from typing import Callable, Any
import sys


class FileCache:

    verbose = False

    PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache",
    )

    @classmethod
    def execute(
            cls,
            filename: str,
            execute: Callable[[], Any],
            load: Callable[[str], Any],
            save: Callable[[str, Any], None],
            read_cache: bool = True,
            write_cache: bool = True,
    ) -> Any:
        if read_cache:
            if cls.exists(filename):
                cls._log("loading cache", cls.filename(filename))
                try:
                    return load(cls.filename(filename))
                except Exception as e:
                    cls._log("cache load error:", type(e).__name__, e)
                    pass
            else:
                cls._log("no cache file:", cls.filename(filename))

        result = execute()

        if write_cache:
            cls._log("writing cache", cls.filename(filename))
            FileCache.make_path(filename)
            save(cls.filename(filename), result)

        return result

    @classmethod
    def execute_json(
            cls,
            filename: str,
            execute: Callable[[], Any],
            read_cache: bool = True,
            write_cache: bool = True,
    ) -> Any:
        def _load(name):
            with open(name) as fp:
                return json.load(fp)

        def _save(name, data):
            with open(name, "w") as fp:
                json.dump(data, fp)

        return cls.execute(
            filename=filename,
            execute=execute,
            load=_load,
            save=_save,
            read_cache=read_cache,
            write_cache=write_cache,
        )

    @classmethod
    def to_hash(cls, anything) -> str:
        try:
            import numpy
            if isinstance(anything, numpy.ndarray):
                return cls.md5(anything.tobytes())
        except ImportError:
            pass
        try:
            import pandas
            if isinstance(anything, pandas.DataFrame):
                return cls.md5(anything.values.tobytes())
        except ImportError:
            pass
        return cls.repr_hash(anything)

    @classmethod
    def repr_hash(cls, anything) -> str:
        bytes_ = repr(anything).encode("utf-8", errors="ignore")
        return cls.md5(bytes_)

    @classmethod
    def md5(cls, bytes_) -> str:
        return hashlib.md5(bytes_).hexdigest()

    @classmethod
    def filename(cls, filename: str) -> str:
        return os.path.join(cls.PATH, filename)

    @classmethod
    def filepath(cls, filename: str) -> str:
        full_path = cls.PATH
        filename_path = os.path.dirname(filename)
        if filename_path:
            full_path = os.path.join(cls.PATH, filename_path)
        return full_path

    @classmethod
    def exists(cls, filename: str) -> bool:
        return os.path.exists(cls.filename(filename))

    @classmethod
    def make_path(cls, filename: str):
        path = cls.filepath(filename)
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def _log(cls, *args, **kwargs):
        if cls.verbose:
            print(*args, **kwargs, file=sys.stderr)