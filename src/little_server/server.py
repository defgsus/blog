import hashlib
import json
from pathlib import Path
from copy import deepcopy
from multiprocessing import Process
from threading import Thread
from functools import partial
import urllib.parse
import html
from io import BytesIO
from typing import Optional, Dict, Sequence, Union

import tornado.ioloop
import tornado.web
import tornado.websocket

import PIL.Image
import matplotlib.pyplot as plt


class LittleServer:

    BASE_PATH = Path(__file__).resolve().parent

    def __init__(
            self,
            host: str = "localhost",
            port: int = 9009,
            debug: bool = True,
            title: str = "littleServer",
    ):
        self.host = host
        self.port = port
        self.title = title
        self._debug = debug

        self._thread: Optional[Thread] = None
        self._app: Optional[tornado.web.Application] = None
        self._io_loop: Optional[tornado.ioloop.IOLoop] = None
        self._ws_clients: Dict[str, tornado.websocket.WebSocketHandler] = dict()

        self._cells = dict()
        self._cells_layout = dict()
        self._cells_update = set()
        self._images = dict()
        self._actions = dict()

    def url(self, protocol: str = "http"):
        return f"{protocol}://{self.host}:{self.port}"

    def start(self):
        if not self._thread:
            self._thread = Thread(target=self._mainloop, name="littleServer")
            self._thread.start()
        return

    def stop(self):
        if self._io_loop:
            self._io_loop.stop()
            self._thread.join()
            self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._io_loop)

    def send_message(self, name: str, data: Optional[dict] = None):
        assert self._io_loop, f"Called send_message on stopped server"
        message = {"name": deepcopy(name), "data": deepcopy(data)}
        self._io_loop.add_callback(partial(self._send_message, message))

    def set_cell_layout(
            self,
            name: str,
            row: Optional[Union[int, Sequence[int]]] = None,
            column: Optional[Union[int, Sequence[int]]] = None,
    ):
        #if row is None or column is None:
        #    assert row is None and column is None, "Must either supply 'row' AND 'column' or none of it"

        if name not in self._cells_layout:
            self._cells_layout[name] = dict()
        cell_layout = self._cells_layout[name]

        layout_changed = False
        if cell_layout.get("row") != row or cell_layout.get("column") != column:
            layout_changed = True

        cell_layout["row"] = row
        cell_layout["column"] = column

        if layout_changed:
            if self._cells.get("name"):
                self._cells[name]["row"] = row
                self._cells[name]["column"] = column

                if self.running:
                    self.send_message("cell", self._cells["name"])
                else:
                    self._cells_update.add(name)

    def set_cell(
            self,
            name: str,
            row: Optional[Union[int, Sequence[int]]] = None,
            column: Optional[Union[int, Sequence[int]]] = None,
            text: Optional[str] = None,
            code: Optional[str] = None,
            image: Optional[Union[PIL.Image.Image]] = None,
    ):
        if row is not None or column is not None:
            if name not in self._cells_layout:
                self._cells_layout[name] = dict()
            if row is not None:
                self._cells_layout[name]["row"] = row
            if column is not None:
                self._cells_layout[name]["column"] = column

        if row is None:
            row = self._cells_layout.get(name, {}).get("row")
        if column is None:
            column = self._cells_layout.get(name, {}).get("column")

        if row is not None:
            row = str(row) if isinstance(row, int) else " / ".join(str(i) for i in row)
        if column is not None:
            column = str(column) if isinstance(column, int) else " / ".join(str(i) for i in column)

        cell = {
            "name": name,
            "row": row,
            "column": column,
        }
        if text:
            cell["text"] = str(text)
        if code:
            cell["code"] = html.escape(str(code))

        if image:
            if isinstance(image, (PIL.Image.Image, plt.Figure)):
                self._set_image(name, image)
                cell["image"] = f"/img/{cell['name']}.png?h={self._images[name]['hash']}"
                cell["width"] = self._images[cell["name"]]["width"]
                cell["height"] = self._images[cell["name"]]["height"]

        hash_source = json.dumps(cell).encode("ascii")
        cell["hash"] = hashlib.md5(hash_source).hexdigest()

        if self._cells.get(name) != cell:
            self._cells[name] = cell
            if self.running:
                self.send_message("cell", cell)
            else:
                self._cells_update.add(name)

    def _set_image(self, name: str, image: Union[PIL.Image.Image, plt.Figure]):
        fp = BytesIO()
        if isinstance(image, PIL.Image.Image):
            image.save(fp, "png")
            width, height = image.width, image.height
        elif isinstance(image, plt.Figure):
            image.savefig(fp, format="png")
            width, height = (
                image.get_figwidth() * image.dpi,
                image.get_figheight() * image.dpi,
            )
        else:
            raise TypeError(f"Unhandled image type {type(image).__name__} in cell '{name}'")

        fp.seek(0)
        data = fp.read()
        self._images[name] = {
            "width": width,
            "height": height,
            "data": data,
            "hash": hashlib.md5(data).hexdigest(),
        }

    def _url_handlers(self) -> list:
        from .handlers import (
            IndexHandler, WebSocketHandler, ImageHandler
        )
        return [
            (r"/", IndexHandler, {"server": self}),
            (r"/ws", WebSocketHandler, {"server": self}),
            (r"/img/([a-z]+).png", ImageHandler, {"server": self}),
        ]

    def _mainloop(self):
        self._io_loop = tornado.ioloop.IOLoop()
        self._io_loop.make_current()

        self._app = tornado.web.Application(
            handlers=self._url_handlers(),
            default_host=self.host,
            static_path=str(self.BASE_PATH / "static"),
            template_path=str(self.BASE_PATH / "templates"),
            debug=self._debug,
        )
        self._app.listen(self.port)

        while self._cells_update:
            self.send_message("cell", self._cells_update.pop())

        self._io_loop.start()
        self._io_loop.close()

        self._io_loop = None
        self._app = None

    def _send_message(self, message: dict):
        for client_id, client in self._ws_clients.items():
            client.write_message(message)

    def _on_ws_message(self, client: tornado.websocket.WebSocketHandler, message: dict):
        name, data = message["name"], message.get("data")

        if name == "dom-loaded":
            for cell in self._cells.values():
                client.write_message({"name": "cell", "data": cell})


