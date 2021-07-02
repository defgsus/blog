import json
import hashlib
from pathlib import Path
from copy import deepcopy
from multiprocessing import Process
from threading import Thread
from queue import Queue
from io import BytesIO
from typing import Union

import PIL.Image
import matplotlib.pyplot as plt

from flask import Flask, request, render_template


class LittleServer:

    def __init__(self):
        self.host = "localhost"
        self.port = 9009

        self._thread: Thread = None
        self._app: Flask = None

        self._cells = dict()
        self._cells_update = set()
        self._images = dict()
        self._actions = dict()

    def start(self):
        if not self._thread:
            self._thread = Thread(target=self._mainloop)
            self._thread.start()
        return

    def stop(self):
        pass
        #if self._thread:
        #    self._thread.terminate()
        #    self._thread.join()

    def _mainloop(self):
        self._app = Flask(
            "LittleServer",
            static_folder=str(Path(__file__).resolve().parent / "static"),
            template_folder=str(Path(__file__).resolve().parent / "templates"),
        )
        self._app.add_url_rule("/", "index", self._index_view)
        self._app.add_url_rule("/cells/", "cells", self._cells_view)
        self._app.add_url_rule("/action/", "action", self._action_view, methods=["POST"])
        self._app.add_url_rule("/img/<name>.png", "image", self._image_view)

        # print(f"http://{self.host}:{self.port}")
        self._app.run(host=self.host, port=self.port)

    def _index_view(self):
        return render_template("cells.html")

    def _cells_view(self):
        cells = [
            self._cells[name]
            for name in sorted(self._cells, key=lambda n: self._cells[n]["index"])
            #if name in self._cells_update
        ]
        #self._cells_update.clear()
        return json.dumps({"cells": cells})

    def set_cell(self, name: str, **params):
        if name in self._cells:
            cell = {"index": self._cells[name]["index"]}
        else:
            cell = {"index": len(self._cells)}

        cell.update(params)
        cell["name"] = name

        if cell.get("image"):
            image = cell["image"]
            if isinstance(image, (PIL.Image.Image, plt.Figure)):
                self._set_image(name, image)
            cell["image"] = f"/img/{cell['name']}.png"
            cell["width"] = self._images[cell["name"]]["width"]
            cell["height"] = self._images[cell["name"]]["height"]

        if cell.get("actions"):
            self._actions.update(cell["actions"])
            cell["actions"] = [
                {"id": key, "name": key}
                for key in cell["actions"]
            ]

        hash_source = (
            json.dumps(cell).encode("ascii")
            + (self._images.get(cell["name"], {}).get("data") or b"")
        )
        cell["hash"] = hashlib.md5(hash_source).hexdigest()

        if self._cells.get(name) != cell:
            self._cells[name] = cell
            #self._cells_update.add(name)

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
            raise TypeError(f"Unhandled image type {type(image).__name__}")

        fp.seek(0)
        self._images[name] = {
            "width": width,
            "height": height,
            "data": fp.read(),
        }

    def _image_view(self, name):
        return self._images.get(name, {}).get("data")

    def _action_view(self):
        action = request.args.get("a")
        if action in self._actions:
            self._actions[action]()
        return ""


if __name__ == "__main__":
    import time
    server = LittleServer()

    img = PIL.Image.open("../snapshot.png")

    server.set_cell("1", text="One!")
    server.set_cell("2", text="Two!")
    server.set_cell("3", text="Three!")
    server.set_cell("4", image=img)

    try:
        server.start()

        while True:
            time.sleep(1)
            server.set_cell("1", text="X")

    finally:
        server.stop()

