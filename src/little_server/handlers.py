import json
import uuid

import tornado.web
import tornado.websocket


class BaseHandler(tornado.web.RequestHandler):

    def initialize(self, server):
        from .server import LittleServer
        self.server: LittleServer = server


class IndexHandler(BaseHandler):

    def get(self):
        context = dict(
            title=self.server.title,
            websocket_url=f"{self.server.url('ws')}/ws",
        )
        self.render("cells.html", **context)


class ImageHandler(BaseHandler):

    def get(self, name, index):
        index = int(index)
        images = self.server._images.get(name)
        if images:
            self.set_header("Content-Type", "image/png")
            self.write(images[index]["data"])


class WebSocketHandler(tornado.websocket.WebSocketHandler):

    def initialize(self, server):
        self.server = server

    def open(self, *args: str, **kwargs: str):
        self.client_id = str(uuid.uuid4())
        self.server._ws_clients[self.client_id] = self

    def on_message(self, message):
        message = json.loads(message)
        self.server._on_ws_message(self, message)

    def on_close(self):
        self.server._ws_clients.pop(self.client_id, None)
