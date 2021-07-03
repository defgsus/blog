import random

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np

from little_server import LittleServer


def random_plot(count: int = 128):
    data = np.random.randn(count)

    fig = plt.figure(figsize=(6, 2))
    plt.title(f"a plot")
    plt.plot(data, label="uniform gauss")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    return fig


if __name__ == "__main__":
    import time
    server = LittleServer()

    server.set_cell_layout("1", 1, [1, 9])
    server.set_cell_layout("actions", 1, [9, 13])
    server.set_cell_layout("random", 3, [1, 3])
    server.set_cell_layout("image", [4, 7], [1, 9])
    server.set_cell_layout("image2", [7, 11], [4, 13])
    server.set_cell_layout("image3", [11, 14], [1, 13])
    server.set_cell_layout("log", [3, 7], [9, 13])

    server.set_cell("1", text="This is a <b>test</b>,\nfor multiple lines")
    for i in range(1, 13):
        server.set_cell(f"i{i}", 2, i, text=f"col {i}")
    server.set_cell("span", 5, [1,4], text="1-4")

    server.set_cell("code", 3, [3, 7], code="def func(x, y):\n    print('<h2>Hello</h2>')")
    server.set_cell("image", image=random_plot(), fit=True)
    server.set_cell("image2", images=[random_plot(1000), random_plot(10000)])
    server.set_cell("image3", images=[random_plot(32)])
    server.set_cell("log", log="Blabla\nblublub\ttabbed! <b>bold</b>")
    server.set_cell("actions", text="some actions", actions=["start", "stop"])

    try:
        server.start()

        while True:
            time.sleep(1)
            server.set_cell("random", text=random.randint(0, 100))
            if random.randrange(4) == 0:
                server.set_cell("image", image=random_plot(), fit=True)
            server.log("log", f"a new line {random.randint(0, 10)}")

    finally:
        server.stop()

