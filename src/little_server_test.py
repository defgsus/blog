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

    server.set_cell_layout("1", 1, [1, 13])
    server.set_cell_layout("random", 3, [1, 3])
    server.set_cell_layout("image", 4, [1, 7])
    server.set_cell_layout("image2", 5, [4, 13])

    server.set_cell("1", text="This is a <b>test</b>,\nfor multiple lines")
    for i in range(1, 13):
        server.set_cell(f"i{i}", 2, i, text=f"col {i}")
    server.set_cell("span", 5, [1,4], text="1-4")

    server.set_cell("code", 3, [3, 7], code="def func(x, y):\n    print('<h2>Hello</h2>')")
    server.set_cell("image", image=random_plot())
    server.set_cell("image2", image=random_plot(1000))

    try:
        server.start()

        while True:
            time.sleep(1)
            server.set_cell("random", text=random.randint(0, 100))
            if random.randrange(1) == 0:
                server.set_cell("image", image=random_plot())

    finally:
        server.stop()

