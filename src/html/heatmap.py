import pathlib
import json
import secrets
from typing import Union, Optional, List

import pandas as pd

import plotly.express as px


_JS_SOURCE = pathlib.Path(__file__).resolve().parent.joinpath("heatmap.js").read_text()


def html_heatmap(
        matrix: Union[List[List], pd.DataFrame],
        labels_x: Optional[List[str]] = None,
        labels_y: Optional[List[str]] = None,
        title: str = None,
        transpose: bool = False,
        display: bool = True,
        colors: Union[str, List[str]] = "Viridis",
        label_width: str = "10rem",
):
    if display:
        from IPython.display import display, HTML

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    if not labels_x:
        labels_x = matrix.columns
    if not labels_y:
        labels_y = matrix.index

    if transpose:
        matrix = matrix.transpose()
        labels_x, labels_y = labels_y, labels_x

    if isinstance(colors, str):
        colors = [c[1] for c in px.colors.PLOTLY_SCALES[colors]]

    ID = secrets.token_hex(8)

    html = """<style>
        .heatmap-%(ID)s {
            display: grid;
        }
        .heatmap-%(ID)s .hmlabel,
        .heatmap-%(ID)s .hmlabelv,
        .heatmap-%(ID)s .hmlabelvb {
            text-align: right;
            padding-right: .3rem;
            overflow: hidden;
        }
        .heatmap-%(ID)s .hmlabelv,
        .heatmap-%(ID)s .hmlabelvb {
            writing-mode: vertical-lr;
            height: %(label_width)s;
            padding: 0;
            padding-bottom: .3rem;
        }
        .heatmap-%(ID)s .hmlabelvb {
            text-align: left;
            padding: 0;
            padding-top: .3rem;
        }
        .heatmap-%(ID)s .hmc {
            padding-bottom: 100%%;
            border: 1px solid black;
        }
    """ % {
        "ID": ID,
        "width": matrix.shape[1],
        "label_width": label_width,
    }
    for i, c in enumerate(colors):
        html += f""".heatmap-{ID} .hmc.v{i} {{ background: {c} }}"""
    html += """</style>"""

    if title:
        html += f"<h3>{title}</h3>"

    html += f"""<div class="heatmap-{ID}"></div>"""

    script_context = {
        "id": ID,
        "num_colors": len(colors),
        "label_width": label_width,
        "data": json.dumps({
            "matrix": matrix.values.tolist(),
            "labels_x": [str(l) for l in labels_x],
            "labels_y": [str(l) for l in labels_y],
        }),
    }
    script = _JS_SOURCE
    for key, value in script_context.items():
        script = script.replace(f"__{key}__", str(value))

    html += f"""<script type="text/javascript">{script}</script>"""

    html = "\n".join(line.strip() for line in html.splitlines())

    if display:
        display(HTML(html))
    else:
        return html


if __name__ == "__main__":
    import random

    def random_name():
        return "".join(
            random.choice(["ku", "ka", "bo", "ba", "su", "la", "to", "mi", "no", "ha"])
            for _ in range(random.randrange(3, 9))
        )

    rows = []
    value = 0
    for y in range(20):
        row = []
        for x in range(60):
            row.append(value)
            value += 1
        rows.append(row)

    labels_x = [random_name() for _ in rows[0]]
    labels_y = [random_name() for _ in rows]

    heatmap_html = html_heatmap(
        rows,
        labels_x=labels_x,
        labels_y=labels_y,
        display=False,
        #colors=["red", "green", "blue"],
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <h3>heatmap test</h3>
        {heatmap_html}
    </body>
    </html>
    """
    with open("heatmap-test.html", "w") as fp:
        fp.write(html)
