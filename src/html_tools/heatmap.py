import pathlib
import json
import secrets
from typing import Union, Optional, List

import pandas as pd


_JS_SOURCE = pathlib.Path(__file__).resolve().parent.joinpath("heatmap.js").read_text()


def html_heatmap(
        matrix: Union[List[List], pd.DataFrame],
        labels_x: Optional[List[str]] = None,
        labels_y: Optional[List[str]] = None,
        title: str = None,
        transpose: bool = False,
        filterable: bool = True,
        filter_x: str = None,
        filter_y: str = None,
        display: bool = True,
        colors: Union[str, List[str]] = "GnBu",
        label_width: str = "10rem",
        min_cells_x: int = 20,
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
        color_name = colors
        try:
            from .colors import get_color_map
        except ImportError:
            from colors import get_color_map
        colors = get_color_map(color_name)

        if not colors:
            raise ValueError(f"No color map found with name '{color_name}'")

    ID = secrets.token_hex(10)

    html = """<style>
        .heatmap-%(ID)s {
            display: grid;
        }
        .heatmap-filters-{ID} label {
            color: #a0a0a0;
            width: 100%%;
        }
        .heatmap-filters-{ID} input {
            width: 70%%;
        }
        .heatmap-%(ID)s .hmlabel,
        .heatmap-%(ID)s .hmlabelv,
        .heatmap-%(ID)s .hmlabelvb {
            text-align: right;
            padding-right: .3rem;
            /*overflow: hidden;*/
            white-space: nowrap;
        }
        .heatmap-%(ID)s .hmlabelv,
        .heatmap-%(ID)s .hmlabelvb {
            writing-mode: sideways-lr;
            /*height: %(label_width)s;*/
            padding: 0;
            padding-bottom: .3rem;
            text-align: left;
            width: 0;
            line-height: 0.6rem;
        }
        .heatmap-%(ID)s .hmlabelvb {
            text-align: right;
            padding: 0;
            padding-top: .3rem;
        }
        .heatmap-%(ID)s .hmc {
            padding-bottom: 100%%;
            border-top: 1px solid rgba(255, 255, 255, .2);
            border-left: 1px solid rgba(255, 255, 255, .1);
            border-right: 1px solid rgba(0, 0, 0, .1);
            border-bottom: 1px solid rgba(0, 0, 0, .3);
        }
        .heatmap-%(ID)s .hmc.hmc-empty {
            border: none;
            border-left: 1px solid rgba(0, 0, 0, .2);
            border-top: 1px solid rgba(0, 0, 0, .2);
            border-right: 1px solid rgba(255, 255, 255, .2);
            border-bottom: 1px solid rgba(255, 255, 255, .2);
        }
        .heatmap-%(ID)s .hmc.hmc-overlap {
            border: none;
            padding: 0;
        }
    """ % {
        "ID": ID,
        "width": matrix.shape[1],
        "label_width": label_width,
    }
    for i, c in enumerate(colors):
        html += f""".heatmap-{ID} .hmc.hmc-{i} {{ background: {c} }}"""
    html += """</style>"""

    if title:
        html += f"<h3>{title}</h3>"

    if filterable:
        html += f"""
            <div class="heatmap-filters-{ID}">
                <div><label>filter x <input class="filter-x" type="text" value="{filter_x or ""}"></label></div>
                <div><label>filter y <input class="filter-y" type="text" value="{filter_y or ""}"></label></div>
            </div>
        """

    html += f"""<div class="heatmap-{ID}"></div>"""

    script_context = {
        "id": ID,
        "num_colors": len(colors),
        "label_width": label_width,
        "filters": json.dumps({
            "x": filter_x,
            "y": filter_y,
        }),
        "min_cells_x": min_cells_x,
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

    if 0:
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

        html = html_heatmap(
            rows,
            labels_x=labels_x,
            labels_y=labels_y,
            display=False,
            #colors=["red", "green", "blue"],
        )
    else:
        df = pd.read_pickle("test-data.pkl")
        html = html_heatmap(
            df,
            transpose=True,
            display=False,
            filter_x="m",
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <h3>heatmap test</h3>
        {html}
    </body>
    </html>
    """
    with open("heatmap-test.html", "w") as fp:
        fp.write(html)
