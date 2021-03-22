import pathlib
import json
import secrets
import html as html_lib
from typing import Union, Optional, List

import pandas as pd


_JS_SOURCE = pathlib.Path(__file__).resolve().parent.joinpath("heatmap.js").read_text()

assert "//" not in _JS_SOURCE, f"Replace // ... with /* ... */ to make it one-line-able."


def html_heatmap(
        matrix: Union[List[List], pd.DataFrame],
        labels_x: Optional[List[str]] = None,
        labels_y: Optional[List[str]] = None,
        title: str = None,
        transpose: bool = False,
        filterable: bool = True,
        filter_x: str = None,
        filter_y: str = None,
        show_empty_x: bool = True,
        show_empty_y: bool = True,
        display: bool = True,
        colors: Union[str, List[str]] = "GnBu",
        label_width: str = "10rem",
        max_label_length: int = 30,
        keep_label_front: bool = False,
        min_cells_x: int = 20,
        max_cells_x: int = 100,
        max_cells_y: int = 100,
):
    if display:
        from IPython.display import display, HTML

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    if not labels_x:
        labels_x = matrix.columns
    if not labels_y:
        labels_y = matrix.index

    has_empty_cells = matrix.isnull().values.any()

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
        .heatmap-filters {
            color: #a0a0a0;
        }
        .heatmap-filters-axis {
            display: grid;
        }
        .heatmap-filters-axis label {
            display: inline;
        }
        .heatmap-filters-axis div {
            grid-row: row;
        }
        .heatmap-filters-axis .heatmap-filter-cell-string {
            grid-column: 1 / 6;
        }
        .heatmap-filters-axis .heatmap-filter-cell-page {
            width: 14rem;
        }
        .heatmap-filters-axis input[type="text"] {
            width: 85%%;
        }
        .heatmap-filters-axis input[type="number"] {
            width: 4rem;
        }
        
        .heatmap-grid {
            display: grid;
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
        .heatmap-%(ID)s .hmc-overlap {
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

    html += f"""<div class="heatmap-{ID}">"""

    if filterable:
        def render_axis_params(dim: str, filter_value: str, show_empty: bool, page: int) -> str:
            return f"""
                <div class="heatmap-filters-axis heatmap-filters-axis-{dim}">
                    <div class="heatmap-filter-cell-string"><label>
                        {dim} filter <input type="text" value="{filter_value or ""}"></label>
                    </div>
                    <div class="heatmap-filter-cell-empty" {"hidden" if not has_empty_cells else ""}><label>
                        show empty <input type="checkbox" {"checked" if show_empty else ""}></label>
                    </div>
                    <div class="heatmap-filter-cell-page"><label>
                        page <input type="number" value="{page+1}" min="1"> of <span class="heatmap-page-count">1</span></label>
                    </div>
                </div>
            """
        html += f"""
            <div class="heatmap-filters">
                {render_axis_params("x", filter_x, show_empty_x, 0)}
                {render_axis_params("y", filter_y, show_empty_y, 0)}
                <label class="heatmap-dimensions"></label>    
            </div>
        """

    html += f"""<div class="heatmap-grid"></div></div>"""

    script_context = {
        "id": ID,
        "num_colors": len(colors),
        "label_width": label_width,
        "filters": json.dumps({
            "x": filter_x,
            "y": filter_y,
            "empty_x": show_empty_x,
            "empty_y": show_empty_y,
            "page_x": 0,
            "page_y": 0,
            "offset_x": 0,
            "offset_y": 0,
        }),
        "max_label_length": max_label_length,
        "keep_label_front": json.dumps(keep_label_front),
        "min_cells_x": min_cells_x,
        "max_cells_x": max_cells_x,
        "max_cells_y": max_cells_y,
        "data": json.dumps({
            "matrix": matrix.values.tolist(),
            "labels_x": [html_lib.escape(str(l)) for l in labels_x],
            "labels_y": [html_lib.escape(str(l)) for l in labels_y],
        }),
    }
    script = _JS_SOURCE
    for key, value in script_context.items():
        script = script.replace(f"__{key}__", str(value))

    html += f"""<script type="text/javascript">{script}</script>"""

    html = " ".join(line.strip() for line in html.splitlines())

    if display:
        display(HTML(html))
    else:
        return html


if __name__ == "__main__":
    import random
    import numpy as np

    if 1:
        def random_name():
            return "".join(
                random.choice(["ku", "ka", "bo", "ba", "su", "la", "to", "mi", "no", "ha", '\"', ">", "<"])
                for _ in range(random.randrange(3, 15))
            )

        rows = []
        value = 0
        for y in range(1000):
            row = []
            for x in range(200):
                row.append(value if random.randint(0, 1) else np.nan)
                value += 1
            rows.append(row)

        labels_x = [f"{i}-{random_name()}" for i in range(len(rows[0]))]
        labels_y = [f"{i}-{random_name()}" for i in range(len(rows))]

        html = html_heatmap(
            rows,
            labels_x=labels_x,
            labels_y=labels_y,
            display=False,
            label_width="15rem",
            max_label_length=20,
            keep_label_front=True,
            min_cells_x=40,
            #max_cells_y=100,
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
    <head>
        <meta charset="UTF-8">
    </head>
    <body>
        <h3>heatmap test</h3>
        {html}
    </body>
    </html>
    """
    with open("heatmap-test.html", "w") as fp:
        fp.write(html)
