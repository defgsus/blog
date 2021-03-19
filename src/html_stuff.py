import secrets
from typing import Union, List, Optional

import plotly.express as px
import pandas as pd


def display_html(content):
    from IPython.display import display, HTML
    display(HTML(content))


def html_heatmap(
        matrix: Union[List[List], pd.DataFrame],
        labels_x: Optional[List[str]] = None,
        labels_y: Optional[List[str]] = None,
        transpose: bool = False,
        display: bool = True,
        colors: Union[str, List[str]] = "Viridis",
):
    if display:
        from IPython.display import display, HTML

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)

    min_val = matrix.min().min()
    max_val = matrix.max().max()
    if not labels_x:
        labels_x = matrix.columns
    if not labels_y:
        labels_y = matrix.index

    if transpose:
        matrix = matrix.transpose()
        labels_x, labels_y = labels_y, labels_x

    if isinstance(colors, str):
        colors = px.colors.PLOTLY_SCALES[colors]

    ID = secrets.token_hex(8)
    html = f"""<style>
        .heatmap-{ID} {{
            display: grid;
            grid-template-columns: repeat({matrix.shape[1]}, 1fr);
        }}
        .heatmap-{ID} .hmc {{
            padding-bottom: 100%;
            /*grid-column: 1;
            grid-row: 1;
            width: 1rem;
            height: 1rem;*/
        }}
    """
    for i, c in enumerate(colors):
        html += f""".heatmap-{ID} .hmc.v{i} {{ background: {c[1]} }}"""
    html += """</style>"""

    html += f"""<div class="heatmap-{ID}">"""
    for row, label_y in zip(matrix.iloc, labels_y):
        for v, label_x in zip(row, labels_x):
            if v > 0:
                vc = int(max(0, (v - min_val) / (max_val - min_val) * len(colors) - 1e-5))
                vc = f" v{vc}"
            else:
                vc = ""
            html += f"""<div class="hmc{vc}" title="{label_x} / {label_y}: {v}"></div>"""
    html += """</div>"""
    if display:
        display(HTML(html))
    else:
        return html
