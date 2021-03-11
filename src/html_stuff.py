import secrets
import plotly.express as px


def html_heatmap(matrix, labels_x, labels_y, display: bool = True):
    if display:
        from IPython import display, HTML

    min_val = min(min(row) for row in matrix)
    max_val = max(max(row) for row in matrix)

    colors = px.colors.PLOTLY_SCALES["Viridis"]

    ID = secrets.token_hex(8)
    html = f"""<style>
        .heatmap-{ID} {{
            display: grid;
            grid-template-columns: repeat({len(matrix[0])}, 1fr);
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
    for row, label_y in zip(matrix, labels_y):
        for v, label_x in zip(row, labels_x):
            vc = int(max(0, (v - min_val) / (max_val - min_val) * len(colors) - 1e-5))
            html += f"""<div class="hmc v{vc}" title="{label_x}/{label_y}: {v}"></div>"""
    html += """</div>"""
    if display:
        display(HTML(html))
    else:
        return html

