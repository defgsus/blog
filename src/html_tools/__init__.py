from .heatmap import html_heatmap
from .datatables import datatable


def html_display(content: str):
    from IPython.display import display, HTML
    display(HTML(content))
