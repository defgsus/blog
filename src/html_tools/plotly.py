from typing import Optional

from ._display import html_display


def html_plotly(
        fig,
        post_script: Optional[str] = None,
        post_html: Optional[str] = None,
        display: bool = True,
) -> Optional[str]:
    html = fig.to_html(
        full_html=False,
        include_mathjax=False,
        include_plotlyjs="require",
        post_script=post_script,
    )
    if post_html:
        html += post_html

    if display:
        html_display(html)
    else:
        return html
