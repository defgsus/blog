import json
import secrets
from typing import Dict

import pandas as pd


def datatable(
        df: pd.DataFrame,
        max_rows: int = 1000,
        max_cols: int = None,
        paging: bool = True,
        compact: bool = True,
        table_id: str = None,
        description: Dict[str, str] = None,
        **kwargs,
):
    from IPython.display import display, HTML

    table_id = table_id or f"table-{secrets.token_hex(10)}"
    html = df.to_html(
        table_id=table_id,
        escape=True,
        max_rows=max_rows,
        max_cols=max_cols,
        classes="compact" if compact else None,
    )

    kwargs.update({
        "paging": paging,
    })
    options_str = json.dumps(kwargs)

    html += f"""<script type="text/javascript">
        jQuery("#{table_id}").DataTable({options_str});
    </script>"""

    if description:
        html += """<div class="table-description"><ul>"""
        for key, text in description.items():
            html += f"""<li><b>{key}</b>: {text}</li>"""
        html += """</ul></div>"""

    display(HTML(html))
