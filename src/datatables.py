import json
import secrets

import pandas as pd


def datatable(
        df: pd.DataFrame,
        max_rows: int = 1000,
        max_cols: int = None,
        paging: bool = True,
        table_id: str = None,
        **kwargs,
):
    from IPython.display import display, HTML

    table_id = table_id or f"table-{secrets.token_hex(10)}"
    html = df.to_html(
        table_id=table_id,
        escape=True,
        max_rows=max_rows,
        max_cols=max_cols,
    )

    kwargs.update({
        "paging": paging,
    })
    options_str = json.dumps(kwargs)

    html += f"""<script type="text/javascript">
        jQuery("#{table_id}").DataTable({options_str});
    </script>"""

    display(HTML(html))
