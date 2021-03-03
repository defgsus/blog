import sys
from copy import copy, deepcopy
from typing import List

# sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import plotly
from ipywidgets import HTML

from elastipy import Search, query, plot, connections

plot.set_backend("plotly")
plotly.io.templates.default = "plotly_dark"
es = connections.get()


def heatmap(s, agg_x, agg_y, metric=None, percent=False, digits=2, height=None, **kwargs):
    s = s.copy()
    agg = agg_x(s)
    agg_x_name = agg.name
    agg = agg_y(agg)
    agg_y_name = agg.name
    if metric:
        agg_m = metric(agg)
        if agg_m == agg and agg.children:
            agg_m = agg.children[0]
        agg = agg_m
        metric_name = agg.name
    else:
        metric_name = "count"

    matrix = agg.execute().df_matrix(sort=agg_y_name)

    if percent:
        df = agg.df()
        if not metric:
            sum_series = df.groupby(agg_x_name).first()[f"{agg_x_name}.doc_count"]
        else:
            sum_series = df.groupby(agg_x_name).sum()[metric_name]

        matrix = matrix.div(sum_series, axis=0) * 100.
        metric_name = f"{metric_name} %"

    matrix = matrix.round(digits).replace({0: np.nan})
    matrix.sort_index(inplace=True)

    if height is None:
        height = matrix.shape[1]*16 + 150

    kwargs.setdefault("labels", {"x": agg_x_name, "y": agg_y_name, "color": metric_name})
    kwargs.setdefault("color_continuous_scale", plotly.colors.sequential.deep_r)

    return plot.heatmap(
        matrix.transpose(),
        height=height,
        **kwargs,
    ).update_layout(margin={"l": 10, "t": 20, "b": 10, "r": 10})


def multi_line(s, agg_x, agg_y, metric=None, percent=False, digits=2, **kwargs):
    df_metric_all = None
    if metric and percent:
        agg = metric(agg_x(s.copy()))
        if agg.children:
            agg = agg.children[0]
        df_metric_all = agg.execute().df(to_index=True)

    s = s.copy()
    agg = agg_x(s)
    agg_x_name = agg.name
    agg = agg_y(agg)
    agg_y_name = agg.name
    if metric:
        agg_m = metric(agg)
        if agg_m == agg and agg.children:
            agg_m = agg.children[0]
        agg = agg_m
        metric_name = agg.name
    else:
        metric_name = "count"

    df = agg.execute().df(to_index=agg_x_name, flat=agg_y_name)
    if metric:
        for key in list(df.keys()):
            if not key.endswith(".doc_count") and not key.endswith(f".{metric_name}"):
                df.pop(key)
            elif key.endswith(f".{metric_name}"):
                df[key[:-len(metric_name)-1]] = df.pop(key)

    if percent:
        if not metric:
            sum_series = df[f"{agg_x_name}.doc_count"]
        else:
            sum_series = df_metric_all[metric_name]
        df = df.div(sum_series, axis=0) * 100.

    df.pop(f"{agg_x_name}.doc_count")
    if digits:
        df = df.round(digits)

    if metric:
        y_name = metric_name
    else:
        y_name = agg_y_name
    if percent:
        y_name = f"{y_name} %"

    kwargs.setdefault("labels", {"value": y_name})
    return df.plot.line(
        **kwargs,
    )


class HNItem:
    def __init__(self, id_or_data, show_kids=False):
        self._kids = None
        self._show_kids = show_kids
        if isinstance(id_or_data, dict):
            self.id = id_or_data["id"]
            self._data = id_or_data
        elif isinstance(id_or_data, HNItem):
            self.id = id_or_data.id
            self._data = id_or_data._data
            self._show_kids = id_or_data._show_kids
        else:
            self.id = id_or_data
            self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = es.get("hackernews-items", self.id)["_source"]
        return self._data

    @property
    def kids(self) -> List['HNItem']:
        if self._kids is None:
            self._kids = [self.__class__(i) for i in (self.data.get("kids") or [])]
        return self._kids

    def show_kids(self, show_kids: bool = True) -> 'HNItem':
        self._show_kids = show_kids
        return self

    def to_html(self, color="white", show_kids = None) -> str:
        from ipywidgets.widgets import Accordion
        data = self.data
        html = f"""
        <div style="color: {color}">
        <p>{data["timestamp"]} {data["id"]} {data["type"]} {data.get("by")}
        """
        if data.get("title"):
            html += f"""<span style="font: caption">{data["title"]}</span>"""
            if data.get("title_entropy"):
                html += f""" <span style="color: #ccf">(ent: {data["title_entropy"]:.2f}/{data["title_entropy_256"]:.2f})</span>"""

        html += "</p>"

        info = []
        if data.get("score") is not None:
            info.append(f"""score {data.get("score") or "-"}""")
        if data.get("descendants") is not None:
            info.append(f"""desc. {data.get("descendants") or "-"}""")
        if data.get("kids"):
            info.append(f"""kids {len(data["kids"])}""")

        if info:
            html += f"""<p>{", ".join(info)}</p>"""

        if data.get("text"):
            html += f"""
            <div style="font: caption">{data["text"]}</div>
            """.replace("<script", "<scriii")
            if data.get("text_entropy"):
                html += f""" <span style="color: #ccf">(ent: {data["text_entropy"]:.2f}/{data["text_entropy_256"]:.2f})</span>"""

        if show_kids is None:
            show_kids = self._show_kids
        if show_kids:
            kids = copy(self.kids)
            kids.sort(key=lambda k: k.data.get("timestamp") or "")
            for kid in kids:
                html += """<div style="margin-left: 2rem">"""
                html += kid.to_html()
                html += "</div>"
            html += "</div>"  # color
        return html

    def _ipython_display_(self, **kwargs):
        from ipywidgets import HTML
        return HTML(self.to_html())._ipython_display_(**kwargs)


class HNItems(list):
    def __init__(self, items_or_ids, show_kids: bool = False):
        super().__init__(HNItem(i) for i in items_or_ids)
        self._show_kids = show_kids

    def show_kids(self, show_kids: bool = True) -> 'HNItems':
        self._show_kids = show_kids
        return self

    def _ipython_display_(self, **kwargs):
        from ipywidgets import HTML
        html = "<hr>".join(i.to_html(show_kids=self._show_kids) for i in self)
        return HTML(html)._ipython_display_(**kwargs)
