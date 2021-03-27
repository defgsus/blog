---
layout: post
custom_css: 

custom_js: 
  - plotly.min.js
  - require-stub.js
title: plotly click events in jupyter and html-export
enable: plotly
tags: plotly elastipy interactive

---


The [Plotly](https://plotly.com) plotting library is really cool. It's actually written in javascript so it's interactive, integrates well with python and jupyter notebooks and it stays intact when converting [the notebook](https://github.com/defgsus/blog/blob/master/src/general/plotly-click-event.ipynb) to html like in this post.

It's possible to attach to plotly events, like clicking a point. They actually provide a [python callback](https://plotly.com/python/click-events/) when running inside the notebook which means the javascript event triggers a python script. Astonishing! But that won't work when exporting to html. Of course they also provide a [javascript callback](https://plotly.com/javascript/click-events/) so here's a way to use it in notebook and html export.


```python
from typing import Optional
import plotly
import plotly.graph_objects

# A small helper function to render a plotly
# Figure with attached javascript
def html_plotly(
        fig: plotly.graph_objects.Figure, 
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
        from IPython.display import display, HTML
        display(HTML(html))
    else:
        return html
```

Plotly's [to_html method](https://plotly.com/python-api-reference/generated/plotly.io.to_html.html) is quite powerful and convenient. And it allows passing javascript code where the special template tag `{plot_id}` will be replaced by the id of the plot `<div>` and more is not needed for catching the events.

This function can be called on a plotly figure, so let's create one using data from ... öhmm .. maybe .. [Überwachung für Alle!](https://github.com/defgsus/ufa), more precisely my personal keystrokes inside the browser during the last month.


```python
from elastipy import Search

keystrokes = (
    Search(index="ufa-events-keyboard")
        .range("timestamp", gte="2021-02-25", lte="2021-03-25")
        .agg_date_histogram("day", calendar_interval="day")
        .agg_terms("key", field="key", size=50)
        .execute().df()
        .replace({" ": "Space"})
        .set_index(["day", "key"])
)
```

This is asking **elasticsearch** for two aggregations: the number of keystrokes per day and the top 50 keys per day. The search is executed, converted to a pandas dataframe and beautified a bit.


```python
keystrokes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>day.doc_count</th>
      <th>key.doc_count</th>
    </tr>
    <tr>
      <th>day</th>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2021-02-25</th>
      <th>Control</th>
      <td>1768</td>
      <td>137</td>
    </tr>
    <tr>
      <th>Backspace</th>
      <td>1768</td>
      <td>130</td>
    </tr>
    <tr>
      <th>ArrowRight</th>
      <td>1768</td>
      <td>117</td>
    </tr>
    <tr>
      <th>ArrowLeft</th>
      <td>1768</td>
      <td>97</td>
    </tr>
    <tr>
      <th>Shift</th>
      <td>1768</td>
      <td>84</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2021-03-25</th>
      <th>-</th>
      <td>18132</td>
      <td>55</td>
    </tr>
    <tr>
      <th>Home</th>
      <td>18132</td>
      <td>47</td>
    </tr>
    <tr>
      <th>&lt;</th>
      <td>18132</td>
      <td>46</td>
    </tr>
    <tr>
      <th>&gt;</th>
      <td>18132</td>
      <td>43</td>
    </tr>
    <tr>
      <th>/</th>
      <td>18132</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
<p>1443 rows × 2 columns</p>
</div>



Then we build a python object that can be passed to javascript to *enhance* the plot experience.


```python
keystrokes_per_day = [
    {
        "date": date.to_pydatetime().isoformat(),
        "count": int(keystrokes.loc[date]["day.doc_count"][0]),
        "keys": list(keystrokes.xs(date)["key.doc_count"].items()),
    }
    for date in keystrokes.index.unique(level=0)
]
```

(This was done with some terrible code before reading [Tom Augspurger's blog](http://tomaugspurger.github.io) and more [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html)..)

Now create some piece of html/javascript that displays the top keys for each clicked day.


```python
import secrets
import json

# I'm writing these things mostly at night, so 
# dark mode is much more eye-friendly
plotly.io.templates.default = "plotly_dark"

# a random id to identify our data object and the html container 
ID = secrets.token_hex(8)

# a place-holder for the data
post_html = f"""
<div id="post-plot-{ID}">
    <div class="info"></div>
</div>
"""

# The usual vanilla-javascript DOM-mangling code 
# But one can use jquery or whatever
post_script = """
const data_%(ID)s = %(data)s;
function on_click(point_index) {
    const 
        data = data_%(ID)s[point_index],
        date_str = new Date(Date.parse(data.date)).toDateString();
    
    let html = `<h3>${date_str}: ${data.count} keystrokes</h3>`;
    html += `<table><tbody>`;
    html += `<tr><td>key</td> <td>count</td> <td>percent</td> </tr>`;
    html += data.keys.map(function(key) {
        return `<tr><td>${key[0]}</td> <td>${key[1]}</td>`
             + `<td>${Math.round(key[1] * 10000 / data.count) / 100}%%</td> </tr>`;
    }).join("") + "</tbody></table>";
    
    document.querySelector("#post-plot-%(ID)s .info").innerHTML = html;
}

// attach to plotly
document.getElementById("{plot_id}").on("plotly_click", function(click_data) {
    on_click(click_data.points[0].pointIndex);
});
""" % {"ID": ID, "data": json.dumps(keystrokes_per_day)}
```

And finally create an actual plot and attach the code.


```python
import plotly.express as px

fig = px.bar(
    keystrokes.groupby("day").first().rename(columns={"day.doc_count": "keystrokes"}),
    y="keystrokes",
    title="Click me!",
)

html_plotly(fig, post_script=post_script, post_html=post_html)
```


<div>                            <div id="ca784fc7-9500-4c9f-89c2-242fe4074b33" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ca784fc7-9500-4c9f-89c2-242fe4074b33")) {                    Plotly.newPlot(                        "ca784fc7-9500-4c9f-89c2-242fe4074b33",                        [{"alignmentgroup": "True", "hovertemplate": "day=%{x}<br>keystrokes=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa"}, "name": "", "offsetgroup": "", "orientation": "v", "showlegend": false, "textposition": "auto", "type": "bar", "x": ["2021-02-25T00:00:00", "2021-02-26T00:00:00", "2021-02-27T00:00:00", "2021-02-28T00:00:00", "2021-03-01T00:00:00", "2021-03-02T00:00:00", "2021-03-03T00:00:00", "2021-03-04T00:00:00", "2021-03-05T00:00:00", "2021-03-06T00:00:00", "2021-03-07T00:00:00", "2021-03-08T00:00:00", "2021-03-09T00:00:00", "2021-03-10T00:00:00", "2021-03-11T00:00:00", "2021-03-12T00:00:00", "2021-03-13T00:00:00", "2021-03-14T00:00:00", "2021-03-15T00:00:00", "2021-03-16T00:00:00", "2021-03-17T00:00:00", "2021-03-18T00:00:00", "2021-03-19T00:00:00", "2021-03-20T00:00:00", "2021-03-21T00:00:00", "2021-03-22T00:00:00", "2021-03-23T00:00:00", "2021-03-24T00:00:00", "2021-03-25T00:00:00"], "xaxis": "x", "y": [1768, 5976, 17814, 11420, 12719, 26769, 10125, 12285, 16899, 7176, 19003, 6056, 1205, 10208, 10263, 262, 3436, 11593, 10063, 3526, 3757, 15810, 12765, 6660, 5140, 21173, 10160, 9440, 18132], "yaxis": "y"}],                        {"barmode": "relative", "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}, "title": {"text": "Click me!"}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "day"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "keystrokes"}}},                        {"responsive": true}                    ).then(function(){

const data_2e8303beedfac2b8 = [{"date": "2021-02-25T00:00:00", "count": 1768, "keys": [["Control", 137], ["Backspace", 130], ["ArrowRight", 117], ["ArrowLeft", 97], ["Shift", 84], ["e", 76], ["Space", 69], ["Enter", 68], ["a", 61], ["Alt", 54], ["s", 52], ["t", 49], ["o", 42], ["r", 42], ["ArrowUp", 41], ["c", 38], ["ArrowDown", 36], ["n", 33], ["i", 32], ["Tab", 31], ["u", 31], ["d", 28], ["h", 28], ["F5", 27], ["l", 26], [".", 24], ["g", 23], ["b", 21], ["f", 20], ["p", 18], ["m", 17], ["\"", 15], ["v", 12], ["-", 10], ["w", 9], ["x", 9], ["#", 8], [",", 8], ["y", 8], ["(", 7], ["=", 7], ["Home", 7], ["+", 6], ["/", 6], ["0", 6], ["1", 5], ["5", 5], [":", 5], ["Process", 5], ["k", 5]]}, {"date": "2021-02-26T00:00:00", "count": 5976, "keys": [["ArrowRight", 517], ["Control", 512], ["ArrowLeft", 400], ["Shift", 375], ["Backspace", 305], ["e", 297], ["Space", 291], ["t", 216], ["s", 213], ["Enter", 212], ["n", 207], ["a", 177], ["i", 170], ["r", 155], ["o", 135], ["ArrowUp", 117], ["d", 109], ["ArrowDown", 108], ["c", 103], ["l", 94], ["u", 76], ["m", 75], ["h", 70], ["b", 63], ["p", 58], [".", 56], ["v", 56], ["Alt", 52], ["(", 49], ["\"", 45], ["f", 45], ["g", 41], ["AltGraph", 31], ["k", 30], ["x", 27], [")", 24], ["w", 24], ["Home", 23], [",", 21], [":", 20], ["_", 19], ["y", 17], ["[", 15], ["=", 14], ["2", 13], ["-", 12], ["A", 12], ["Tab", 12], ["Process", 11], ["#", 10]]}, {"date": "2021-02-27T00:00:00", "count": 17814, "keys": [["ArrowRight", 1690], ["Shift", 1569], ["ArrowLeft", 1298], ["Control", 1115], ["ArrowDown", 944], ["ArrowUp", 861], ["Backspace", 857], ["Enter", 809], ["e", 680], ["t", 591], ["Space", 518], ["s", 480], ["a", 464], ["r", 457], ["i", 424], ["m", 293], ["n", 245], ["d", 241], ["o", 239], ["c", 231], ["l", 221], ["g", 206], ["(", 203], ["p", 196], ["\"", 192], [".", 183], ["_", 181], ["=", 179], ["f", 166], [",", 163], ["u", 139], ["v", 130], ["y", 126], ["h", 119], [")", 106], ["x", 104], ["Home", 95], ["AltGraph", 89], ["End", 87], ["Tab", 83], ["#", 82], ["0", 68], ["b", 62], [":", 60], ["Alt", 48], ["k", 42], ["1", 37], ["[", 37], ["z", 33], ["Escape", 32]]}, {"date": "2021-02-28T00:00:00", "count": 11420, "keys": [["Shift", 1073], ["ArrowRight", 1057], ["Control", 761], ["ArrowLeft", 749], ["Backspace", 628], ["Enter", 492], ["ArrowDown", 443], ["e", 429], ["t", 383], ["ArrowUp", 362], ["Space", 339], ["s", 301], ["r", 282], ["a", 277], ["i", 238], ["o", 228], ["d", 221], ["l", 203], ["c", 193], ["\"", 179], ["n", 157], ["u", 141], [".", 130], ["m", 129], ["_", 126], ["(", 121], [",", 118], ["p", 105], ["f", 97], ["h", 97], ["g", 89], ["v", 85], ["=", 74], ["y", 68], ["Tab", 67], [")", 66], ["x", 66], ["Alt", 62], ["AltGraph", 57], ["b", 49], [":", 47], ["z", 47], ["Home", 42], ["w", 42], ["End", 36], ["k", 36], ["0", 35], ["#", 31], ["Escape", 24], ["[", 23]]}, {"date": "2021-03-01T00:00:00", "count": 12719, "keys": [["ArrowRight", 1144], ["Shift", 1009], ["Backspace", 915], ["Control", 865], ["ArrowLeft", 777], ["Space", 557], ["e", 541], ["Enter", 534], ["t", 409], ["ArrowUp", 361], ["ArrowDown", 353], ["a", 343], ["r", 316], ["o", 310], ["i", 286], ["s", 283], ["n", 239], ["c", 216], ["l", 214], ["d", 198], ["Alt", 179], ["m", 160], ["h", 158], ["p", 151], ["u", 119], ["g", 116], [".", 113], ["f", 111], ["0", 103], ["\"", 101], ["y", 101], ["_", 98], ["(", 88], [",", 82], ["v", 82], ["=", 77], ["b", 65], ["1", 58], ["#", 56], ["k", 55], [":", 54], [")", 47], ["w", 45], ["x", 44], ["Tab", 40], ["AltGraph", 39], ["-", 33], ["2", 32], ["Escape", 29], ["5", 28]]}, {"date": "2021-03-02T00:00:00", "count": 26769, "keys": [["ArrowRight", 2419], ["Shift", 1851], ["Control", 1711], ["Space", 1677], ["Backspace", 1633], ["ArrowLeft", 1547], ["e", 1296], ["t", 1061], ["a", 842], ["r", 763], ["Enter", 754], ["s", 727], ["i", 695], ["o", 647], ["ArrowDown", 562], ["n", 529], ["ArrowUp", 520], ["c", 508], ["u", 465], ["l", 458], ["h", 431], ["d", 414], ["m", 396], ["p", 300], [".", 276], ["g", 226], ["y", 218], ["v", 217], ["f", 201], ["\"", 200], ["(", 164], ["b", 154], ["w", 151], [",", 147], ["Home", 132], ["AltGraph", 131], ["Alt", 128], ["k", 121], ["0", 116], [")", 106], ["End", 103], ["#", 99], ["_", 95], ["=", 79], ["Tab", 75], ["Escape", 70], ["x", 69], ["T", 67], ["*", 63], ["2", 63]]}, {"date": "2021-03-03T00:00:00", "count": 10125, "keys": [["ArrowRight", 1033], ["Shift", 787], ["Backspace", 675], ["Control", 610], ["ArrowLeft", 598], ["ArrowUp", 449], ["ArrowDown", 445], ["e", 440], ["Enter", 413], ["r", 309], ["Space", 304], ["t", 255], ["s", 247], ["a", 234], ["i", 223], ["l", 201], ["o", 179], ["d", 165], ["n", 159], ["f", 142], ["\"", 128], ["p", 119], ["(", 111], ["AltGraph", 108], ["u", 103], [".", 100], ["g", 98], ["c", 96], ["v", 80], ["m", 79], [",", 77], ["h", 76], ["[", 70], [":", 66], ["Home", 63], ["_", 57], ["=", 56], ["0", 55], ["y", 54], ["w", 50], [")", 49], ["Alt", 49], ["b", 48], ["k", 42], ["Tab", 36], ["1", 28], ["z", 28], ["2", 23], ["]", 22], ["#", 21]]}, {"date": "2021-03-04T00:00:00", "count": 12285, "keys": [["ArrowRight", 1155], ["Control", 928], ["ArrowLeft", 798], ["Shift", 770], ["Backspace", 749], ["Space", 642], ["e", 596], ["ArrowDown", 588], ["ArrowUp", 577], ["t", 377], ["Enter", 345], ["s", 335], ["o", 295], ["a", 294], ["r", 287], ["i", 285], ["n", 255], ["c", 206], ["v", 161], ["l", 158], ["h", 149], ["d", 140], ["m", 127], ["\"", 117], ["f", 115], ["u", 107], ["p", 94], ["g", 93], ["AltGraph", 80], ["y", 79], [".", 77], ["b", 77], ["Home", 74], ["w", 70], [",", 69], ["(", 64], [":", 61], ["Alt", 55], ["[", 50], ["End", 45], ["k", 44], ["z", 41], ["_", 38], ["=", 37], ["x", 35], ["q", 32], ["0", 31], [")", 29], ["S", 27], ["1", 23]]}, {"date": "2021-03-05T00:00:00", "count": 16899, "keys": [["ArrowRight", 1382], ["Backspace", 1160], ["Control", 1121], ["Space", 1112], ["Shift", 1058], ["ArrowLeft", 957], ["e", 788], ["t", 612], ["Enter", 566], ["ArrowDown", 545], ["a", 518], ["s", 514], ["i", 498], ["ArrowUp", 483], ["o", 466], ["n", 409], ["r", 390], ["l", 316], ["c", 296], ["h", 287], ["d", 269], ["u", 178], ["p", 169], ["v", 165], ["g", 142], ["m", 138], ["w", 138], ["b", 137], [".", 126], ["Alt", 114], ["f", 112], ["y", 100], ["\"", 92], [",", 85], ["AltGraph", 81], ["(", 71], ["Home", 64], ["x", 54], ["z", 53], ["k", 50], [")", 48], ["*", 48], ["End", 46], ["[", 46], ["1", 45], ["=", 44], ["-", 40], ["2", 38], ["#", 34], ["'", 32]]}, {"date": "2021-03-06T00:00:00", "count": 7176, "keys": [["ArrowRight", 634], ["Shift", 615], ["Backspace", 417], ["Control", 375], ["ArrowLeft", 360], ["Space", 302], ["e", 302], ["ArrowDown", 291], ["Enter", 287], ["ArrowUp", 282], ["t", 231], ["o", 198], ["i", 191], ["n", 183], ["s", 179], ["a", 163], ["r", 157], ["c", 115], ["d", 98], ["\"", 94], ["u", 84], ["p", 80], ["(", 79], ["f", 79], ["l", 79], ["AltGraph", 73], ["_", 70], ["m", 69], ["=", 68], ["h", 65], ["Alt", 63], ["[", 60], [".", 58], ["v", 55], [",", 52], [":", 47], ["0", 45], ["Home", 44], ["g", 41], ["y", 37], [")", 35], ["End", 35], ["x", 35], ["Tab", 32], ["k", 30], ["+", 28], ["z", 27], ["1", 25], ["b", 22], ["#", 19]]}, {"date": "2021-03-07T00:00:00", "count": 19003, "keys": [["ArrowRight", 1657], ["Control", 1281], ["Shift", 1258], ["Backspace", 1144], ["ArrowLeft", 1110], ["ArrowDown", 881], ["Space", 840], ["e", 772], ["Enter", 742], ["ArrowUp", 686], ["t", 610], ["s", 562], ["o", 542], ["i", 517], ["r", 496], ["n", 470], ["a", 429], ["d", 315], ["c", 313], ["h", 293], ["l", 291], ["p", 199], ["m", 195], ["v", 193], [".", 174], ["AltGraph", 172], ["u", 169], ["f", 168], ["(", 162], ["\"", 136], ["g", 135], ["Alt", 118], ["Home", 118], ["b", 115], [",", 103], ["[", 100], ["=", 96], ["_", 94], [")", 89], ["End", 82], ["y", 80], [":", 77], ["x", 74], ["*", 73], ["0", 73], ["1", 64], ["w", 55], ["z", 49], ["k", 46], ["Tab", 44]]}, {"date": "2021-03-08T00:00:00", "count": 6056, "keys": [["Shift", 478], ["ArrowRight", 452], ["Control", 368], ["Backspace", 324], ["Enter", 278], ["ArrowLeft", 257], ["ArrowUp", 243], ["ArrowDown", 242], ["r", 240], ["e", 225], ["Space", 223], ["s", 211], ["t", 211], ["i", 188], ["o", 145], ["a", 139], ["n", 110], ["l", 104], ["d", 103], ["h", 100], ["Alt", 87], ["AltGraph", 82], [".", 79], ["p", 79], ["\"", 75], ["c", 66], ["(", 65], ["u", 57], ["f", 56], ["m", 55], ["[", 50], ["_", 50], ["=", 41], [":", 39], ["v", 37], ["Home", 36], ["End", 33], ["w", 30], [")", 29], ["g", 29], ["0", 28], ["z", 28], [",", 25], ["y", 25], ["q", 23], ["x", 22], ["k", 20], ["Tab", 14], ["-", 13], ["/", 13]]}, {"date": "2021-03-09T00:00:00", "count": 1205, "keys": [["Alt", 128], ["Control", 93], ["0", 77], ["Shift", 76], ["Backspace", 64], ["ArrowRight", 51], ["F5", 46], ["Tab", 45], ["Space", 42], ["ArrowLeft", 41], ["Enter", 35], ["e", 30], ["r", 28], ["a", 25], ["1", 24], ["+", 23], [":", 21], ["c", 18], ["o", 18], ["\"", 16], ["2", 15], ["h", 15], ["-", 14], ["i", 13], ["n", 13], ["d", 12], ["t", 12], ["s", 11], [".", 10], ["AltGraph", 10], ["z", 9], ["u", 8], ["l", 7], ["3", 6], ["k", 6], ["m", 6], ["v", 6], ["ArrowUp", 5], ["b", 5], ["p", 5], ["/", 4], ["f", 4], ["g", 4], ["w", 4], ["y", 4], ["#", 3], [",", 3], ["5", 3], ["=", 3], ["ArrowDown", 3]]}, {"date": "2021-03-10T00:00:00", "count": 10208, "keys": [["ArrowRight", 842], ["Shift", 828], ["Control", 639], ["ArrowLeft", 618], ["Backspace", 608], ["ArrowDown", 495], ["ArrowUp", 457], ["Enter", 420], ["Space", 377], ["e", 373], ["t", 291], ["a", 261], ["s", 257], ["r", 256], ["o", 243], ["i", 223], ["n", 195], ["l", 189], ["d", 155], ["c", 143], ["p", 130], ["m", 122], ["_", 113], ["\"", 106], ["v", 104], ["(", 94], ["f", 92], ["x", 92], ["h", 85], ["Alt", 81], ["=", 75], ["End", 74], ["AltGraph", 71], ["b", 70], ["y", 70], [",", 68], ["u", 66], [":", 65], ["g", 64], [".", 59], ["0", 56], ["Home", 51], ["[", 38], ["#", 32], ["1", 28], ["w", 25], ["Tab", 24], ["/", 23], [")", 22], ["{", 19]]}, {"date": "2021-03-11T00:00:00", "count": 10263, "keys": [["ArrowRight", 978], ["Control", 948], ["Shift", 768], ["ArrowLeft", 650], ["Enter", 538], ["Backspace", 536], ["e", 366], ["ArrowDown", 344], ["ArrowUp", 291], ["i", 284], ["o", 267], ["t", 266], ["Space", 264], ["s", 263], ["r", 247], ["a", 232], ["n", 215], ["c", 192], ["d", 164], ["l", 131], ["m", 131], ["u", 114], ["p", 108], ["f", 106], ["v", 104], ["Alt", 96], ["\"", 95], [".", 87], ["h", 86], ["AltGraph", 73], ["g", 72], ["b", 71], ["(", 63], ["_", 62], ["Home", 60], ["End", 56], ["x", 56], ["w", 55], ["y", 49], ["#", 48], ["Escape", 48], ["0", 45], ["[", 45], [",", 43], [":", 43], ["=", 40], ["q", 32], ["-", 27], [")", 24], ["k", 22]]}, {"date": "2021-03-12T00:00:00", "count": 262, "keys": [["Alt", 52], ["Control", 30], ["ArrowRight", 14], ["ArrowLeft", 13], ["Shift", 12], ["c", 12], ["Backspace", 10], ["Enter", 9], ["e", 8], ["ArrowDown", 6], ["a", 6], ["Space", 5], ["ArrowUp", 5], ["r", 5], ["s", 5], ["d", 4], ["h", 4], ["_", 3], ["i", 3], ["n", 3], ["z", 3], ["Home", 2], ["S", 2], ["f", 2], ["o", 2], ["t", 2], ["v", 2], ["#", 1], ["*", 1], [".", 1], ["1", 1], ["3", 1], ["A", 1], ["D", 1], ["E", 1], ["End", 1], ["R", 1], ["g", 1], ["l", 1], ["m", 1], ["p", 1], ["u", 1], ["w", 1]]}, {"date": "2021-03-13T00:00:00", "count": 3436, "keys": [["ArrowRight", 325], ["Shift", 259], ["Control", 249], ["Backspace", 201], ["ArrowLeft", 189], ["Enter", 174], ["ArrowUp", 157], ["ArrowDown", 148], ["e", 148], ["Space", 115], ["s", 108], ["r", 90], ["t", 88], ["n", 83], ["i", 77], ["l", 59], ["a", 55], ["AltGraph", 49], ["c", 48], ["\"", 46], ["u", 46], ["o", 43], ["d", 41], ["(", 36], ["[", 34], ["m", 34], ["p", 34], ["Alt", 33], ["f", 25], [".", 24], ["y", 24], ["x", 23], [",", 22], ["=", 21], ["z", 21], ["Home", 20], ["End", 19], [")", 17], ["1", 16], [":", 16], ["_", 16], ["b", 16], ["0", 15], ["k", 15], ["v", 14], ["Escape", 13], ["h", 12], ["#", 9], ["Tab", 8], ["]", 8]]}, {"date": "2021-03-14T00:00:00", "count": 11593, "keys": [["ArrowRight", 1239], ["Control", 854], ["Shift", 818], ["ArrowLeft", 773], ["Backspace", 708], ["ArrowDown", 509], ["e", 501], ["s", 500], ["t", 405], ["Enter", 400], ["ArrowUp", 399], ["Space", 398], ["r", 322], ["i", 304], ["a", 194], ["d", 179], ["l", 172], ["u", 165], ["n", 163], ["AltGraph", 162], ["o", 159], ["\"", 139], ["(", 122], ["[", 106], ["_", 105], ["h", 105], ["p", 103], ["c", 96], ["b", 92], ["v", 92], ["w", 92], ["m", 87], ["f", 79], [".", 78], [",", 67], [")", 66], ["x", 64], ["Home", 57], ["=", 56], ["End", 51], ["g", 49], [":", 48], ["q", 45], ["Alt", 44], ["y", 38], ["z", 36], ["0", 26], ["]", 26], ["#", 24], ["k", 23]]}, {"date": "2021-03-15T00:00:00", "count": 10063, "keys": [["ArrowRight", 714], ["Shift", 685], ["Control", 629], ["Space", 598], ["Backspace", 581], ["ArrowLeft", 438], ["Enter", 412], ["ArrowDown", 396], ["ArrowUp", 395], ["e", 392], ["t", 392], ["a", 349], ["s", 303], ["o", 249], ["r", 247], ["i", 243], ["l", 203], ["n", 191], ["d", 171], ["h", 155], [".", 139], ["c", 126], ["p", 125], ["Alt", 117], ["m", 117], ["b", 113], ["u", 92], ["y", 81], ["v", 79], ["(", 76], ["f", 69], [",", 59], ["\"", 57], ["j", 54], ["x", 52], ["w", 51], ["_", 47], ["=", 43], ["Home", 43], ["g", 42], ["#", 40], ["End", 36], [")", 35], ["AltGraph", 32], ["Escape", 31], ["T", 28], ["D", 25], ["k", 25], ["*", 24], [":", 24]]}, {"date": "2021-03-16T00:00:00", "count": 3526, "keys": [["Space", 323], ["e", 223], ["Control", 213], ["ArrowRight", 196], ["ArrowLeft", 176], ["Backspace", 172], ["Shift", 153], ["a", 153], ["t", 153], ["i", 124], ["o", 122], ["n", 116], ["r", 114], ["s", 103], ["d", 81], ["l", 78], ["h", 77], ["c", 76], ["ArrowDown", 58], ["ArrowUp", 54], ["Alt", 53], ["Enter", 53], ["u", 51], ["m", 41], ["p", 40], ["b", 36], ["f", 36], ["w", 36], ["v", 31], ["y", 29], ["g", 28], ["*", 22], ["-", 21], [",", 16], [".", 13], ["k", 13], ["AltGraph", 11], ["Tab", 11], ["#", 10], ["(", 9], [")", 9], ["Escape", 9], ["S", 9], ["Home", 8], ["z", 8], ["0", 7], [":", 7], ["A", 7], ["I", 7], ["[", 7]]}, {"date": "2021-03-17T00:00:00", "count": 3757, "keys": [["Backspace", 308], ["Space", 235], ["Shift", 221], ["ArrowRight", 202], ["e", 194], ["Control", 183], ["ArrowLeft", 147], ["n", 135], ["Enter", 125], ["a", 120], ["t", 117], ["r", 116], ["i", 115], ["ArrowDown", 113], ["ArrowUp", 112], ["s", 107], ["o", 106], ["d", 78], ["l", 72], ["Alt", 66], ["h", 65], ["m", 62], ["p", 54], [".", 44], ["(", 36], ["c", 32], ["u", 31], ["AltGraph", 28], ["=", 27], ["b", 27], ["g", 27], ["\"", 26], ["w", 25], ["v", 24], ["k", 22], [",", 21], ["y", 19], ["f", 18], [")", 16], ["0", 16], [":", 16], ["Tab", 15], ["[", 15], ["Home", 14], ["_", 13], ["<", 10], ["End", 10], ["1", 9], ["x", 9], ["#", 8]]}, {"date": "2021-03-18T00:00:00", "count": 15810, "keys": [["Space", 1454], ["ArrowRight", 1250], ["e", 997], ["ArrowLeft", 953], ["Control", 946], ["Backspace", 862], ["t", 721], ["Shift", 583], ["a", 581], ["s", 576], ["r", 568], ["i", 550], ["o", 540], ["n", 438], ["ArrowDown", 399], ["h", 388], ["d", 353], ["l", 326], ["ArrowUp", 302], ["c", 246], ["Enter", 221], ["u", 208], ["p", 180], ["m", 172], ["f", 166], ["b", 154], ["w", 154], ["y", 136], ["g", 132], ["v", 119], ["*", 106], ["Alt", 95], [".", 80], ["\"", 73], [",", 58], ["k", 47], ["-", 35], ["x", 33], ["T", 30], ["(", 27], ["z", 27], ["AltGraph", 24], [":", 22], ["0", 21], ["#", 19], ["=", 19], ["Home", 19], ["'", 18], ["End", 18], [">", 16]]}, {"date": "2021-03-19T00:00:00", "count": 12765, "keys": [["ArrowRight", 1204], ["Control", 988], ["ArrowLeft", 848], ["Space", 781], ["Backspace", 720], ["Shift", 635], ["e", 567], ["t", 508], ["ArrowDown", 464], ["ArrowUp", 409], ["s", 407], ["r", 376], ["i", 366], ["o", 360], ["a", 351], ["Enter", 318], ["n", 298], ["d", 228], ["h", 221], ["c", 184], ["l", 179], ["u", 152], ["p", 143], ["w", 132], ["m", 126], ["f", 108], ["Alt", 105], ["AltGraph", 96], ["v", 92], [".", 88], ["y", 86], ["\"", 84], ["b", 84], ["(", 75], ["g", 69], ["[", 64], ["_", 56], ["=", 50], ["k", 47], [")", 44], ["#", 41], ["End", 37], ["Home", 35], ["-", 33], [":", 33], ["x", 32], [",", 31], ["z", 23], ["Escape", 22], ["Tab", 22]]}, {"date": "2021-03-20T00:00:00", "count": 6660, "keys": [["ArrowRight", 511], ["Backspace", 499], ["Shift", 490], ["Control", 439], ["ArrowLeft", 403], ["ArrowDown", 392], ["ArrowUp", 348], ["Enter", 265], ["Space", 209], ["a", 206], ["t", 193], ["s", 150], ["l", 147], ["o", 140], ["r", 134], ["e", 131], ["m", 129], ["i", 121], ["n", 97], ["c", 88], ["p", 87], ["d", 81], ["x", 73], ["(", 67], ["h", 65], ["Alt", 63], ["v", 63], ["AltGraph", 59], [".", 57], ["_", 56], ["\"", 43], ["y", 42], ["=", 41], ["u", 41], [",", 38], ["f", 35], ["Tab", 33], [")", 29], ["End", 29], ["Home", 29], ["/", 28], [";", 28], ["g", 27], ["[", 23], ["0", 22], ["F5", 22], ["#", 19], ["b", 17], ["{", 17], ["1", 16]]}, {"date": "2021-03-21T00:00:00", "count": 5140, "keys": [["Backspace", 335], ["Control", 329], ["ArrowRight", 309], ["Shift", 300], ["Space", 243], ["e", 241], ["ArrowLeft", 231], ["s", 204], ["t", 200], ["o", 175], ["Enter", 167], ["a", 160], ["r", 160], ["ArrowUp", 151], ["Alt", 141], ["ArrowDown", 140], ["i", 140], ["h", 111], ["l", 103], ["n", 88], ["m", 75], ["F5", 74], ["d", 74], ["_", 72], ["f", 67], ["c", 50], ["u", 50], ["v", 44], ["(", 43], ["b", 38], ["Tab", 36], ["x", 36], [".", 35], ["=", 35], ["g", 32], ["p", 32], ["AltGraph", 30], ["\"", 28], ["w", 28], [",", 27], ["y", 27], ["[", 21], ["k", 20], [":", 19], ["Home", 19], [")", 18], ["Escape", 15], ["End", 14], ["#", 12], ["0", 10]]}, {"date": "2021-03-22T00:00:00", "count": 21173, "keys": [["ArrowRight", 1694], ["Control", 1550], ["Shift", 1445], ["ArrowLeft", 1308], ["Space", 1131], ["Backspace", 1060], ["e", 936], ["ArrowDown", 901], ["t", 844], ["s", 842], ["o", 779], ["ArrowUp", 676], ["Enter", 662], ["r", 536], ["i", 529], ["a", 515], ["h", 472], ["n", 440], ["_", 343], ["d", 319], ["l", 271], ["c", 252], ["m", 238], ["v", 211], ["u", 197], ["f", 191], ["(", 159], ["b", 147], ["Alt", 143], ["w", 140], ["g", 137], ["p", 134], [",", 125], ["AltGraph", 116], [".", 105], ["y", 97], ["x", 95], ["=", 86], [")", 80], [":", 77], ["0", 68], ["Tab", 66], ["End", 63], ["#", 59], ["Escape", 56], ["[", 56], ["\"", 53], ["k", 53], ["Home", 45], ["<", 43]]}, {"date": "2021-03-23T00:00:00", "count": 10160, "keys": [["ArrowRight", 702], ["Shift", 648], ["Control", 642], ["Space", 592], ["Backspace", 528], ["ArrowLeft", 513], ["e", 403], ["ArrowDown", 400], ["i", 345], ["ArrowUp", 342], ["t", 323], ["Enter", 321], ["s", 302], ["r", 295], ["n", 290], ["o", 285], ["a", 265], ["d", 174], ["l", 169], ["h", 157], ["v", 130], ["m", 128], ["c", 126], ["u", 108], ["AltGraph", 103], ["_", 97], ["(", 96], ["=", 87], ["f", 86], ["p", 84], ["w", 83], ["Alt", 81], [",", 76], ["g", 75], ["k", 70], ["[", 67], ["b", 58], [":", 54], ["*", 52], ["-", 52], ["\"", 51], [".", 51], ["x", 48], ["End", 44], ["Tab", 44], [")", 42], ["0", 40], ["Home", 38], ["y", 38], ["#", 31]]}, {"date": "2021-03-24T00:00:00", "count": 9440, "keys": [["ArrowRight", 896], ["Control", 689], ["Shift", 606], ["Backspace", 594], ["ArrowLeft", 564], ["ArrowDown", 536], ["ArrowUp", 432], ["Space", 395], ["Enter", 345], ["e", 322], ["i", 294], ["s", 271], ["t", 262], ["a", 261], ["r", 193], ["n", 168], ["c", 165], ["d", 152], ["l", 149], ["o", 149], ["h", 135], ["m", 126], ["_", 98], ["v", 88], ["(", 86], ["Alt", 86], ["g", 78], [",", 74], ["f", 74], ["k", 69], ["p", 62], ["u", 55], ["b", 50], ["2", 48], ["Home", 47], ["x", 47], ["1", 45], [":", 45], ["End", 45], [".", 44], ["=", 44], ["AltGraph", 41], ["y", 40], [")", 35], ["Escape", 34], ["*", 33], ["\"", 32], ["#", 31], ["Tab", 28], ["-", 27]]}, {"date": "2021-03-25T00:00:00", "count": 18132, "keys": [["ArrowRight", 1522], ["ArrowLeft", 1298], ["Control", 1287], ["Shift", 1173], ["Backspace", 931], ["Space", 870], ["ArrowDown", 720], ["e", 702], ["t", 689], ["Enter", 671], ["ArrowUp", 577], ["a", 528], ["i", 443], ["o", 436], ["s", 431], ["l", 383], ["n", 357], ["r", 351], ["d", 329], ["c", 325], ["p", 257], ["h", 219], ["m", 201], ["y", 176], ["_", 170], [".", 153], ["u", 152], ["g", 148], ["v", 145], ["f", 141], ["Process", 133], ["(", 131], ["k", 111], [",", 108], ["AltGraph", 101], ["Alt", 96], ["\"", 91], ["=", 89], ["b", 86], ["x", 79], ["#", 67], ["Escape", 61], ["w", 60], ["Tab", 58], [")", 56], ["-", 55], ["Home", 47], ["<", 46], [">", 43], ["/", 40]]}];
function on_click(point_index) {
    const 
        data = data_2e8303beedfac2b8[point_index],
        date_str = new Date(Date.parse(data.date)).toDateString();

    let html = `<h3>${date_str}: ${data.count} keystrokes</h3>`;
    html += `<table><tbody>`;
    html += `<tr><td>key</td> <td>count</td> <td>percent</td> </tr>`;
    html += data.keys.map(function(key) {
        return `<tr><td>${key[0]}</td> <td>${key[1]}</td>`
             + `<td>${Math.round(key[1] * 10000 / data.count) / 100}%</td> </tr>`;
    }).join("") + "</tbody></table>";

    document.querySelector("#post-plot-2e8303beedfac2b8 .info").innerHTML = html;
}

// attach to plotly
document.getElementById("ca784fc7-9500-4c9f-89c2-242fe4074b33").on("plotly_click", function(click_data) {
    on_click(click_data.points[0].pointIndex);
});

                        })                };                });            </script>        </div>
<div id="post-plot-2e8303beedfac2b8">
    <div class="info"></div>
</div>



Obviously, i'm a right-arrow affine, 4-day-interval writer.