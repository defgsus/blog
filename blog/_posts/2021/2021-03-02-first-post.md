---
layout: post
date: 2021-03-02T00:30:42.928204+01:00
title: jupekyll
enable: plotly
custom_js: 
  - require-stub
  - plotly.min

---


Hi there!

I am yet another jupyter notebook converted to some markdownish/webish thing, this time using jekyll and a couple of ad-hoc helper scripts.

While more and more terrible stupid cruel stuff seems to happen in the world, my author feels ever more pulled into coding and publishing stuff on github. Is he happy about Microsoft owning it? Not much. We will see.. 

So he's somewhat skilled in programming, knows a couple of libraries, reads more source and documentation than anything else on the web and knows about these *github pages*, even has tried it once or twice.

So here am i! IPython. Yet another random notebook converted to the web (the old or the new? we don't know).

To publish myself, author did
```bash
python post-notebook.py ./src/first-post.ipynb
```

To see if a couple of jupyter notebook tricks work in the blog he added a table of blocked requests of today's jekyll/jupyter related browsing  







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
      <th>script</th>
      <th>stylesheet</th>
      <th>font</th>
      <th>image</th>
      <th>ping</th>
      <th>beacon</th>
      <th>other</th>
      <th>main_frame</th>
    </tr>
    <tr>
      <th>host</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>duckduckgo.com</th>
      <td>155</td>
      <td>57</td>
      <td>40</td>
      <td>23</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>use.typekit.net</th>
      <td></td>
      <td></td>
      <td>93</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>www.gravatar.com</th>
      <td></td>
      <td></td>
      <td></td>
      <td>53</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>i.stack.imgur.com</th>
      <td></td>
      <td></td>
      <td></td>
      <td>48</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>api.github.com</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>25</td>
      <td>17</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>localhost</th>
      <td>6</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>34</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nbconvert.readthedocs.io</th>
      <td>38</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>media.readthedocs.org</th>
      <td>29</td>
      <td>2</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>cdn.jsdelivr.net</th>
      <td>29</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>ajax.googleapis.com</th>
      <td>22</td>
      <td>4</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>collector.githubapp.com</th>
      <td></td>
      <td></td>
      <td></td>
      <td>22</td>
      <td>2</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>fonts.googleapis.com</th>
      <td></td>
      <td>23</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>stackpath.bootstrapcdn.com</th>
      <td>10</td>
      <td>9</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>assets.readthedocs.org</th>
      <td>10</td>
      <td>7</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>images.contentstack.io</th>
      <td></td>
      <td></td>
      <td></td>
      <td>16</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>maxcdn.bootstrapcdn.com</th>
      <td></td>
      <td>12</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>cdnjs.cloudflare.com</th>
      <td>10</td>
      <td>1</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>ryankuhn.net</th>
      <td>10</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



(They are blocked mostly thanks to umatrix)

Then he plotted the positions of his recorded mouse events of the past days into a png





    
![png]({{site.baseurl}}/assets/nb/2021-03-02-first-post_files/2021-03-02-first-post_4_0.png)
    


and the average free parking places across germany during the last year via plotly





<div>                            <div id="004ecb31-38ad-46c4-835c-44abf4f98737" class="plotly-graph-div" style="height:350px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("004ecb31-38ad-46c4-835c-44abf4f98737")) {                    Plotly.newPlot(                        "004ecb31-38ad-46c4-835c-44abf4f98737",                        [{"hovertemplate": "per week=%{x}<br>free parking %=%{y}<extra></extra>", "legendgroup": "", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": ["2020-03-23T00:00:00", "2020-03-30T00:00:00", "2020-04-06T00:00:00", "2020-04-13T00:00:00", "2020-04-20T00:00:00", "2020-04-27T00:00:00", "2020-05-04T00:00:00", "2020-05-11T00:00:00", "2020-05-18T00:00:00", "2020-05-25T00:00:00", "2020-06-01T00:00:00", "2020-06-08T00:00:00", "2020-06-15T00:00:00", "2020-06-22T00:00:00", "2020-06-29T00:00:00", "2020-07-06T00:00:00", "2020-07-13T00:00:00", "2020-07-20T00:00:00", "2020-07-27T00:00:00", "2020-08-03T00:00:00", "2020-08-10T00:00:00", "2020-08-17T00:00:00", "2020-08-24T00:00:00", "2020-08-31T00:00:00", "2020-09-07T00:00:00", "2020-09-14T00:00:00", "2020-09-21T00:00:00", "2020-09-28T00:00:00", "2020-10-05T00:00:00", "2020-10-12T00:00:00", "2020-10-19T00:00:00", "2020-10-26T00:00:00", "2020-11-02T00:00:00", "2020-11-09T00:00:00", "2020-11-16T00:00:00", "2020-11-23T00:00:00", "2020-11-30T00:00:00", "2020-12-07T00:00:00", "2020-12-14T00:00:00", "2020-12-21T00:00:00", "2020-12-28T00:00:00", "2021-01-04T00:00:00", "2021-01-11T00:00:00", "2021-01-18T00:00:00", "2021-01-25T00:00:00", "2021-02-01T00:00:00"], "xaxis": "x", "y": [81.39335406503318, 78.488042377512, 78.18305148624327, 78.12249005777372, 75.36820019351688, 74.38202060620233, 71.15440456502984, 68.83523803579847, 67.51763653937834, 66.32854457762568, 67.19746434626656, 64.51577024616326, 63.80912536241573, 65.19905999979292, 62.652927555057595, 62.01854085955365, 63.486908610281915, 63.41855486104345, 63.961098604381654, 63.22078221202305, 62.87007594949558, 61.81700884001577, 59.78846293201503, 59.93534889472693, 60.675525245645126, 60.09869017665972, 60.76959200862859, 61.03163464959719, 59.7515148236435, 60.987604822803135, 63.58939209302269, 64.56453300679019, 69.66440400607104, 68.39344694613271, 67.2122258710455, 65.41033082509455, 65.95598363193447, 64.01227467731701, 71.82144812889933, 81.29715495242107, 84.08032214066913, 77.93441354607342, 75.80797463604058, 75.49523422189849, 76.17777873411897, 75.10342660200449], "yaxis": "y"}],                        {"height": 350, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "per week"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "free parking %"}}},                        {"responsive": true}                    ).then(function(){


                        })                };                });            </script>        </div>




