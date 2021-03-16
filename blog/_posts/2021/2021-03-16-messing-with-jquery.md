---
layout: post
title: Messin' with jQuery and Jupyter
enable: datatables
custom_js: 
  - jquery-3.6.0.min.js
  - jquery.dataTables-1.10.24.min.js
custom_css: 
  - jquery.dataTables.min.css

---


Just yesterday, my tables got too long!

#### First reaction
Maybe i can just attach a piece of javascript to the pandas.DataFrame rendering to make them at least sortable.

#### Second to kind o'third reaction
A well.. let's look up someone elses code first before spending two nights implementing horrible javascript templating within python strings kind o'stuff.

#### Fourth to sixteenth reaction
Ohh! There's a jQuery plugin called [DataTables](https://datatables.net/manual/options) which can add sorting and paging to pure html tables. There must be something for jupyter already.. Let's try [itables](https://github.com/mwouts/itables)... Ohh, i need to adjust my script blocker because everything is loaded from `cdn.datatables.net`... Aah, they use this jupyter built-in `requirejs`, maybe i can just replace the paths... Ohh, how do i provide them to the notebook?
Ahh, maybe just render a `<script>` tag into an output cell... Uhm, `$(...).DataTables is not a function`.. Maybe if i add it to the header of the notebook html.. Öh, how exactly? ... C'mon! ... F** S** .. Aaahh. Just write an [nbextension](https://github.com/ipython-contrib/jupyter_contrib_nbextensions)... What the? ... Ömmh, have all these javascript errors been there before? .. Oh! It works! It works! Let's .. oh, it doesn't work ...

... and so on.

Finally, there is [this extension](https://github.com/defgsus/blog/tree/master/src/datatables-offline) which just cruedly injects the DataTables into the jupyter-provided jquery. I find it often complicated when the framework provides jquery but one wants to add another plugin.

The css got adjusted a bit so it matches with my `onedork` [jupyter theme](https://github.com/dunovank/jupyter-themes). Also the images are provided by the extension and do not require a CDN.

Now is it possible to render a blog post with this homebrewn setup? 

Here's the standard DataFrame representation:


```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.uniform(0, 100, (30, 10))).round()
df.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76.0</td>
      <td>68.0</td>
      <td>77.0</td>
      <td>30.0</td>
      <td>67.0</td>
      <td>68.0</td>
      <td>44.0</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33.0</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>97.0</td>
      <td>83.0</td>
      <td>62.0</td>
      <td>98.0</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.0</td>
      <td>55.0</td>
      <td>96.0</td>
      <td>91.0</td>
      <td>18.0</td>
      <td>79.0</td>
      <td>86.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92.0</td>
      <td>84.0</td>
      <td>50.0</td>
      <td>6.0</td>
      <td>33.0</td>
      <td>96.0</td>
      <td>67.0</td>
      <td>21.0</td>
      <td>85.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99.0</td>
      <td>6.0</td>
      <td>96.0</td>
      <td>77.0</td>
      <td>36.0</td>
      <td>34.0</td>
      <td>68.0</td>
      <td>97.0</td>
      <td>78.0</td>
      <td>42.0</td>
    </tr>
  </tbody>
</table>
</div>



It is already quite cool and useful, but not for 500 rows.. 

The DataTables enhanced version:


```python
import json
import secrets

from IPython.display import display, HTML
```


```python
def datatable(
    df: pd.DataFrame,
    max_rows: int = 1000,
    max_cols: int = None,
    paging: bool = True,
    table_id: str = None,
    **kwargs,
):    
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

datatable(df, paging=True)
```


<table border="1" class="dataframe" id="table-2344fbd85f84944566e4">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76.0</td>
      <td>68.0</td>
      <td>77.0</td>
      <td>30.0</td>
      <td>67.0</td>
      <td>68.0</td>
      <td>44.0</td>
      <td>36.0</td>
      <td>35.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33.0</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>97.0</td>
      <td>83.0</td>
      <td>62.0</td>
      <td>98.0</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.0</td>
      <td>55.0</td>
      <td>96.0</td>
      <td>91.0</td>
      <td>18.0</td>
      <td>79.0</td>
      <td>86.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92.0</td>
      <td>84.0</td>
      <td>50.0</td>
      <td>6.0</td>
      <td>33.0</td>
      <td>96.0</td>
      <td>67.0</td>
      <td>21.0</td>
      <td>85.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99.0</td>
      <td>6.0</td>
      <td>96.0</td>
      <td>77.0</td>
      <td>36.0</td>
      <td>34.0</td>
      <td>68.0</td>
      <td>97.0</td>
      <td>78.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>70.0</td>
      <td>76.0</td>
      <td>67.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>100.0</td>
      <td>29.0</td>
      <td>58.0</td>
      <td>49.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14.0</td>
      <td>18.0</td>
      <td>25.0</td>
      <td>26.0</td>
      <td>93.0</td>
      <td>47.0</td>
      <td>78.0</td>
      <td>2.0</td>
      <td>25.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>64.0</td>
      <td>6.0</td>
      <td>28.0</td>
      <td>56.0</td>
      <td>55.0</td>
      <td>65.0</td>
      <td>10.0</td>
      <td>34.0</td>
      <td>55.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>65.0</td>
      <td>72.0</td>
      <td>87.0</td>
      <td>45.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>10.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>66.0</td>
      <td>53.0</td>
      <td>54.0</td>
      <td>87.0</td>
      <td>24.0</td>
      <td>79.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>57.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>32.0</td>
      <td>95.0</td>
      <td>63.0</td>
      <td>94.0</td>
      <td>86.0</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>57.0</td>
      <td>81.0</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23.0</td>
      <td>22.0</td>
      <td>58.0</td>
      <td>17.0</td>
      <td>33.0</td>
      <td>16.0</td>
      <td>85.0</td>
      <td>95.0</td>
      <td>19.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>99.0</td>
      <td>62.0</td>
      <td>33.0</td>
      <td>88.0</td>
      <td>78.0</td>
      <td>19.0</td>
      <td>28.0</td>
      <td>42.0</td>
      <td>61.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>42.0</td>
      <td>49.0</td>
      <td>45.0</td>
      <td>30.0</td>
      <td>35.0</td>
      <td>27.0</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>30.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>86.0</td>
      <td>75.0</td>
      <td>7.0</td>
      <td>29.0</td>
      <td>32.0</td>
      <td>2.0</td>
      <td>98.0</td>
      <td>18.0</td>
      <td>89.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>78.0</td>
      <td>98.0</td>
      <td>80.0</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>81.0</td>
      <td>35.0</td>
      <td>96.0</td>
      <td>50.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>32.0</td>
      <td>78.0</td>
      <td>18.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>64.0</td>
      <td>4.0</td>
      <td>69.0</td>
      <td>24.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>99.0</td>
      <td>17.0</td>
      <td>27.0</td>
      <td>83.0</td>
      <td>33.0</td>
      <td>29.0</td>
      <td>56.0</td>
      <td>20.0</td>
      <td>25.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4.0</td>
      <td>14.0</td>
      <td>89.0</td>
      <td>95.0</td>
      <td>35.0</td>
      <td>34.0</td>
      <td>35.0</td>
      <td>38.0</td>
      <td>88.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>87.0</td>
      <td>37.0</td>
      <td>11.0</td>
      <td>67.0</td>
      <td>52.0</td>
      <td>66.0</td>
      <td>76.0</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>70.0</td>
      <td>71.0</td>
      <td>42.0</td>
      <td>94.0</td>
      <td>24.0</td>
      <td>76.0</td>
      <td>75.0</td>
      <td>63.0</td>
      <td>7.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10.0</td>
      <td>8.0</td>
      <td>45.0</td>
      <td>18.0</td>
      <td>39.0</td>
      <td>96.0</td>
      <td>63.0</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>46.0</td>
      <td>56.0</td>
      <td>4.0</td>
      <td>95.0</td>
      <td>24.0</td>
      <td>63.0</td>
      <td>38.0</td>
      <td>67.0</td>
      <td>53.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.0</td>
      <td>44.0</td>
      <td>74.0</td>
      <td>95.0</td>
      <td>94.0</td>
      <td>79.0</td>
      <td>19.0</td>
      <td>4.0</td>
      <td>27.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>30.0</td>
      <td>70.0</td>
      <td>77.0</td>
      <td>13.0</td>
      <td>26.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>17.0</td>
      <td>52.0</td>
      <td>44.0</td>
      <td>34.0</td>
      <td>53.0</td>
      <td>89.0</td>
      <td>19.0</td>
      <td>40.0</td>
      <td>14.0</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>53.0</td>
      <td>14.0</td>
      <td>71.0</td>
      <td>92.0</td>
      <td>32.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>58.0</td>
      <td>90.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>96.0</td>
      <td>98.0</td>
      <td>73.0</td>
      <td>61.0</td>
      <td>36.0</td>
      <td>70.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>83.0</td>
      <td>39.0</td>
      <td>8.0</td>
      <td>82.0</td>
      <td>25.0</td>
      <td>71.0</td>
      <td>0.0</td>
      <td>80.0</td>
      <td>71.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>16.0</td>
      <td>87.0</td>
      <td>49.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>27.0</td>
      <td>20.0</td>
      <td>57.0</td>
      <td>92.0</td>
      <td>47.0</td>
    </tr>
  </tbody>
</table><script type="text/javascript">
        jQuery("#table-2344fbd85f84944566e4").DataTable({"paging": true});
    </script>


Does it work when loading the notebook the first time? **No**

Does it always work? **Not always**

Why not using `$().DataTable`? **It does not work**

Is the styling of this whole blog worth the trouble? **The styling is terrible but it's worth the trouble**

What was the point again? **It works offline and does not involve third parties when browsing this post**
