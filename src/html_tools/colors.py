from typing import Optional, List


def get_color_map(name: str, max_length: int = 50) -> Optional[List[str]]:
    """
    Picks one of plotly's or matplotlib's color maps

    https://matplotlib.org/stable/tutorials/colors/colormaps.html

    :param name: str, case-sensitive name
    :return: List of html color strings or None
    """
    try:
        import matplotlib.colors
        import matplotlib.pyplot
        try:
            cmap = matplotlib.pyplot.get_cmap(name)
            if hasattr(cmap, "colors"):
                return [matplotlib.colors.to_hex(c) for c in cmap.colors]
            else:
                return [
                    matplotlib.colors.to_hex(cmap(i / (max_length - 1)))
                    for i in range(max_length)
                ]
        except ValueError:
            pass

    except ImportError:
        pass

    try:
        import plotly.express as px
        if name in px.colors.PLOTLY_SCALES:
            return [c[1] for c in px.colors.PLOTLY_SCALES[name]]

    except ImportError:
        pass
