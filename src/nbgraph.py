import graphviz


class Graph:

    def __init__(
            self,
            digraph: bool = True,
            engine: str = "neato",
            max_size: str = "10,10",
            **kwargs,
    ):
        if digraph:
            self.g = graphviz.Digraph(engine=engine, **kwargs)
        else:
            self.g = graphviz.Graph(engine=engine, **kwargs)
        self.g.attr("graph", size=max_size)
        # parameters applied to nodes and edges
        self.default_kwargs = {
            "fontname": "Helvetica"
        }

    def node(self, id: str, label: str = None, **kwargs):
        merged_kwargs = self.default_kwargs.copy()
        merged_kwargs.update(kwargs)
        merged_kwargs.setdefault("style", "filled")
        self.g.node(id, label or id, **merged_kwargs)

    def edge(self, *args, **kwargs):
        merged_kwargs = self.default_kwargs.copy()
        merged_kwargs.update(kwargs)
        merged_kwargs.setdefault("fontsize", "12")
        self.g.edge(*args, **merged_kwargs)

    def display(self):
        from IPython.display import display, HTML
        svg = self.g._repr_svg_()
        svg = svg[svg.index('DTD/svg11.dtd">')+15:]

        # TODO: this does not work when exporting to markdown/jekyll
        #html = f"""
        #    <div style="justify-content: center; display: flex">{svg}</div>
        #"""
        display(HTML(svg))
