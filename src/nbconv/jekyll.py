import os
import datetime
import re

from traitlets.config import Config
from nbconvert.exporters.markdown import MarkdownExporter
from nbformat.notebooknode import NotebookNode


class JekyllExporter(MarkdownExporter):

    RE_HIDE = re.compile(r"^# hide[^\-]")
    RE_HIDE_CODE = re.compile(r"^# hide-code")

    ENABLE = {
        "plotly": {
            "custom_js": ["require-stub", "plotly.min"],
        }
    }

    @property
    def template_paths(self):
        return super().template_paths + [
            os.path.join(os.path.dirname(__file__))
        ]

    def _template_file_default(self):
        return 'jekyll_template.j2'

    def context_variables(self, nb: NotebookNode):
        context = {
            "meta": {
                "layout": "post",
                "date": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            },
        }
        if nb.get("cells") and nb["cells"][0].get("source"):
            keywords = ["layout", "date", "categories", "title", "enable"]
            for line in nb["cells"][0]["source"].splitlines():
                if ":" in line:
                    key, value = line[:line.index(":")].strip(), line[line.index(":")+1:].strip()
                    if key in keywords:
                        context["meta"][key] = value
                    if key == "enable":
                        for en_key in self.ENABLE:
                            if en_key in value:
                                context["meta"].update(self.ENABLE[en_key])

        for key, value in context["meta"].items():
            if isinstance(value, (list, tuple)):
                context["meta"][key] = "\n" + "\n".join(f"  - {v}" for v in value)

        return context

    def from_notebook_node(self, nb: NotebookNode, resources=None, my_var=None, **kw):
        self.environment.globals.update(self.context_variables(nb))
        self.process_notebook(nb)
        return super().from_notebook_node(nb, resources=resources, **kw)

    @property
    def default_config(self):
        c = Config()
        c.merge(super().default_config)
        return c

    def process_notebook(self, nb: NotebookNode):
        cells = []
        for cell in nb["cells"]:
            if cell.get("source"):
                if self.RE_HIDE.findall(cell["source"]):
                    continue

                cell["source"] = self.process_cell_source(cell, cell["source"])

            if cell.get("outputs"):
                for i, output in enumerate(cell["outputs"]):
                    if output.get("data"):
                        for key, value in output["data"].items():
                            if isinstance(value, str):
                                output["data"][key] = self.process_cell_output(output, key, value)

            cells.append(cell)

        nb["cells"] = cells

    def process_cell_source(self, cell: dict, text: str) -> str:
        if self.RE_HIDE_CODE.findall(text):
            return ""

        return text

    def process_cell_output(self, output: dict, key: str, text: str) -> str:
        if "MutationObserver" in text:
            try:
                # plotly adds some notebook specific javascript which we do not need
                # and which actually breaks the Liquid template engine in Jekyll
                idx = text.index("var gd = document.getElementById(")
                idx2 = text.index("x.observe(outputEl, {childList: true});\n}}")
                text = text[:idx] + text[idx2+43:]
            except IndexError:
                pass

        return text