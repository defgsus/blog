"""
"""
import re
from pathlib import Path
import argparse
import sys
from typing import List, Tuple

import marko
from marko.md_renderer import MarkdownRenderer
from marko.html_renderer import HTMLRenderer
from marko.block import FencedCode
from marko.inline import RawText
import scss


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", type=str,
        help="Markdown filename",
    )
    parser.add_argument(
        "--html", type=bool, nargs="?", default=False, const=True,
        help="render as html"
    )
    return parser.parse_args(args)


class ChordRenderer(MarkdownRenderer):

    NOTES = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

    RE_TAG_AUDIO = re.compile(r"\{audio:([^}]+)}")

    AUDIO_TEMPLATE = """
<audio src="{filename}" controls="controls"></audio>
<a href="{filename}">{short_filename}</a>
""".strip().replace("\n", " ")

    def render_raw_text(self, element: RawText) -> str:
        text = super().render_raw_text(element)
        def _sub(match):
            filename = match.groups()[0]
            short_filename = Path(filename).name
            markup = self.AUDIO_TEMPLATE.format(filename=filename, short_filename=short_filename)
            return f"\n\n{markup}\n\n"

        return self.RE_TAG_AUDIO.sub(_sub, text)

    def render_fenced_code(self, element: FencedCode) -> str:
        language = element.lang
        if not element.children:
            return super().render_fenced_code(element)

        if language.startswith("chords-"):
            strings = language[7:].split("/")
            strings = self._convert_strings(strings)
            text: RawText = element.children[0]

            chords = []
            for line in text.children.splitlines():
                line = line.strip()
                if line:
                    chords.append(line)

            return self._render_chords(strings, chords)

        elif language.startswith("bars"):
            return self._render_bars(super().render_raw_text(element.children[0]))

    def _convert_strings(self, strings: List[str]) -> List[int]:
        root_note = self.NOTES.index(strings[0][0].lower())
        if len(strings[0]) > 1:
            root_note += 12 * int(strings[0][1:])
        result = [root_note]
        for s in strings[1:]:
            note = root_note + int(s)
            result.append(note)
            root_note = note
        while any(n < 3*12 for n in result):
            result = [n + 12 for n in result]
        return result

    def _render_chords(self, strings: List[int], chords: List[str]) -> str:
        markup = ""
        for chord in chords:
            if ":" in chord:
                title, chord = [i.strip() for i in chord.split(":")]
            else:
                title = None

            fret_matrix, all_notes = self._get_fret_data(strings, chord)

            markup += f'<div class="chord strings-{len(strings)}">'
            if title:
                notes = ",".join(str(n) for n in all_notes)
                markup += f'<div class="title" data-notes="{notes}">{title}</div>'
            for fret_idx, fret_row in enumerate(fret_matrix):
                markup += f'<div class="fret fret-{fret_idx}">'

                note_infos = []
                notes_on_fret = []
                note_names_on_fret = []
                for string_idx, f in enumerate(fret_row):
                    data_attr = ""
                    if f["play_notes"]:
                        data_attr = ",".join(str(n) for n in f["play_notes"])
                        data_attr = f' data-notes="{data_attr}"'

                    field_class = "" if fret_idx == 0 else " inside"
                    note_infos.append(
                        f'<div class="note-field{field_class}"{data_attr} title="{f["title"]}"></div>'
                    )

                    if f["pressed"]:
                        length = "" if f["press_length"] < 2 else f' length-{f["press_length"]}'
                        notes_on_fret.append(
                            f'<div class="note filled string-{string_idx}{length}"{data_attr} title="{f["title"]}"></div>'
                        )
                    if f["show_name"]:
                        note_names_on_fret.append(
                            f'<div class="note string-{string_idx}"{data_attr} title="{f["title"]}"><div class="note-name">{f["note_name"]}</div></div>'
                        )
                    if f["closed"]:
                        note_names_on_fret.append(
                            f'<div class="note string-{string_idx}"{data_attr} title="{f["title"]}"><div class="crossed">X</div></div>'
                        )

                markup += f'<div class="fret-container">{"".join(note_infos)}</div>'

                if fret_idx != 0:
                    markup += f'<div class="fret-container pass-events">'
                    for string in strings:
                        markup += f'<div class="string" title="{string}">'
                        markup += '</div>'
                    markup += '</div>'  # .fret-container

                if notes_on_fret:
                    markup += f'<div class="fret-container">{"".join(notes_on_fret)}</div>'

                if note_names_on_fret:
                    markup += f'<div class="fret-container">{"".join(note_names_on_fret)}</div>'

                markup += '</div>'  # .fret

            markup += '</div>'

        return f'<div class="chord-row">{markup}</div>'

    def _get_fret_data(self, strings: List[int], chord: str) -> Tuple[
        List[List[dict]],
        List[int],
    ]:
        chord = chord.split("/")
        while len(chord) < len(strings):
            chord.append("")

        all_played_notes = set()

        result = []
        for string_idx, fret_str in enumerate(chord):
            if fret_str:
                try:
                    fret_idx = int(fret_str)
                    press_length = 1
                except ValueError:
                    fret_idx, press_length = [int(i) for i in fret_str.split("-")]

                while len(result) <= max(1, fret_idx):
                    result.append([
                        {
                            "note": s + len(result),
                            "pressed": False,
                            "press_length": 1,
                            "show_name": False,
                            "closed": False,
                            "play_notes": [],
                        } for s in strings
                    ])

                result_note = result[fret_idx][string_idx]
                result_note["pressed"] = True
                result_note["press_length"] = press_length

                for i in range(press_length):
                    if string_idx + i < len(strings):
                        result[fret_idx][string_idx + i]["show_name"] = True
                        result_note["play_notes"].append(result[fret_idx][string_idx + i]["note"])
                        all_played_notes.add(result[fret_idx][string_idx + i]["note"])

        # mark silent strings
        for string_idx, f in enumerate(result[0]):
            is_played = False
            for row in result:
                if row[string_idx]["show_name"]:
                    is_played = True
                    break
            if not is_played:
                f["closed"] = True

        all_played_notes = sorted(all_played_notes)
        for fret_idx, row in enumerate(result):
            for f in row:
                if not f["play_notes"]:
                    f["play_notes"] = all_played_notes

                note_name = self._note_name(f['note'])
                f["note_name"] = note_name
                if not f["pressed"]:
                    f["title"] = f"Note {note_name}"
                else:
                    if f["press_length"] > 1:
                        note_names = [
                            self._note_name(f["note"] + n)
                            for n in range(f["press_length"])
                        ]
                        f["title"] = f"Notes {' '.join(note_names)} (held on {self._fret_name(fret_idx)})"
                    else:
                        f["title"] = f"Note {note_name} (held on {self._fret_name(fret_idx)})"

        return result, all_played_notes

    def _render_bars(self, text: str) -> str:
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if all(c.isnumeric() or c.isspace() or c == "." for c in line):
                pass
            else:
                lines[idx] = f'<b>{line}</b>'
        lines = "\n".join(lines)
        return f'<pre class="bars">{lines}</pre>'

    def _note_name(self, n: int) -> str:
        return f"{self.NOTES[n % 12].upper()}"#{n // 12}"

    def _fret_name(self, n: int) -> str:
        if n == 0:
            return "open string"
        elif n == 1:
            return "1st fret"
        elif n == 2:
            return "2nd fret"
        elif n == 3:
            return "3rd fret"
        else:
            return f"{n}th fret"


def render_markdown(input_filename: str) -> str:
    doc = marko.parse(Path(input_filename).read_text())

    compiler = scss.compiler.Compiler(
        output_style="compressed"
    )
    css = compiler.compile(Path(__file__).resolve().parent / "chords.scss")
    (Path(__file__).resolve().parent / "chords.css").write_text(css)

    with ChordRenderer() as renderer:
        new_markdown = renderer.render(doc)
        return new_markdown


def main(args):
    options = parse_args(args)
    md = render_markdown(input_filename=options.filename)

    if not options.html:
        result = md
    else:
        with HTMLRenderer() as renderer:
            doc = marko.parse(md)
            markup = renderer.render(doc)
            result = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Chords</title>
    <meta http-equiv="Content-Security-Policy" content="script-src 'self' 'unsafe-inline' 'unsafe-eval'">
    <link rel="stylesheet" href="chords.css">
    <script src="chords.js"></script>
</head>
<body>
{markup}
</body>
</html>
"""

    print(result)


if __name__ == "__main__":
    main(sys.argv[1:])