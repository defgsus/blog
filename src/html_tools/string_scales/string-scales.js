document.addEventListener("DOMContentLoaded", () => {

    const ROOT_OCTAVE = 4;
    const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const INTERVALS = [
        {short: "r", name: "root"},
        {short: "♭2", name: "minor second"},
        {short: "2", name: "major second"},
        {short: "♯2", name: "augmented second"},
        {short: "♭3", name: "minor third"},
        {short: "3", name: "major third"},
        {short: "4", name: "perfect fourth"},
        {short: "♯4", name: "augmented fourth"},
        {short: "♭5", name: "diminished fifth"},
        {short: "5", name: "perfect fifth"},
        {short: "♯5", name: "augmented fifth"},
        {short: "♭6", name: "minor sixth"},
        {short: "6", name: "major sixth"},
        {short: "7", name: "minor seventh"},
        {short: "M7", name: "major seventh"},
    ];
    const TUNINGS = [
        {"name": "Ukulele A543", strings: "A2 5 4 3"},
        {"name": "Ukulele G-745", strings: "G3 -7 4 5"},
        {"name": "Guitar (6-string)", strings: "E3 A3 D4 G4 B4 E5"},
        {"name": "Violin/Mandolin", strings: "G2 7 7 7"},
        {"name": "Custom...", strings: ""},
    ];
    const CHORDS = [
        {name: "major", short: "", intervals:       [0, 4, 7], simple: true},
        {name: "6", short: "6", intervals:          [0, 4, 7, 9]},
        {name: "6/9", short: "6/9", intervals:      [0, 4, 7, 9, 14]},
        {name: "7", short: "7", intervals:          [0, 4, 7, 10]},
        {name: "7sus4", short: "7sus4", intervals:  [0, 5, 7, 10]},
        {name: "7/9", short: "7/9", intervals:      [0, 4, 7, 10, 14]},
        {name: "maj7", short: "maj7", intervals:    [0, 4, 7, 11]},
        {name: "maj9", short: "maj9", intervals:    [0, 4, 7, 11, 14]},
        {name: "add9", short: "add9", intervals:    [0, 4, 7, 14]},
        {name: "sus4", short: "sus4", intervals:    [0, 5, 7]},
        {name: "aug", short: "+", intervals:        [0, 4, 8]},
        {name: "minor", short: "m", intervals:      [0, 3, 7], simple: true},
        {name: "m6", short: "m6", intervals:        [0, 3, 7, 9]},
        {name: "m7", short: "m7", intervals:        [0, 3, 7, 10]},
        {name: "m9", short: "m9", intervals:        [0, 3, 7, 10, 14]},
        {name: "madd9", short: "madd9", intervals:  [0, 3, 7, 14]},
        {name: "dim7", short: "o7", intervals:      [0, 3, 6, 9]},
    ];
    const SCALES = [
        ...CHORDS.map(chord => ({
            "name": `Chord: ${chord.name}`,
            "intervals": chord.intervals,
        })),
        {
            "name": "Acoustic scale",
            "intervals": [0, 2, 4, 6, 7, 9, 10],
        },
        {
            "name": "Aeolian mode or natural minor scale",
            "intervals": [0, 2, 3, 5, 7, 8, 10],
        },
        {
            "name": "Algerian scale",
            "intervals": [0, 2, 3, 6, 7, 9, 11, 12, 14, 15, 17],
        },
        {
            "name": "Altered scale or Super Locrian scale",
            "intervals": [0, 1, 3, 4, 6, 8, 10],
        },
        {
            "name": "Augmented scale",
            "intervals": [0, 3, 4, 7, 8, 11],
        },
        {
            "name": "Bebop dominant scale",
            "intervals": [0, 2, 4, 5, 7, 9, 10, 11],
        },
        {
            "name": "Blues scale",
            "intervals": [0, 3, 5, 6, 7, 10],
        },
        {
            "name": "Chromatic scale",
            "intervals": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
            "name": "Dorian mode",
            "intervals": [0, 2, 3, 5, 7, 9, 10],
        },
        {
            "name": "Double harmonic scale",
            "intervals": [0, 1, 4, 5, 7, 8, 11],
        },
        {
            "name": "Enigmatic scale",
            "intervals": [0, 1, 4, 6, 8, 10, 11],
        },
        {
            "name": "Flamenco mode",
            "intervals": [0, 1, 4, 5, 7, 8, 11],
        },
        {
            "name": '"Gypsy" scale',
            "intervals": [0, 2, 3, 6, 7, 8, 10],
        },
        {
            "name": "Half diminished scale",
            "intervals": [0, 2, 3, 5, 6, 8, 10],
        },
        {
            "name": "Harmonic major scale",
            "intervals": [0, 2, 4, 5, 7, 8, 11],
        },
        {
            "name": "Harmonic minor scale",
            "intervals": [0, 2, 3, 5, 7, 8, 11],
        },
        {
            "name": "Hirajoshi scale",
            "intervals": [0, 4, 6, 7, 11],
        },
        {
            "name": 'Hungarian "Gypsy" scale/Hungarian minor scale',
            "intervals": [0, 2, 3, 6, 7, 8, 11],
        },
        {
            "name": "Hungarian major scale",
            "intervals": [0, 3, 4, 6, 7, 9, 10],
        },
        {
            "name": "In scale",
            "intervals": [0, 1, 5, 7, 8],
        },
        {
            "name": "Insen scale",
            "intervals": [0, 1, 5, 7, 10],
        },
        {
            "name": "Ionian mode or major scale",
            "intervals": [0, 2, 4, 5, 7, 9, 11],
        },
        {
            "name": "Istrian scale",
            "intervals": [0, 1, 3, 4, 6, 7],
        },
        {
            "name": "Iwato scale",
            "intervals": [0, 1, 5, 6, 10],
        },
        {
            "name": "Locrian mode",
            "intervals": [0, 1, 3, 5, 6, 8, 10],
        },
        {
            "name": "Lydian augmented scale",
            "intervals": [0, 2, 4, 6, 8, 9, 11],
        },
        {
            "name": "Lydian mode",
            "intervals": [0, 2, 4, 6, 7, 9, 11],
        },
        {
            "name": "Major Locrian scale",
            "intervals": [0, 2, 4, 5, 6, 8, 10],
        },
        {
            "name": "Major pentatonic scale",
            "intervals": [0, 2, 4, 7, 9],
        },
        {
            "name": "Melodic minor scale",
            "intervals": [0, 2, 3, 5, 7, 9, 11],
        },
        {
            "name": "Melodic minor scale (ascending)",
            "intervals": [0, 2, 3, 5, 7, 9, 11],
        },
        {
            "name": "Minor pentatonic scale",
            "intervals": [0, 3, 5, 7, 10],
        },
        {
            "name": "Mixolydian mode or Adonai malakh mode",
            "intervals": [0, 2, 4, 5, 7, 9, 10],
        },
        {
            "name": "Neapolitan major scale",
            "intervals": [0, 1, 3, 5, 7, 9, 11],
        },
        {
            "name": "Neapolitan minor scale",
            "intervals": [0, 1, 3, 5, 7, 8, 11],
        },
        {
            "name": "Octatonic scale",
            "intervals": [0, 2, 3, 5, 6, 8, 9, 11],
        },
        {
            "name": "Persian scale",
            "intervals": [0, 1, 4, 5, 6, 8, 11],
        },
        {
            "name": "Phrygian dominant scale",
            "intervals": [0, 1, 4, 5, 7, 8, 10],
        },
        {
            "name": "Phrygian mode",
            "intervals": [0, 1, 3, 5, 7, 8, 10],
        },
        {
            "name": "Prometheus scale",
            "intervals": [0, 2, 4, 6, 9, 10],
        },
        {
            "name": "Scale of harmonics",
            "intervals": [0, 3, 4, 5, 7, 9],
        },
        {
            "name": "Tritone scale",
            "intervals": [0, 1, 4, 6, 7, 10],
        },
        {
            "name": "Two-semitone tritone scale",
            "intervals": [0, 1, 2, 6, 7, 8],
        },
        {
            "name": "Ukrainian Dorian scale",
            "intervals": [0, 2, 3, 6, 7, 9, 10],
        },
        {
            "name": "Whole tone scale",
            "intervals": [0, 2, 4, 6, 8, 10],
        },
        {
            "name": "Yo scale",
            "intervals": [0, 3, 5, 7, 10],
        },
    ]

    const settings = {
        tuning: TUNINGS[0],
        scale: SCALES[0],
        root_note: 0,
        root_note_normal: ROOT_OCTAVE * 12,
        strings: [],
        num_frets: 12,
        show_octave: false,
        is_note_highlighted: n => false,
        is_fret_dot: f => (f % 12) === 5 || (f % 12) === 7 || (f % 12) === 10,
    };

    const $custom_tuning = document.querySelector("#custom-tuning");
    const $custom_scale = document.querySelector("#custom-scale");

    function name_to_note(name) {
        const num_index = Array.from(name).findIndex(c => parseInt(c) >= 0);
        if (num_index < 0)
            return Math.max(0, NOTE_NAMES.indexOf(name.toUpperCase()));
        const note_str = name.substr(0, num_index);
        const octave_str = name.substr(num_index);
        return parseInt(octave_str) * 12 + Math.max(0, NOTE_NAMES.indexOf(note_str.toUpperCase()));
    }
    window.name_to_note = name_to_note;
    function parse_intervals(string) {
        return string.split(" ").map(s => parseInt(s)).filter(i => i === i);
    }

    function chords_in_scale(scale, scale_root, simple_only) {
        const intervals = scale.intervals.map(n => (n + scale_root) % 12);
        const chords = [];
        for (const chord of CHORDS) {
            for (let root=0; root<12; ++root) {
                let found = true;
                for (const chord_note of chord.intervals) {
                    found &= intervals.indexOf((root + chord_note) % 12) >= 0;
                    if (!found)
                        break;
                }
                if (found && (!simple_only || chord.simple))
                    chords.push({root, chord});
            }
        }
        chords.sort((a, b) => a.root < b.root ? -1 : a.root > b.root ? 1 : 0);
        return chords;
    }

    function note_name(note, pad_length, pad_char) {
        const n = note % 12, o = Math.floor(note / 12);
        let name = NOTE_NAMES[n];
        if (settings.show_octave && o > 0)
            name = `${name}${o}`;
        if (pad_length) {
            if (!pad_char)
                pad_char = " ";
            name = name.padEnd(pad_length, pad_char);
        }
        return name;
    }
    window.note_to_name = note_name;
    function note_elem(note, pad_length, pad_char) {
        const name = note_name(note);
        const elem = `<span class="note" data-notes="${note}">${name}</span>`;
        if (pad_length) {
            return elem.padEnd(elem.length - name.length + pad_length, pad_char);
        }
        return elem;
    }
    function note_fret_span(note, fret) {
        const is_hl = settings.is_note_highlighted(note);
        const note_class = is_hl ? "note hl" : "note";
        const space = fret === 0 ? " " : "-";
        const name = note_name(note);
        const spacing = space.padEnd(4 - name.length, space);
        return `${space}<span class="${note_class}" data-notes="${note}">${name}</span>${spacing}|`;
    }
    function chord_elem(chord, root) {
        let notes = chord.intervals.map(c => (c + root));
        while (notes.findIndex(n => n < ROOT_OCTAVE * 12) >= 0)
            notes = notes.map(n => n + 12);
        const notes_str = notes.map(n => `${n}`).join(",");
        const title = `${note_name(root || 0)} ${chord.name}: ${notes.map(n => note_name(n)).join(" ")}`;
        return `<span class="chord" data-notes="${notes_str}" title="${title}">${note_name(root || 0)}${chord.short}</span>`;
    }
    function interval_elem(interval, pad_length, pad_char) {
        const inter = INTERVALS[interval];
        const elem =`<span class="interval" title="${inter.name}">${inter.short}</span>`;
        if (pad_length) {
            return elem.padEnd(elem.length - inter.short.length + pad_length, pad_char);
        }
        return elem;
    }

    function render_menu() {
        let markup = ``;
        for (const i in SCALES) {
            const scale = SCALES[i];
            markup += `<option value="${i}">${scale.name}</option>`;
        }
        let $select = document.querySelector("select#scale-select");
        $select.innerHTML = markup;
        $select.onchange = (e) => set_scale(e.target.value);

        markup = ``;
        for (const n of NOTE_NAMES) {
            markup += `<option value="${n}">${n}</option>`;
        }
        $select = document.querySelector("select#note-select");
        $select.innerHTML = markup;
        $select.onchange = (e) => set_root_note(e.target.value);

        markup = ``;
        for (const i in TUNINGS) {
            const tuning = TUNINGS[i];
            markup += `<option value="${i}">${tuning.name}</option>`;
        }
        $select = document.querySelector("select#tuning-select");
        $select.innerHTML = markup;
        $select.onchange = (e) => set_tuning(e.target.value);

        document.querySelector("#octave-switch").onchange = (e) => {
            settings.show_octave = !!e.target.checked;
            update_from_settings();
        };
        $custom_tuning.oninput = (e) => {
            TUNINGS[TUNINGS.length-1].strings = $custom_tuning.value;
            update_from_settings();
        };
        $custom_scale.oninput = (e) => {
            SCALES[SCALES.length-1].intervals = parse_intervals(e.target.value);
            update_from_settings();
        };
    }

    function render_strings() {
        let markup = `<span class="text">`;

        // --- intervals ---

        markup += `intervals:`.padEnd(14);
        for (const i of settings.scale.intervals) {
            markup += `${i}`.padEnd(4);
        }
        markup += `\n` + ``.padEnd(14);
        for (const i of settings.scale.intervals) {
            markup += interval_elem(i, 4);
        }
        markup += `\n` + `notes on ${note_name(settings.root_note)}:`.padEnd(14);
        for (const i of settings.scale.intervals) {
            markup += note_elem(settings.root_note_normal + i, 4);
        }
        const note_str = settings.scale.intervals.map(n => `${n+settings.root_note_normal}`).join(",");
        markup += ` <span data-notes="${note_str}">(play all)</span>\n`;

        markup += `</span>\n\n`; // .text

        // --- strings ---

        for (let i=settings.strings.length-1; i>=0; --i) {
            const string_note = settings.strings[i];
            for (let fret = 0; fret <= settings.num_frets; ++fret) {
                if (fret === 1)
                    markup += `<span class="neck">`;
                const note = string_note + fret;
                markup += note_fret_span(note, fret);
            }
            if (settings.num_frets > 0)
                markup += `</span>`; // .neck
            markup += "\n";
        }
        for (let fret = 0; fret <= settings.num_frets; ++fret) {
            markup += settings.is_fret_dot(fret) ? "  *   " : "      ";
        }
        markup += `\n\n<span class="text wrap">`;

        // --- contained chords ---

        const simple_chords = chords_in_scale(settings.scale, settings.root_note, true);
        const all_chords = chords_in_scale(settings.scale, settings.root_note);

        if (simple_chords.length && all_chords.length !== simple_chords.length) {
            markup += `contained chords (simple): `;
            markup += simple_chords.map(c => chord_elem(c.chord, c.root)).join(", ");

            markup += `\n\nall contained chords: `;
            markup += all_chords.map(c => chord_elem(c.chord, c.root)).join(", ");
        } else {
            markup += `contained chords: `;
            markup += all_chords.map(c => chord_elem(c.chord, c.root)).join(", ");
        }
        markup += `</span>`; // .text

        document.querySelector(".string-scales pre").innerHTML = markup;

        _render_note_svg();

        for (const elem of document.querySelectorAll(`[data-notes]`)) {
            elem.onclick = event => {
                play_notes(elem.getAttribute("data-notes"));
            };
        }
    }

    function _render_note_svg() {
        const radius = 100;
        const fs = 10; // font-size
        const notes = settings.scale.intervals.map(n => (n + settings.root_note) % NOTE_NAMES.length);

        const _circle_pos = (note) => {
            const t = note / NOTE_NAMES.length * Math.PI * 2;
            const fx = Math.sin(t);
            const fy = -Math.cos(t);
            return [fx, fy];
        }

        let markup = `<circle cx="0" cy="0" r="${radius}" fill="none" stroke="#aaa"/>`;

        for (let note=0; note<NOTE_NAMES.length; ++note) {
            const note_name = NOTE_NAMES[note];
            const play_note = note + 12 * 4;
            const [fx, fy] = _circle_pos(note);
            const color = note === settings.root_note
                ? "#2a2"
                : notes.indexOf(note) >= 0 ? "black" : "#aaa";
            const r = note === settings.root_note ? 7 : 5;
            markup += `<circle cx="${fx * radius}" cy="${fy * radius}" r="${r}" fill="${color}" data-notes="${play_note}"/>`;
            markup += `<text x="${fx * (radius+fs*2) - fs*(note_name.length-.5)}" y="${fy * (radius+fs*2) + fs}" data-notes="${play_note}">${note_name}</text>`;
        }

        for (let idx=0; idx<notes.length; ++idx) {
            const [x1, y1] = _circle_pos(notes[idx]);
            const [x2, y2] = _circle_pos(notes[(idx + 1) % notes.length]);
            markup += `<line x1="${x1*radius}" y1="${y1*radius}" x2="${x2*radius}" y2="${y2*radius}" stroke="black"/>`;
        }

        const $elem = document.querySelector(".string-scales #note-svg");
        $elem.innerHTML = markup;
        const r = radius + fs * 4;
        $elem.setAttribute("viewBox", `${-r} ${-r} ${r*2} ${r*2}`)
    }

    function set_tuning(tuning_id) {
        settings.tuning = TUNINGS[parseInt(tuning_id)];
        update_from_settings();
    }
    function set_scale(scale_id) {
        settings.scale = SCALES[parseInt(scale_id)];
        update_from_settings();
    }
    function set_root_note(note_str) {
        settings.root_note = NOTE_NAMES.indexOf(note_str);
        settings.root_note_normal = settings.root_note;
        while (settings.root_note_normal < ROOT_OCTAVE * 12)
            settings.root_note_normal += 12;
        update_from_settings();
    }

    function update_from_settings() {
        const tuning_str = settings.tuning.strings.split(" ").filter(s => s.length);
        settings.strings = [];
        for (const s of tuning_str) {
            const prev_note = settings.strings[settings.strings.length-1] || ROOT_OCTAVE;
            const num = parseInt(s);
            if (num !== num) {
                settings.strings.push(name_to_note(s));
            } else {
                settings.strings.push(prev_note + num);
            }
        }
        //console.log("string notes", settings.strings);
        settings.is_note_highlighted = (note) => {
            const note12 = note % 12;
            for (const i of settings.scale.intervals) {
                if (note12 === (i + settings.root_note) % 12)
                    return true;
            }
            return false;
        };
        render_strings();
    }

    render_menu();
    update_from_settings();

    // ---------------------------------- AUDIO ----------------------------------

    const audio_context = new window.AudioContext();
    const audio_sources = {};

    function play_notes(notes) {
        clean_sources();
        if (typeof notes === "string")
            notes = notes.split(",").map(n => parseInt(n) + 12);

        let interval = 0;
        for (const note of notes) {
            const osc = audio_context.createOscillator();
            const osc2 = audio_context.createOscillator();

            const env = audio_context.createGain();
            const env2 = audio_context.createGain();

            osc.connect(env).connect(audio_context.destination);

            const freq = 440.0 * Math.pow(Math.pow(2., 1./12), note - 57);
            const time = audio_context.currentTime + interval;
            interval = interval * 1.05 + 1./6;
            osc.frequency.setValueAtTime(freq, 0);
            env.gain.setValueAtTime(0, 0);
            env.gain.linearRampToValueAtTime(0, time);
            env.gain.linearRampToValueAtTime(.95 / 6, time + 1./150);
            env.gain.linearRampToValueAtTime(0, time + 1.3);

            osc2.frequency.setValueAtTime(freq * .125, 0);
            env2.gain.setValueAtTime(20, 0)
            osc2.connect(env2).connect(osc.frequency);

            audio_sources[Object.keys(audio_sources).length] = {
                osc, env,
                time: audio_context.currentTime,
                stop: () => {
                    osc.stop();
                    osc2.stop();
                    osc.disconnect();
                    osc2.disconnect();
                    env.disconnect();
                },
            };
            osc.start();
            osc2.start();
        }

        console.log("playing notes", notes, "active voices:", Object.keys(audio_sources).length);
    }

    function clean_sources() {
        for (const key of Object.keys(audio_sources)) {
            const source = audio_sources[key];
            if (audio_context.currentTime - source.time > 3) {
                source.stop();
                delete audio_sources[key];
            }
        }
    }

});