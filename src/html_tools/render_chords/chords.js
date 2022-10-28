document.addEventListener("DOMContentLoaded", () => {

    // create web audio api context
    const audio_context = new window.AudioContext();
    const audio_sources = {};

    function play_notes(notes) {
        clean_sources();
        notes = notes.split(",").map(n => parseInt(n) + 12);
        console.log("playing notes", notes);

        let interval = 0;
        for (const note of notes) {
            const osc = audio_context.createOscillator();
            const osc2 = audio_context.createOscillator();

            const env = audio_context.createGain();
            const env2 = audio_context.createGain();

            osc.connect(env).connect(audio_context.destination);

            const freq = 440.0 * Math.pow(Math.pow(2., 1./12), note - 57);
            const time = audio_context.currentTime + interval;
            interval = interval * 1.3 + 1./30;
            osc.frequency.setValueAtTime(freq, 0);
            osc2.frequency.setValueAtTime(freq * 3., 0);
            env.gain.setValueAtTime(0, 0);
            env.gain.linearRampToValueAtTime(0, time);
            env.gain.linearRampToValueAtTime(.95 / 6, time + 1./150);
            env.gain.linearRampToValueAtTime(0, time + 2);

            env2.gain.setValueAtTime(100, 0)
            osc2.connect(env2).connect(osc.frequency);

            audio_sources[Object.keys(audio_sources).length] = {osc, env, time: audio_context.currentTime};
            osc.start();
            osc2.start();
        }
    }

    function clean_sources() {
        for (const key of Object.keys(audio_sources)) {
            const source = audio_sources[key];
            if (audio_context.currentTime - source.time > 3) {
                source.osc.disconnect();
                source.env.disconnect();
                delete audio_sources[key];
            }
        }
    }

    for (const elem of document.querySelectorAll('.chord [data-notes]')) {
        elem.addEventListener("click", event => {
            play_notes(elem.getAttribute("data-notes"));
        });
    }
});