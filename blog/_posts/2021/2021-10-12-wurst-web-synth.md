---
layout: post
title: Wurst - a web synthesizer
custom_css: wurst1/index.css
custom_js: wurst1/index.js
---

A live demo of [wurst](https://github.com/defgsus/wurst), the 
**W**eb-**U**tilized **R**eact **S**ynthesizer **T**est.

<div id="app"></div>
<br>

*What can i do?* 

Not much at the moment ;) The sequences have a **target** selection, 
this is where the value is directed to. The sequence buttons
are either **0** or **1** and then multiplied by the **amp** value.

A **gate** starts an attack/decay envelope. Everything else is **added**
on top of the modulated value.

The sequences can target **anything** of a voice generator, the BPM and beat
settings or other sequences' values.

It's all fairly untested and just a weekend project after all. 

If you like a song, you can type in a name and press the **SAVE** button. 
It will be stored in the browser's 
[sessionStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage)
and also printed to the console output.

*What is this, anyway?*

My first contact with the browser's 
[Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API).
I used [React](https://reactjs.org/) for the user interface and, apparently, 
[setTimeout](https://developer.mozilla.org/en-US/docs/Web/API/setTimeout) to 
step the sequencer which is funny because of it's timing irregularity but 
probably not the way i'll do it next time. 

The UI elements also need some more work although i certainly spent more time for 
[CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) 
and [DOM event handling](https://developer.mozilla.org/en-US/docs/Web/Events)
than the actual audio stuff.
