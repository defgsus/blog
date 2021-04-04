---
layout: post
title: kali-set explorer
custom_js: 
  - kali/kali-gl.js
---

<!-- Actually i'd like to place them somewhere else.. -->
<script type="application/x-glsl" class="kali-exp-vert">

attribute vec3 aVertexPosition;
varying vec3 vVertexPosition;

void main() {
    vVertexPosition = aVertexPosition;
    gl_Position = vec4(aVertexPosition, 1.0);
}

</script>

<script type="application/x-glsl" class="kali-exp-frag">

#line 23
precision mediump float;
varying vec3 vVertexPosition;
uniform vec2 uResolution;
uniform vec4 uKaliParam;
uniform vec4 uKaliPosition;
uniform float uKaliScale;


// AntiAliasing 0=turn off, n=use n by n sub-pixels
#define AA {AA}
#define ITERATIONS {ITERATIONS}


vec3 kali(in vec2 p) {
    for (int i=0; i<ITERATIONS-1; ++i) {
        p = abs(p) / dot(p, p);
        p -= uKaliParam.xy;
    }
    p = abs(p) / dot(p, p);
    return vec3(p, 0);
}

vec3 kali_3d(in vec2 p) {
    vec3 v = vec3(p, 0.);
    for (int i=0; i<ITERATIONS-1; ++i) {
        v = abs(v) / dot(v, v);
        v -= uKaliParam.xyz;
    }
    v = abs(v) / dot(v, v);
    return v;
}

vec3 kali_4d(in vec2 p) {
    vec4 v = vec4(p, p);
    for (int i=0; i<ITERATIONS-1; ++i) {
        v = abs(v) / dot(v, v);
        v -= uKaliParam;
    }
    v = abs(v) / dot(v, v);
    return v.xyz;
}


vec3 frag_to_color(in vec2 fragCoord) {
    vec2 uv = (fragCoord - uResolution * .5) / uResolution.y * 2.;

    uv = uv * uKaliScale + uKaliPosition.xy;

    vec3 col = kali(uv);

    return col;
}

void main() {
    vec2 fragCoord = (vVertexPosition.xy * .5 + .5) * uResolution;

    #if AA <= 1
        vec3 col = frag_to_color(fragCoord);
    #else
        vec3 col = vec3(0);
        for (int y=0; y<AA; ++y) {
            for (int x=0; x<AA; ++x) {
                vec2 ofs = vec2(x, y) / float(AA);
                col += frag_to_color(fragCoord + ofs);
            }
        }
        col /= float(AA * AA);
    #endif

    gl_FragColor = vec4(col, 1.0);
}

</script>


Yeah, i know, *explorer* sounds so 90ies but that's what this post is about.
Exploring the amazing **kali set**.


<div id="kali-01" style="width: 512px; height: 512px;"></div>
<div id="kali-01-ctl"></div>

<script>
    render_kali("kali-01", "kali-01-ctl");
</script>

