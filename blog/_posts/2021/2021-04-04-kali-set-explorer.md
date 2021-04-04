---
layout: post
title: kali-set explorer
custom_js: 
  - kali/kali-gl.js
custom_css: 
  - kali/kali-gl.css
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
#define DIMENSIONS {DIMENSIONS}
#define KALI kali_final

vec4 kali_final(in vec2 p) {
    for (int i=0; i<ITERATIONS-1; ++i) {
        p = abs(p) / dot(p, p);
        p -= uKaliParam.xy;
    }
    p = abs(p) / dot(p, p);
    return vec4(p, 0, 1);
}

vec4 kali_final(in vec3 p) {
    for (int i=0; i<ITERATIONS-1; ++i) {
        p = abs(p) / dot(p, p);
        p -= uKaliParam.xyz;
    }
    p = abs(p) / dot(p, p);
    return vec4(p, 1);
}

vec4 kali_final(in vec4 p) {
    for (int i=0; i<ITERATIONS-1; ++i) {
        p = abs(p) / dot(p, p);
        p -= uKaliParam;
    }
    p = abs(p) / dot(p, p);
    return p;
}


vec4 frag_to_color(in vec2 fragCoord) {
    vec2 uv = (fragCoord - uResolution * .5) / uResolution.y * 2.;
    uv *= uKaliScale;
    
    #if DIMENSIONS <= 2
        vec4 col = KALI(
            uv * uKaliScale + uKaliPosition.xy
        );
    #endif
    
    #if DIMENSIONS == 3
        vec4 col = KALI(
            vec3(uv, 0) * uKaliScale + uKaliPosition.xyz
        );
    #endif
    
    #if DIMENSIONS == 4
        vec4 col = KALI(
            vec4(uv, 0, 0) * uKaliScale + uKaliPosition
        );
    #endif
    
    return col;
}

void main() {
    vec2 fragCoord = (vVertexPosition.xy * .5 + .5) * uResolution;

    #if AA <= 1
        vec4 col = frag_to_color(fragCoord);
    #else
        vec4 col = vec4(0);
        for (int y=0; y<AA; ++y) {
            for (int x=0; x<AA; ++x) {
                vec2 ofs = vec2(x, y) / float(AA);
                col += frag_to_color(fragCoord + ofs);
            }
        }
        col /= float(AA * AA);
    #endif
    
    col = mix(vec4(0,0,0,1), vec4(col.xyz,1), col.a);

    gl_FragColor = col;
}

</script>


Yeah, i know, *explorer* sounds so 90ies but that's what this post is about.
Exploring the amazing **kali set**.


<div id="kali-01" style="width: 512px; height: 512px;"></div>
<div id="kali-01-ctl"></div>

<script>
    render_kali("kali-01", "kali-01-ctl");
</script>

