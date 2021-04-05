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
uniform float uAmplitude;


// AntiAliasing 0=turn off, n=use n by n sub-pixels
#define AA {AA}
#define ITERATIONS {ITERATIONS}
#define DIMENSIONS {DIMENSIONS}
#define KALI kali_{ACCUMULATOR}
#define EIFFIE_MOD {EIFFIE_MOD}


vec4 kali_final(in vec4 p, in vec4 param) {
    for (int i=0; i<ITERATIONS-1; ++i) {
        p = abs(p) / dot(p, p);
        p -= param;
    }
    p = abs(p) / dot(p, p);
    return p;
}


vec4 kali_average(in vec4 p, in vec4 param) {
    vec4 acc = vec4(0);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc += p;
        p -= param;
    }
    return acc / float(ITERATIONS);
}


vec4 kali_min(in vec4 p, in vec4 param) {
    vec4 acc = vec4(1e8);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc = min(acc, p);
        p -= param;
    }
    return acc;
}


vec4 kali_max(in vec4 p, in vec4 param) {
    vec4 acc = vec4(0);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc = max(acc, p);
        p -= param;
    }
    return acc;
}


vec4 kali_distance_axis(in vec4 p, in vec4 param, in vec4 axis) {
    float min_dist = 1e8;
    float scale = 1.;
    for (int i=0; i<ITERATIONS; ++i) {
        scale /= dot(p, p);
        p = abs(p) / dot(p, p);
        float dist = dot(p, axis);
        #if EIFFIE_MOD == 1
            dist *= scale;
        #endif
        min_dist = min(dist, min_dist);
        p -= param;
    }
    return vec4(min_dist);
}

vec4 kali_distance_x(in vec4 p, in vec4 param) {
    return kali_distance_axis(p, param, vec4(1, 0, 0, 0));
}

vec4 kali_distance_y(in vec4 p, in vec4 param) {
    return kali_distance_axis(p, param, vec4(0, 1, 0, 0));
}

vec4 kali_distance_z(in vec4 p, in vec4 param) {
    return kali_distance_axis(p, param, vec4(0, 0, 1, 0));
}

vec4 kali_distance_w(in vec4 p, in vec4 param) {
    return kali_distance_axis(p, param, vec4(0, 0, 0, 1));
}


vec4 frag_to_color(in vec2 fragCoord) {
    vec4 uv = vec4((fragCoord - uResolution * .5) / uResolution.y * 2., 0., 0.);
    uv *= uKaliScale;
    
    #if DIMENSIONS <= 2
        vec4 col = KALI(
            (uv * uKaliScale + uKaliPosition) * vec4(1, 1, 0, 0),
            uKaliParam * vec4(1, 1, 0, 0)
        );
        col.zw = vec2(0, 1);
    #endif
    
    #if DIMENSIONS == 3
        vec4 col = KALI(
            (uv * uKaliScale + uKaliPosition) * vec4(1, 1, 1, 0),
            uKaliParam * vec4(1, 1, 1, 0)
        );
        col.w = 1.;
    #endif
    
    #if DIMENSIONS == 4
        vec4 col = KALI(
            (uv * uKaliScale + uKaliPosition),
            uKaliParam
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
    
    col = clamp(col * uAmplitude, 0., 1.);
    col = mix(vec4(0,0,0,1), vec4(col.xyz,1), col.a);

    gl_FragColor = col;
}

</script>



<div id="kali-01" style="width: 512px; height: 512px;"></div>
<div id="kali-01-ctl"></div>

<script>
    render_kali("kali-01", "kali-01-ctl");
</script>

