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

#line 25
precision mediump float;
varying vec3 vVertexPosition;
uniform vec2 uResolution;
uniform vec4 uKaliParam;
uniform vec4 uKaliPosition;
uniform float uKaliScale;
uniform float uAmplitude;
uniform float uObjectDistance;
uniform float uObjectRadius;


// AntiAliasing 0=turn off, n=use n by n sub-pixels
#define AA {AA}
#define ITERATIONS {ITERATIONS}
#define DIMENSIONS {DIMENSIONS}
#define KALI kali_{ACCUMULATOR}
#define EIFFIE_MOD {EIFFIE_MOD}



vec4 reduce_dimensions(in vec4 v, in vec4 replace) {
    #if DIMENSIONS <= 2
        v.z = replace.z;
        v.w = replace.w;
    #endif
    
    #if DIMENSIONS == 3
        v.w = replace.w;
    #endif
    
    return v;
}



vec4 kali_final(in vec4 p, in vec4 param) {
    for (int i=0; i<ITERATIONS-1; ++i) {
        p = abs(p) / dot(p, p);
        p -= param;
    }
    p = abs(p) / dot(p, p);
    return reduce_dimensions(p, vec4(0, 0, 0, 1));
}


vec4 kali_average(in vec4 p, in vec4 param) {
    vec4 acc = vec4(0);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc += p;
        p -= param;
    }
    acc /= float(ITERATIONS);
    return reduce_dimensions(acc, vec4(0, 0, 0, 1));
}


vec4 kali_min(in vec4 p, in vec4 param) {
    vec4 acc = vec4(1e8);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc = min(acc, p);
        p -= param;
    }
    return reduce_dimensions(acc, vec4(0, 0, 0, 1));
}


vec4 kali_max(in vec4 p, in vec4 param) {
    vec4 acc = vec4(0);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc = max(acc, p);
        p -= param;
    }
    return reduce_dimensions(acc, vec4(0, 0, 0, 1));
}


vec4 kali_distance_plane(in vec4 p, in vec4 param) {
    float min_dist = 1e8;
    float scale = 1.;
    for (int i=0; i<ITERATIONS; ++i) {
        float dot_p = 1e-8 + dot(p, p);
        scale /= dot_p;
        p = abs(p) / dot_p;
        float dist = p.x - uObjectDistance;
        #if EIFFIE_MOD == 1
            dist *= scale;
        #endif
        min_dist = min(dist, min_dist);
        p -= param;
    }
    float inside = max(
        smoothstep(.01, .0, abs(min_dist - uObjectRadius)),
        .5 * smoothstep(.0, -0.01, min_dist - uObjectRadius)
    );
    return vec4(inside, inside, inside, 1);
}

vec4 kali_distance_cylinder(in vec4 p, in vec4 param) {
    float min_dist = 1e8;
    float scale = 1.;
    for (int i=0; i<ITERATIONS; ++i) {
        float dot_p = 1e-8 + dot(p, p);
        scale /= dot_p;
        p = abs(p) / dot_p;
        float dist = length(p.xy - vec2(uObjectDistance,0));
        #if EIFFIE_MOD == 1
            dist *= scale;
        #endif
        min_dist = min(dist, min_dist);
        p -= param;
    }
    float inside = max(
        smoothstep(.01, .0, abs(min_dist - uObjectRadius)),
        .5 * smoothstep(.0, -0.01, min_dist - uObjectRadius)
    );
    return vec4(inside, inside, inside, 1);
}

vec4 kali_distance_sphere(in vec4 p, in vec4 param) {
    float min_dist = 1e8;
    float scale = 1.;
    for (int i=0; i<ITERATIONS; ++i) {
        float dot_p = 1e-8 + dot(p, p);
        scale /= dot_p;
        p = abs(p) / dot_p;
        float dist = length(p.xyz - vec3(uObjectDistance, 0, 0));
        #if EIFFIE_MOD == 1
            dist *= scale;
        #endif
        min_dist = min(dist, min_dist);
        p -= param;
    }
    float inside = max(
        smoothstep(.01, .0, abs(min_dist - uObjectRadius)),
        .5 * smoothstep(.0, -0.01, min_dist - uObjectRadius)
    );
    return vec4(inside, inside, inside, 1);
}


vec4 kali_distance_cube(in vec4 p, in vec4 param) {
    vec4 location = vec4(uObjectDistance, 0, 0, 0);
    vec4 radius = reduce_dimensions(vec4(uObjectRadius), vec4(0));
    float min_dist = 1e8;
    float scale = 1.;
    for (int i=0; i<ITERATIONS; ++i) {
        float dot_p = 1e-8 + dot(p, p);
        scale /= dot_p;
        p = abs(p) / dot_p;
        /* https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm */
        vec4 q = abs(p - location) - radius;
        float dist = length(max(q,0.0)) + min(max(q.x,max(q.y,max(q.z, q.w))),0.0);
        #if EIFFIE_MOD == 1
            dist *= scale;
        #endif
        min_dist = min(dist, min_dist);
        p -= param;
    }
    float inside = max(
        smoothstep(.01, .0, abs(min_dist - uObjectRadius)),
        .5 * smoothstep(.0, -0.01, min_dist - uObjectRadius)
    );
    return vec4(inside, inside, inside, 1);
}


vec4 frag_to_color(in vec2 fragCoord) {
    vec4 uv = vec4((fragCoord - uResolution * .5) / uResolution.y * 2., 0., 0.);
    uv = uv * uKaliScale + uKaliPosition;
    
    vec4 col = KALI(
        reduce_dimensions(uv, vec4(0, 0, 0, 0)),
        reduce_dimensions(uKaliParam, vec4(0, 0, 0, 0))
    );
    
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



<div id="kali-01" style="width: 768px; height: 768px;"></div>
<div id="kali-01-ctl"></div>

<script>
    render_kali("kali-01", "kali-01-ctl");
</script>

