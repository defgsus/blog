---
layout: post
title: kaliset explorer
custom_js: 
  - kali/kali-gl.js
custom_css: 
  - kali/kali-gl.css
---

<!-- I'd like to place the glsl code into own files but it's more complicated.. -->
<script type="application/x-glsl" class="kali-exp-vert">
#line 11
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
uniform float uExpScale;


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


vec4 kali_xyz(in vec4 p, in vec4 param) {
    vec4 acc = vec4(0);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        if (p.x > p.y)
            acc.x += 1.;
        else if (p.y > p.z)
            acc.y += 1.;
        else if (p.z > p.x)
            acc.z += 1.;
        p -= param;
    }
    acc.xyz /= max(acc.x, max(acc.y, acc.z));
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


vec4 kali_exp(in vec4 p, in vec4 param) {
    vec4 acc = vec4(0);
    for (int i=0; i<ITERATIONS; ++i) {
        p = abs(p) / dot(p, p);
        acc += exp(-p * uExpScale);
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
<div id="kali-01-ctl" style="margin-bottom: 2rem"></div>

<script>
    render_kali("kali-01", "kali-01-ctl");
</script>


In 2015, i started [shadertoying](https://www.shadertoy.com/user/bergi) a lot. And the first 
10 or so shaders where just about the ***kaliset***, a fractal discovered by the *Fractal Supremo*
Kali published in [this post](https://www.fractalforums.com/new-theories-and-research/very-simple-formula-for-fractal-patterns)
on fractal forums. Other *shadertoyers* had picked on the formula already. 
Like [Dave Hoskins](https://www.shadertoy.com/view/Msl3WH), [huwb](https://www.shadertoy.com/view/MslGDX) or
[guil](https://www.shadertoy.com/view/lsBGWK) and [Kali](https://www.shadertoy.com/view/XlfGRj) himself.

The formula is basically:

    p = abs(p) / dot(p, p) - param

Where `p` is the input position and `param` some parameter in about the range [0, 2]. 
`p` and `param` can have any number of dimensions. 

The javascript version above is the remake of [this shader](https://www.shadertoy.com/view/XdVSRc).
There to find good magic parameters. A disciplined form of 
[interactive evolution](https://www.shadertoy.com/view/XdyGWw).

And there are some good magic parameters! I was looking for ways to use the resulting numbers in a 
[signed distance field](https://en.wikipedia.org/wiki/Signed_distance_function) (SDF)
to render 3D versions which did not work so well. 
The kaliset is chaotic and discontinuous. Used as a surface
displacement it creates exponential peaks and holes and distorts the continuity required
for an SDF. 
The [first](https://www.shadertoy.com/view/4tfGD8) and [second try](https://www.shadertoy.com/view/Xlf3D8)
where just nicely coloring the artifacts so they might look deliberate. 
This [shader by guil](https://www.shadertoy.com/view/MdB3DK) somehow rendered complicated
3d surfaces but i did not understand the accumulation part. Later i found a 
[visually pleasing way](https://www.shadertoy.com/view/XtlGDH) of making it look right which brought
[eiffie](https://www.shadertoy.com/user/eiffie) to [an elegant method](https://www.shadertoy.com/view/XtlGRj)
of approximating a true distance function.
    
Here's an example in GLSL.   

```glsl
float kaliset_distance_sphere(in vec3 p, in vec3 param) {
    float dist = 1e8;   // start distance is infinitely away
    float scale = 1.;   // start with unstretched space
    for (int i=0; i<ITERATIONS; ++i) {
        float dot_p = dot(p, p);
        scale /= dot_p;             // stretch the space accordingly
        p = abs(p) / dot_p;         // the magic formula
        dist = min(dist,            // find closest distance
            length(p.xyz) * scale   // measure distance in stretched space
        );
        p -= param;                 // the magic parameter
    }
    return dist;
}
```

The [geometric union](https://en.wikipedia.org/wiki/Constructive_solid_geometry) is
assembled from spheres at the unit position at each iteration step. By using the `scale`
parameter the distance function stays *true-ish*. 

This made 3d environments like [KaliTrace](https://www.shadertoy.com/view/4sKXWG)
or [Cave Quest](https://www.shadertoy.com/view/XdGXD3) a piece of cake.
Apart from the *eiffie mod*, one can simply disregard the distance and just look 
for the surface [in fixed steps](https://www.shadertoy.com/view/Mtf3W7)
or by [random search](https://www.shadertoy.com/view/XdKXzh). Here's one by 
[janiorca](https://www.shadertoy.com/view/tdVGDy).

Now, this post has been lying around for a while. It shows *the true kaliset*, without
polishing. And no 3d. And in ordered chaoticity. And allows a couple of parameters to change. 

The **accumulator** setting defines how the color is calculated from the `p` value.
You can check the [GLSL source](https://raw.githubusercontent.com/defgsus/blog/master/blog/_posts/2021/2021-10-12-kaliset-explorer.md)
for detail.

- **final**: simply the last value of p
- **average**: average of all p values
- **xyz**: sum of x > y, y > z or z > x
- **min** and **max**: the minimum or maximum value of p
- **exp**: sum of exp(-p * exp_scale)
- **plane X**: distance to a place perpendicular to x axis
- **cylinder XY**: distance to a cylinder extending on the xy plane
- **sphere XYZ**: distance to a sphere
- **cube XYZW**: distance to a 3d cube in 4 dimensions or.. i don't know really

If one of the distance accumulators is selected, additional parameters pop up and 
allow changing the size and position of the objects and you can turn on the **eiffie mod**. 
The rest of the settings should be inferable.

Thanks to Kali, eiffie, Dave, iq and all the others for the fantastic shadertoy years.
