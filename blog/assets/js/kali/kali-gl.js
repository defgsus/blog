
/*
    element_id must be id of a DIV with a certain size

 */
function render_kali(element_id, control_element_id, parameters) {

    const controls = [
        {id: "dimensions", name: "dimensions", type: "int", step: 1, min: 2, max: 4, default: 2, recompile: true, group: 1},
        {id: "iterations", name: "iterations", type: "int", step: 1, min: 1, default: 11, recompile: true, group: 1},
        {id: "accumulator", name: "accumulator", type: "select", default: "final", recompile: true, group: 2, options: [
                ["final", "final"],
                ["average", "average"],
                ["min", "min"],
                ["max", "max"],
                ["exp", "exp"],
                ["distance_plane", "plane X"],
                ["distance_cylinder", "cylinder XY"],
                ["distance_sphere", "sphere XYZ"],
                ["distance_cube", "cube XYZW"],
            ],
        },
        {id: "exp_scale", name: "exp scale", type: "float", step: 1, default: 50., group: 2, exp_acc: true},
        {id: "object_distance", name: "object location (x)", type: "float", step: 0.01, default: .0, group: 5, distance_acc: true},
        {id: "object_radius", name: "object radius", type: "float", step: 0.01, default: .1, group: 5, distance_acc: true},
        {id: "eiffie_mod", name: "eiffie mod", type: "checkbox", default: false, recompile: true, group: 5, distance_acc: true},
        {id: "amplitude", name: "brightness", type: "float", step: .1, default: 1., group: 6},
        {id: "antialiasing", name: "antialiasing", type: "int", step: 1, min: 1, default: 2, recompile: true, group: 6},
        {id: "kali_param_x", name: "parameter x", type: "float", step: 0.01, default: .5, group: 10},
        {id: "kali_param_y", name: "y", type: "float", step: 0.01, default: .5, group: 10},
        {id: "kali_param_z", name: "z", type: "float", step: 0.01, default: .5, dimensions: 3, group: 10},
        {id: "kali_param_w", name: "w", type: "float", step: 0.01, default: .5, dimensions: 4, group: 10},
        {id: "position_x", name: "position x", type: "float", step: 0.01, default: .0, group: 11},
        {id: "position_y", name: "y", type: "float", step: 0.01, default: .0, group: 11},
        {id: "position_z", name: "z", type: "float", step: 0.01, default: .0, dimensions: 3, group: 11},
        {id: "position_w", name: "w", type: "float", step: 0.01, default: .0, dimensions: 4, group: 11},
        {id: "scale", name: "scale", type: "float", step: 0.01, default: 1.},
    ];

    const default_params = {};
    for (const p of controls) {
        default_params[p.id] = p.default;
    }

    parameters = {...default_params, ...parameters};

    const kaliShaderVert = document.querySelector("script.kali-exp-vert").innerText;
    const kaliShaderFrag = document.querySelector("script.kali-exp-frag").innerText;
    const element = document.getElementById(element_id);

    function log_error() {
        console.log(arguments);
        for (const a of arguments) {
            element.innerText += ` Error: ${a}`;
        }
    }

    function createGLContext(canvas) {
        let names = ["webgl", "experimental-webgl"];
        let context = null;
        for (let i = 0; i < names.length; i++) {
            try {
                context = canvas.getContext(names[i]);
            } catch (e) {
                log_error(e);
            }
            if (context) {
                break;
            }
        }
        if (context) {
            context.viewportWidth = canvas.width;
            context.viewportHeight = canvas.height;
        } else {
            log_error("Failed to create WebGL context!");
            return null;
        }
        return context;
    }

    function compileShader(ctx, shaderType, shaderSource) {
        const glShaderType = shaderType === "VERTEX"
            ? ctx.gl.VERTEX_SHADER
            : ctx.gl.FRAGMENT_SHADER;
        let shader = ctx.gl.createShader(glShaderType);

        ctx.gl.shaderSource(shader, shaderSource);
        ctx.gl.compileShader(shader);

        if (!ctx.gl.getShaderParameter(shader, ctx.gl.COMPILE_STATUS)) {
            log_error("shader compilation failed", shaderType, ctx.gl.getShaderInfoLog(shader));
            // console.log(shaderSource);
            return null;
        }
        return shader;
    }

    function setupShaders(ctx) {
        if (ctx.shaderProgram)
            ctx.gl.deleteProgram(ctx.shaderProgram);
        if (ctx.vertexShader)
            ctx.gl.deleteShader(ctx.vertexShader);
        if (ctx.fragmentShader)
            ctx.gl.deleteShader(ctx.fragmentShader);

        const shader_code = kaliShaderFrag
            .replace("{ITERATIONS}", `${ctx.parameters.iterations}`)
            .replace("{DIMENSIONS}", `${ctx.parameters.dimensions}`)
            .replace("{AA}", `${ctx.parameters.antialiasing}`)
            .replace("{ACCUMULATOR}", ctx.parameters.accumulator)
            .replace("{EIFFIE_MOD}", ctx.parameters.eiffie_mod ? "1": "0")
        ;
        ctx.vertexShader = compileShader(ctx, "VERTEX", kaliShaderVert);
        ctx.fragmentShader = compileShader(ctx, "FRAGMENT", shader_code);
        if (!ctx.vertexShader || !ctx.fragmentShader)
            return;

        ctx.shaderProgram = ctx.gl.createProgram();
        ctx.gl.attachShader(ctx.shaderProgram, ctx.vertexShader);
        ctx.gl.attachShader(ctx.shaderProgram, ctx.fragmentShader);
        ctx.gl.linkProgram(ctx.shaderProgram);
        if (!ctx.gl.getProgramParameter(ctx.shaderProgram, ctx.gl.LINK_STATUS)) {
            log_error("Failed to setup shaders");
            return;
        }
        ctx.gl.useProgram(ctx.shaderProgram);

        ctx.uniformLocation = {
            resolution: ctx.gl.getUniformLocation(ctx.shaderProgram, "uResolution"),
            kali_param: ctx.gl.getUniformLocation(ctx.shaderProgram, "uKaliParam"),
            position: ctx.gl.getUniformLocation(ctx.shaderProgram, "uKaliPosition"),
            scale: ctx.gl.getUniformLocation(ctx.shaderProgram, "uKaliScale"),
            amplitude: ctx.gl.getUniformLocation(ctx.shaderProgram, "uAmplitude"),
            object_distance: ctx.gl.getUniformLocation(ctx.shaderProgram, "uObjectDistance"),
            object_radius: ctx.gl.getUniformLocation(ctx.shaderProgram, "uObjectRadius"),
            exp_scale: ctx.gl.getUniformLocation(ctx.shaderProgram, "uExpScale"),
        };

        ctx.shaderProgram.vertexPositionAttribute = ctx.gl.getAttribLocation(ctx.shaderProgram, "aVertexPosition");

        ctx.shadersReady = true;
    }

    function setupBuffers(ctx) {
        ctx.vertexBuffer = ctx.gl.createBuffer();
        ctx.gl.bindBuffer(ctx.gl.ARRAY_BUFFER, ctx.vertexBuffer);
        let triangleVertices = [
            -1., -1., 0.0,
            1., -1., 0.0,
            1., 1., 0.0,

            -1., -1., 0.0,
            1., 1., 0.0,
            -1., 1., 0.0,
        ];
        ctx.gl.bufferData(ctx.gl.ARRAY_BUFFER, new Float32Array(triangleVertices), ctx.gl.STATIC_DRAW);
        ctx.vertexBuffer.itemSize = 3;
        ctx.vertexBuffer.numberOfItems = 6;
    }

    function draw(ctx) {
        let gl = ctx.gl;

        let bb = ctx.canvas.getBoundingClientRect();
        //gl.uniform2f(ctx.uniformLocation.resolution, ctx.canvas.width, ctx.canvas.height);
        gl.uniform2f(ctx.uniformLocation.resolution, bb.width, bb.height);
        gl.uniform4f(ctx.uniformLocation.kali_param,
            ctx.parameters.kali_param_x,
            ctx.parameters.kali_param_y,
            ctx.parameters.kali_param_z,
            ctx.parameters.kali_param_w,
        );
        gl.uniform4f(ctx.uniformLocation.position,
            ctx.parameters.position_x,
            ctx.parameters.position_y,
            ctx.parameters.position_z,
            ctx.parameters.position_w,
        );
        gl.uniform1f(ctx.uniformLocation.scale, ctx.parameters.scale);
        gl.uniform1f(ctx.uniformLocation.amplitude, ctx.parameters.amplitude);
        gl.uniform1f(ctx.uniformLocation.object_distance, ctx.parameters.object_distance);
        gl.uniform1f(ctx.uniformLocation.object_radius, ctx.parameters.object_radius);
        gl.uniform1f(ctx.uniformLocation.exp_scale, ctx.parameters.exp_scale);

        gl.viewport(0, 0, ctx.canvas.width, ctx.canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.vertexAttribPointer(ctx.shaderProgram.vertexPositionAttribute,
            ctx.vertexBuffer.itemSize, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(ctx.shaderProgram.vertexPositionAttribute);

        gl.drawArrays(gl.TRIANGLES, 0, ctx.vertexBuffer.numberOfItems);
    }

    function start_context(canvas, parameters) {
        let ctx = {};
        ctx.canvas = canvas;
        ctx.parameters = parameters;
        ctx.gl = createGLContext(ctx.canvas);
        if (ctx.gl) {
            setupShaders(ctx);
            if (ctx.shadersReady) {
                setupBuffers(ctx);
                ctx.gl.clearColor(0.0, 0.0, 0.0, 1.0);
                draw(ctx);
            }
            return ctx;
        }
    }

    const bb = element.getBoundingClientRect();

    element.classList.add("kali-gl");
    element.innerHTML = `<canvas width="${bb.width}" height="${bb.height}"></canvas>`;
    const canvas = element.querySelector("canvas");

    const context = start_context(canvas, parameters);

    context._update_params_timeout = null;
    context.update_parameters = function(params) {
        //console.log(params);
        const old_params = JSON.parse(JSON.stringify(context.parameters));
        context.parameters = {...context.parameters, ...params};
        for (const c of controls) {
            const elem = document.querySelector(`#${control_element_id}-${c.id}`);
            if (!elem)
                continue;
            if (c.type == "checkbox")
                elem.checked = context.parameters[c.id];
            else {
                let v = context.parameters[c.id];
                if (c.type === "float")
                    v = Math.round(v * 100000.) / 100000.;
                elem.value = v;
            }

            let hidden = false;
            if (c.dimensions) {
                const label = elem.parentElement;
                hidden = (c.dimensions > context.parameters.dimensions);
            }
            if (c.distance_acc) {
                hidden = !context.parameters.accumulator.startsWith("distance_");
            }
            if (c.exp_acc) {
                hidden = context.parameters.accumulator !== "exp";
            }
            if (hidden)
                elem.parentElement.classList.add("hidden");
            else
                elem.parentElement.classList.remove("hidden");
        }

        function render() {
            for (const c of controls) {
                if (c.recompile && context.parameters[c.id] !== old_params[c.id]) {
                    setupShaders(context);
                    break;
                }
            }
            draw(context);
        }

        if (context._update_param_timeout)
            clearTimeout(context._update_param_timeout);
        context._update_param_timeout = setTimeout(render, 1. / 30.);

    };

    context._move_timeout = null;
    context.move_to = function(new_position, new_scale, length=1.) {
        const
            old_pos_x = context.parameters.position_x,
            old_pos_y = context.parameters.position_y,
            old_scale = context.parameters.scale,
            startTime = new Date().getTime();

        function update() {
            let curTime = new Date().getTime(),
                t = Math.min(1, (curTime - startTime) / length / 1000.);
            t = t*t*(3-2*t);

            context.update_parameters({
                position_x: old_pos_x * (1. - t) + t * new_position[0],
                position_y: old_pos_y * (1. - t) + t * new_position[1],
                scale: old_scale * (1. - t) + t * new_scale,
            });

            if (t < 1) {
                if (context._move_timeout)
                    clearTimeout(context._move_timeout);
                context._move_timeout = setTimeout(update, 1. / 30.);
            }
        }
        update();
    };

    function create_control_elements(parameters) {
        let prev_group = 0;
        const html = controls.map(function(c, i) {
            const new_group = (c.group !== prev_group) || c.group === undefined;
            prev_group = c.group;
            let html = ``;
            let param_class = c.name.length > 1 ? "param-name" : "param-name-short";
            if (new_group) {
                if (i > 0)
                    html += `</div>`;
                html += `<div class="param-group">`;
            }
            else
                param_class += " right";
            html += `<label><b class="${param_class}">${c.name}</b> `;
            let type = c.type === "int" || c.type === "float" ? "number" : c.type;
            let elem_tag = c.type === "select" ? "select" : "input";
            html += `<${elem_tag} id="${control_element_id}-${c.id}" class="${c.type}" type="${type}"`;
            if (type === "checkbox" && parameters[c.id])
                html += ` checked`;
            else
                html += ` value="${parameters[c.id]}"`;
            if (c.step)
                html += ` step="${c.step}"`;
            if (c.min)
                html += ` min="${c.min}"`;
            if (c.max)
                html += ` max="${c.max}"`;
            html += `>`;
            if (c.options) {
                for (const o of c.options) {
                    html += `<option value="${o[0]}">${o[1]}</option>`;
                }
            }
            html += `</${elem_tag}>`;
            html += `<button id="${control_element_id}-${c.id}-reset">R</button>`;
            html += `</label>`;
            if (i+1 == controls.length)
                html += `</div>`;
            return html;
        }).join(" ");

        const elem = document.getElementById(control_element_id);
        elem.classList.add("kali-gl")
        elem.innerHTML = html;

        for (const c of controls) {
            document.querySelector(`#${control_element_id}-${c.id}`).addEventListener("input", function(e) {
                let v = e.target.value;
                if (c.type === "int")
                    v = parseInt(v)
                else if (c.type === "float")
                    v = parseFloat(v);
                else if (c.type === "checkbox")
                    v = e.target.checked;
                context.update_parameters({[c.id]: v});
            });
            document.querySelector(`#${control_element_id}-${c.id}-reset`).addEventListener("click", function(e) {
                context.update_parameters({[c.id]: c.default});
            });
        }
    }

    if (control_element_id) {
        create_control_elements(parameters);
        context.update_parameters();
    }

    canvas.addEventListener("click", function(e) {
        const bb = canvas.getBoundingClientRect(),
            pixel_x = Math.round(e.clientX - bb.left),
            pixel_y = bb.height - 1 - Math.round(e.clientY - bb.top),
            space_x = (pixel_x - bb.width * .5) / bb.height * 2. * context.parameters.scale + context.parameters.position_x,
            space_y = (pixel_y - bb.height * .5) / bb.height * 2. * context.parameters.scale + context.parameters.position_y;

        //console.log(pixel_x, pixel_y, space_x, space_y);
        context.move_to([space_x, space_y], context.parameters.scale * .66);
    });


    return context;
}
