
/*
    element_id must be id of a DIV with a certain size

 */
function render_kali(element_id, control_element_id, parameters) {

    const controls = [
        {id: "dimensions", name: "dimensions", type: "int", step: 1, min: 1, default: 2, recompile: true},
        {id: "iterations", name: "iterations", type: "int", step: 1, min: 1, default: 11, recompile: true},
        {id: "kali_param_x", name: "param x", type: "float", step: 0.01, default: .5},
        {id: "kali_param_y", name: "param y", type: "float", step: 0.01, default: .5},
        {id: "kali_param_z", name: "param z", type: "float", step: 0.01, default: .5},
        {id: "kali_param_w", name: "param w", type: "float", step: 0.01, default: .5},
        {id: "position_x", name: "pos x", type: "float", step: 0.01, default: .0},
        {id: "position_y", name: "pos y", type: "float", step: 0.01, default: .0},
        {id: "position_z", name: "pos z", type: "float", step: 0.01, default: .0},
        {id: "position_w", name: "pos w", type: "float", step: 0.01, default: .0},
        {id: "scale", name: "scale", type: "float", step: 0.01, default: 1.},
        {id: "antialiasing", name: "antialiasing", type: "int", step: 1, min: 1, default: 2, recompile: true},
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
            .replace("{AA}", `${ctx.parameters.antialiasing}`)
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

        gl.viewport(0, 0, ctx.canvas.width, ctx.canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.vertexAttribPointer(ctx.shaderProgram.vertexPositionAttribute,
            ctx.vertexBuffer.itemSize, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(ctx.shaderProgram.vertexPositionAttribute);

        gl.drawArrays(gl.TRIANGLES, 0, ctx.vertexBuffer.numberOfItems);

        /*setTimeout(function () {
            draw(ctx)
        }, 1. / 30.);
         */
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

    element.innerHTML = `<canvas width="${bb.width}" height="${bb.height}"></canvas>`;
    const canvas = element.querySelector("canvas");

    const context = start_context(canvas, parameters);

    context.update_parameters = function(params) {
        const old_dimensions = this.parameters.dimensions;
        const old_iterations = this.parameters.iterations;
        const old_antialiasing = this.parameters.antialiasing;
        this.parameters = {...this.parameters, ...params};
        if (old_iterations !== this.parameters.iterations
            || old_antialiasing !== this.parameters.antialiasing
            || old_dimensions !== this.parameters.dimensions)
            setupShaders(this);
        draw(this);
    }.bind(context);

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
            //console.log(t, startTime, curTime);
            context.parameters.position_x = old_pos_x * (1. - t) + t * new_position[0];
            context.parameters.position_y = old_pos_y * (1. - t) + t * new_position[1];
            context.parameters.scale = old_scale * (1. - t) + t * new_scale;

            draw(context);
            if (t < 1) {
                if (context._move_timeout)
                    clearTimeout(context._move_timeout);
                context._move_timeout = setTimeout(update, 1. / 30.);
            }
        }
        update();
    };

    function create_control_elements(parameters) {
        const html = controls.map(function(c) {
            let html = `<label style="white-space: nowrap">${c.name} `;
            let type = c.type === "int" || c.type === "float" ? "number" : type;
            html += `<input id="${control_element_id}-${c.id}" type="${type}" value="${parameters[c.id]}"`;
            if (c.step)
                html += ` step="${c.step}"`;
            if (c.min)
                html += ` min="${c.min}"`;
            html += `></input></label>`;
            return html;
        }).join(" ");

        document.getElementById(control_element_id).innerHTML = html;

        for (const c of controls) {
            document.querySelector(`#${control_element_id}-${c.id}`).addEventListener("input", function(e) {
                let v = e.target.value;
                if (c.type === "int")
                    v = parseInt(v)
                else if (c.type === "float")
                    v = parseFloat(v);
                context.update_parameters({[c.id]: v});
            });
        }
    }

    if (control_element_id) {
        create_control_elements(parameters);
    }

    canvas.addEventListener("click", function(e) {
        const bb = canvas.getBoundingClientRect(),
            pixel_x = Math.round(e.clientX - bb.left),
            pixel_y = bb.height - 1 - Math.round(e.clientY - bb.top),
            space_x = (pixel_x - bb.width * .5) / bb.height * 2. * context.parameters.scale + context.parameters.position_x,
            space_y = (pixel_y - bb.height * .5) / bb.height * 2. * context.parameters.scale + context.parameters.position_y;

        //console.log(pixel_x, pixel_y, space_x, space_y);
        context.move_to([space_x, space_y], context.parameters.scale * .7);
    });


    return context;
}
