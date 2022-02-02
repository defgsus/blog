window.addEventListener("DOMContentLoaded", () => {
    /*
    class Diagram {
        constructor(element) {
            this.element = element;
            this.view = {
                x: 0, y: 0,
                scale: 10.;
            }
            diagram.setAttribute("width", "100%");
            diagram.setAttribute("height", `${window.innerHeight-30}px`);
            diagram.setAttribute("viewBox", "-100 -100 200 200");
        }

        width = () => this.element.boundingBox.width();
        height = () => this.element.boundingBox.height();

        viewbox = () => {

        }

    }
    */
    const diagram = document.getElementById("network");
    window.diagram = diagram;
    const view = {
        //x: 1., y: 0., zoom: 5.,
        x: 983., y: 258., zoom: 1.7,
    };

    const network_data = {
        nodes: [],
        edges: [],
        // node.id -> node
        node_map: {},
    };

    document.querySelector('button[name="stop"]').addEventListener("click", e => {
    });

    function event_to_world_coords(event) {
        const
            bb = diagram.getBoundingClientRect(),
            elem_x = event.clientX - bb.left,
            elem_y = event.clientY - bb.top,
            x = (elem_x - bb.width/2) / view.zoom + view.x,
            y = (elem_y - bb.height/2) / view.zoom + view.y;
        return [x, y];
    }
    diagram.addEventListener("click", e => {
        const [x, y] = event_to_world_coords(e);
        const node = node_at(x, y);
        console.log(view, node);
        select_nodes([node.id]);
    });

    let last_mouse_down, last_mouse_down_view;
    diagram.addEventListener("mousedown", e => {
        last_mouse_down = [e.clientX, e.clientY];
        last_mouse_down_view = [view.x, view.y];
    });
    diagram.addEventListener("mousemove", e => {
        if (!e.buttons)
            return;
        if (last_mouse_down) {
            view.x = last_mouse_down_view[0] - (e.clientX - last_mouse_down[0]) / view.zoom;
            view.y = last_mouse_down_view[1] - (e.clientY - last_mouse_down[1]) / view.zoom;
        }
        update_network_view();
    });
    diagram.addEventListener("wheel", e => {
        e.stopPropagation();
        e.preventDefault();
        const
            new_zoom = Math.max(0.0001, view.zoom * (e.deltaY > 0 ? .95 : 1.05)),
            coords = event_to_world_coords(e);
        // TODO: zoom centered around cursor position
        view.zoom = new_zoom;
        update_network_view();
    });

    fetch("../layouted.dot")
        .then(response => response.text())
        .then(dot_string => {
            try {
                return vis.network.convertDot(dot_string)
            }
            catch (e) {
                let msg = `${e}`;
                const match = msg.match(/.*\(char (\d+)\)/);
                if (match && match.length) {
                    const idx = Math.max(0, parseInt(match[1]) - 200);
                    msg += ` at: ${dot_string.slice(idx, idx + 300)}`
                }
                throw `error parsing dot: ${msg}`;
            }
        })
        .then(data => {
            network_data.nodes = data.nodes;
            network_data.edges = data.edges;
            for (const n of network_data.nodes) {
                n.x *= 100.;
                n.y *= 100.;
                n.hidden = false;
                n.alpha = .3;
                n.zindex = 0;
                n.radius = 1. + n.hubOrAuthority * 10.;
                network_data.node_map[n.id] = n;
            }
            init_network();
        });

    function node_at(x, y) {
        for (const n of network_data.nodes) {
            const r2 = (n.x - x) * (n.x - x) + (n.y - y) * (n.y - y);
            if (r2 < n.radius * n.radius)
                return n;
        }
    }

    function node_color(node) {
        return `rgba(${node.hub*255},${node.authority*255},0,${node.alpha})`;
    }

    function init_network() {
        /*d3.select("#network")
            .attr("width", "100%")
            .attr("height", height)
            .attr("viewBox", `0 0 ${width} ${height}`)
            .append("g")
                .attr("class", "world-transform")
                .attr("transform", `translate(${width/2.},${height/2}) scale(${view.zoom}) translate(${view.x},${view.y})`)
                .selectAll("circle")
                .data(network_data.nodes)
                .enter()
                    .append("circle")
                    .attr("r", 0.05)
                    .attr("cx", n => n.x)
                    .attr("cy", n => n.y)
                    .attr("style", n => `fill: rgb(${n.hub*255},${n.authority*255},0)`)
        ;*/
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "world-transform");
        for (const edge of network_data.edges) {
            const
                node_from = network_data.node_map[edge.from],
                node_to = network_data.node_map[edge.to];
            const elem = document.createElementNS("http://www.w3.org/2000/svg", "line");
            elem.setAttribute("class", "edge");
            elem.setAttribute("id", `edge-${edge.id}`);
            elem.setAttribute("x1", `${node_from.x}`);
            elem.setAttribute("y1", `${node_from.y}`);
            elem.setAttribute("x2", `${node_to.x}`);
            elem.setAttribute("y2", `${node_to.y}`);
            elem.setAttribute("style", `stroke: ${node_color(node_to)}`);
            g.appendChild(elem);
        }
        for (const node of network_data.nodes) {
            const elem = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            elem.setAttribute("class", "node");
            elem.setAttribute("id", `node-${node.id}`);
            elem.setAttribute("r", `${node.radius}`);
            elem.setAttribute("cx", `${node.x}`);
            elem.setAttribute("cy", `${node.y}`);
            elem.setAttribute("style", `fill: ${node_color(node)}`);
            g.appendChild(elem);
        }
        diagram.appendChild(g);
        update_network_view();
    }
    function update_network_view() {
        diagram.setAttribute("width", "100%");
        const
            width = diagram.getBoundingClientRect().width,
            height = window.innerHeight - 30;
        diagram.setAttribute("height", `${height}px`);
        diagram.setAttribute("viewBox", `0 0 ${width} ${height}`);
        const tr = diagram.querySelector("g.world-transform");
        if (tr) tr.setAttribute(
            "transform", `translate(${width/2.},${height/2}) scale(${view.zoom}) translate(${-view.x},${-view.y})`
        );
    }

    function update_network() {
        for (const elem of diagram.querySelectorAll(".node")) {
            const node = network_data.node_map[parseInt(elem.id.slice(5))];
            update_network_element(elem, node);
        }
    }

    function update_network_element(elem, node) {
        if (node.hidden)
            elem.classList.add("hidden");
        else
            elem.classList.remove("hidden");
        elem.setAttribute("r", `${node.radius}`);
        elem.setAttribute("cx", `${node.x}`);
        elem.setAttribute("cy", `${node.y}`);
        elem.setAttribute("style", `fill: ${node_color(node)}; z-index: ${node.zindex}`);
    }

    function select_nodes(node_ids) {
        for (const id of node_ids) {
            const node = network_data.node_map[id];
            node.alpha = 1.;
            node.hidden = false;
            const
                elem = diagram.querySelector(`#node-${id}`),
                parent = elem.parentElement;
            parent.removeChild(elem);
            update_network_element(elem, node);
            parent.appendChild(elem);
        }
    }
});
