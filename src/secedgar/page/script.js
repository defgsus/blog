class Network {

    constructor(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
        this.node_map = {};
        this.edge_map = {};
        for (const n of this.nodes) {
            n.id = `${n.id}`;
            n.x *= 1000.;
            n.y *= 1000.;
            n.hidden = false;
            n.selected = false;
            n.highlight = 0.;
            n.alpha = .3;
            n.radius = 10.;
            n.edges = [];
            n.edges_out = [];
            n.edges_in = [];
            this.node_map[n.id] = n;
        }
        for (const e of this.edges) {
            e.id = `${e.from}-${e.to}`;
            e.node_from = this.node_map[e.from];
            e.node_to = this.node_map[e.to];
            e.node_from.edges.push(e);
            e.node_from.edges_out.push(e);
            e.node_to.edges.push(e);
            e.node_to.edges_in.push(e);
            this.edge_map[e.id] = e;
        }
        this.selected_nodes = [];
        this.on_selection_changed = (
            selected_nodes, unselected_nodes,
            selected_edges, unselected_edges
        ) => {};
    }

    get_node_min_max = (field) => {
        let mi = null, ma = null;
        for (const n of this.nodes) {
            if (typeof n[field] === "number") {
                if (mi === null) {
                    mi = n[field];
                    ma = n[field];
                } else {
                    mi = Math.min(mi, n[field]);
                    ma = Math.max(ma, n[field]);
                }
            }
        }
        return [mi, ma];
    };

    center = () => {
        let x = 0, y = 0;
        for (const n of this.nodes) {
            x += n.x;
            y += n.y;
        }
        return [x / this.nodes.length, y / this.nodes.length];
    };

    node_at = (x, y) => {
        for (const n of this.nodes) {
            const r2 = (n.x - x) * (n.x - x) + (n.y - y) * (n.y - y);
            if (r2 < n.radius * n.radius)
                return n;
        }
    };

    traverse_nodes = (start_id_or_node, distance=1) => {
        const start_id = typeof start_id_or_node === "object" ? start_id_or_node.id : start_id_or_node;
        const todo = new Set([[start_id, 0]]);
        const done = new Set();
        const returned_nodes = [];
        while (todo.size) {
            const node_id_and_distance = todo.entries().next().value[0];
            const node_id = node_id_and_distance[0];
            const cur_distance = node_id_and_distance[1];
            todo.delete(node_id_and_distance);
            done.add(node_id);
            returned_nodes.push([this.node_map[node_id], cur_distance]);
            if (cur_distance >= distance)
                continue;

            const node = this.node_map[node_id];

            for (const e of node.edges_out) {
                if (!done.has(e.to)) {
                    todo.add([e.node_to.id, cur_distance + 1]);
                }
            }
            for (const e of node.edges_in) {
                if (!done.has(e.from)) {
                    todo.add([e.node_from.id, cur_distance + 1]);
                }
            }
        }
        return returned_nodes;
    };

    update_radius = (field, min_radius, max_radius) => {
        const field_min_max = this.get_node_min_max(field);
        for (const n of this.nodes) {
            const radius = min_radius + (max_radius - min_radius) * (
                (n[field] - field_min_max[0]) / (field_min_max[1] - field_min_max[0])
            );
            n.radius = isNaN(radius) ? min_radius : radius;
        }
    };

    set_selected = (nodes_or_ids_and_distances, max_distance) => {
        const selected_ids = new Set(nodes_or_ids_and_distances.map(n => (
            typeof n[0] === "object" ? n[0].id : n[0]
        )));
        const selected_edge_ids = new Set();
        for (const node_id of selected_ids) {
            const node = this.node_map[node_id];
            for (const edge of node.edges)
                if (selected_ids.has(edge.from) && selected_ids.has(edge.to))
                    selected_edge_ids.add(edge.id);
        }
        const
            selected_nodes = [],
            unselected_nodes = [],
            selected_edges = new Set(),
            unselected_edges = new Set();
        for (const node of this.selected_nodes) {
            if (!selected_ids.has(node.id)) {
                unselected_nodes.push(node);
                node.selected = false;
                node.alpha = .3;
            }
            for (const edge of node.edges) {
                if (!selected_edge_ids.has(edge.id)) {
                    unselected_edges.add(edge);
                    edge.selected = false;
                    edge.alpha = .3;
                }
            }
        }
        for (const node_or_id_and_distance of nodes_or_ids_and_distances) {
            const node = typeof node_or_id_and_distance[0] === "object"
                ? node_or_id_and_distance[0] : this.node_map[node_or_id_and_distance];
            const weight = 1. - Math.max(0, node_or_id_and_distance[1] - 1) / max_distance;

            node.selected = true;
            node.alpha = .4 + .6 * weight;
            node.highlight = .3 * weight;
            selected_nodes.push(node);
            for (const edge of node.edges) {
                if (selected_ids.has(edge.from) && selected_ids.has(edge.to)) {
                    selected_edges.add(edge);
                    edge.selected = true;
                    edge.alpha = .3 + .3 * weight;
                    edge.highlight = .3 * weight;
                }
            }
        }
        this.selected_nodes = selected_nodes;
        this.on_selection_changed(
            selected_nodes, unselected_nodes,
            Array.from(selected_edges), Array.from(unselected_edges),
        );
    };
}


class Diagram {

    constructor(element) {
        this.element = element;
        this.view = {
            x: 0,
            y: 0,
            zoom: 10.
        };
        this.network = null;
        this.on_click = (x, y) => {};

        this._last_mouse_down = [0, 0];
        this._last_mouse_down_view = [0, 0];
        this._has_mouse_moved = false;
        this.element.addEventListener("mousedown", e => {
            this._last_mouse_down = [e.clientX, e.clientY];
            this._last_mouse_down_view = [this.view.x, this.view.y];
            this._has_mouse_moved = false;
        });
        this.element.addEventListener("mouseup", e => {
            if (!this._has_mouse_moved) {
                const [x, y] = this.event_to_world_coords(e);
                this.on_click(x, y);
            }
        });
        this.element.addEventListener("mousemove", e => {
            if (!e.buttons)
                return;
            this._has_mouse_moved = true;
            if (this._last_mouse_down) {
                this.view.x =
                    this._last_mouse_down_view[0] - (e.clientX - this._last_mouse_down[0]) / this.view.zoom;
                this.view.y =
                    this._last_mouse_down_view[1] - (e.clientY - this._last_mouse_down[1]) / this.view.zoom;
            }
            this.update_view();
        });
        this.element.addEventListener("wheel", e => {
            e.stopPropagation();
            e.preventDefault();
            const
                new_zoom = Math.max(0.0001, this.view.zoom * (e.deltaY > 0 ? .95 : 1.05)),
                coords = this.event_to_world_coords(e);
            // TODO: zoom centered around cursor position
            this.view.zoom = new_zoom;
            this.update_view();
        });
    }

    width = () => this.element.getBoundingClientRect().width;
    height = () => this.element.getBoundingClientRect().height;

    event_to_world_coords = (event) => {
        const
            bb = this.element.getBoundingClientRect(),
            elem_x = event.clientX - bb.left,
            elem_y = event.clientY - bb.top,
            x = (elem_x - bb.width/2) / this.view.zoom + this.view.x,
            y = (elem_y - bb.height/2) / this.view.zoom + this.view.y;
        return [x, y];
    };

    node_rgb = (node) => {
        let r = node.hub,
            g = node.authority,
            b = 1. - node.hubOrAuthority;
        r += node.highlight;
        g += node.highlight;
        b += node.highlight;
        return [r, g, b];
    };

    node_color = (node) => {
        const [r, g, b] = this.node_rgb(node);
        return this._to_rgba(r, g, b, node.alpha);
    };

    edge_color = (edge) => {
        const t = .8;
        const [r1, g1, b1] = this.node_rgb(edge.node_from);
        const [r2, g2, b2] = this.node_rgb(edge.node_to);
        const r = r1 * (1. - t) + t * r2;
        const g = g1 * (1. - t) + t * g2;
        const b = b1 * (1. - t) + t * b2;
        const a = edge.node_from.alpha * (1. - t) + t * edge.node_to.alpha;
        return this._to_rgba(r, g, b, a);
    };

    _to_rgba = (r, g, b, a) => {
        r = Math.max(0, Math.min(255, r * 256)).toFixed();
        g = Math.max(0, Math.min(255, g * 256)).toFixed();
        b = Math.max(0, Math.min(255, b * 256)).toFixed();
        a = Math.max(0, Math.min(1, a)).toFixed(3);
        return `rgba(${r},${g},${b},${a})`
    };

    _set_title = (elem, text) => {
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        const tex = document.createTextNode(text);
        title.appendChild(tex);
        elem.appendChild(title);
    };

    _rotate = (x, y, degree) => {
        const a = degree / 180. * Math.PI;
        const si = Math.sin(a);
        const co = Math.cos(a);
        return [
            x * co - y * si,
            x * si + y * co,
        ]
    };

    _get_edge_positions = (edge) => {
        const node1 = edge.node_from;
        const node2 = edge.node_to;
        let dx = (node2.x - node1.x);
        let dy = (node2.y - node1.y);
        const length = Math.sqrt(dx*dx + dy*dy);
        if (length) {
            dx /= length;
            dy /= length;
        }
        const arrow_length = 10 + edge.weight * 10.;
        const arrow_degree = 10 + edge.weight * 10.;
        const [dx1, dy1] = this._rotate(dx, dy, arrow_degree);
        const [dx2, dy2] = this._rotate(dx, dy, -arrow_degree);
        return [
            node1.x + node1.radius * dx,
            node1.y + node1.radius * dy,
            node2.x - node2.radius * dx,
            node2.y - node2.radius * dy,
            node2.x - (node2.radius + arrow_length) * dx2,
            node2.y - (node2.radius + arrow_length) * dy2,
            node2.x - (node2.radius + arrow_length) * dx1,
            node2.y - (node2.radius + arrow_length) * dy1,
        ];
    };

    set_network = (network) => {
        this.network = network;
        this.network.on_selection_changed = this.update_selection;

        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "world-transform");
        for (const edge of this.network.edges) {
            const color = this.edge_color(edge);
            const edge_g = document.createElementNS("http://www.w3.org/2000/svg", "g");
            edge_g.setAttribute("class", `edge edge-${edge.id}`);
            edge_g.setAttribute("style", `stroke: ${color}; fill: ${color}`);
            edge.element = edge_g;
            this._set_title(edge_g,
                `${edge.node_from.name} holds`
                + ` ${(edge.weight*100).toFixed(2)}% of ${edge.node_to.name}`
                + ` ($ ${(edge.sharesThousandDollar * 1000).toLocaleString("en")})`
            );
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            edge_g.appendChild(line);
            edge.element_line = line;

            const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
            edge_g.appendChild(arrow);
            edge.element_polygon = arrow;
            g.appendChild(edge_g);
            this.update_edge(edge);
        }
        for (const node of this.network.nodes) {
            const node_g = document.createElementNS("http://www.w3.org/2000/svg", "g");
            node_g.setAttribute("class", `node node-${node.id}`);
            node.element = node_g;
            this._set_title(
                node_g,
                `${node.name}`
                + `\n$ ${(node.totalHoldingsMillionDollar * 1000000).toLocaleString("en")} total holdings`
            );
            let elem = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            node_g.appendChild(elem);
            node.element_circle = elem;
            elem = document.createElementNS("http://www.w3.org/2000/svg", "text");
            node_g.appendChild(elem);
            node.element_text = elem;
            elem.setAttribute("style", "stroke: black; font-size: 1rem");
            elem.setAttribute("paint-order", "stroke");
            elem.classList.add("hidden");
            elem.appendChild(document.createTextNode(node.name));
            g.appendChild(node_g);
            this.update_node(node, node_g);
        }
        this.element.appendChild(g);
        this.update_view();
    };

    update_all = () => {
        this.update_view();
        for (const node of this.network.nodes) {
            this.update_node(node);
        }
        for (const edge of this.network.edges) {
            this.update_edge(edge);
        }
    };

    update_view = () => {
        this.element.setAttribute("width", "100%");
        const
            width = this.element.getBoundingClientRect().width,
            height = window.innerHeight - 30;
        this.element.setAttribute("height", `${height}px`);
        this.element.setAttribute("viewBox", `0 0 ${width} ${height}`);
        const tr = this.element.querySelector("g.world-transform");
        if (tr) {
            tr.setAttribute(
                "transform",
                `translate(${width/2.},${height/2})`
                + ` scale(${this.view.zoom}) translate(${-this.view.x},${-this.view.y})`
            );
        }
    };

    update_node = (node) => {
        const color = this.node_color(node);
        const elem = node.element;
        elem.setAttribute("style", `fill: ${color}`);
        if (node.hidden)
            elem.classList.add("hidden");
        else
            elem.classList.remove("hidden");
        if (node.selected) {
            elem.classList.add("selected");
            node.element_text.classList.remove("hidden");
            // TODO: bb is zero-sized
            const bb = node.element_text.getBoundingClientRect();
            node.element_text.setAttribute("transform", `translate(${-bb.width/2},0)`);
            node.element_text.setAttribute("style", `fill: ${color}; stroke: black`);
            node.element_text.setAttribute("x", `${node.x}`);
            node.element_text.setAttribute("y", `${node.y}`);
        } else {
            elem.classList.remove("selected");
            elem.querySelector("text").classList.add("hidden");
        }
        node.element_circle.setAttribute("r", `${node.radius}`);
        node.element_circle.setAttribute("cx", `${node.x}`);
        node.element_circle.setAttribute("cy", `${node.y}`);
    };

    update_edge = (edge) => {
        const elem = edge.element;
        const color = this.edge_color(edge);
        let width = 1 + 3. * edge.weight;
        if (edge.selected)
            width *= 3.;
        elem.setAttribute(
            "style",
            `stroke: ${color}; fill: ${color}; stroke-width: ${width}px`
        );
        const [x1, y1, x2, y2, x3, y3, x4, y4] = this._get_edge_positions(edge);
        edge.element_line.setAttribute("x1", `${x1}`);
        edge.element_line.setAttribute("y1", `${y1}`);
        edge.element_line.setAttribute("x2", `${x2}`);
        edge.element_line.setAttribute("y2", `${y2}`);
        edge.element_polygon.setAttribute("points", `${x2},${y2} ${x3},${y3} ${x4},${y4}`);
    };

    update_selection = (
        selected_nodes, unselected_nodes,
        selected_edges, unselected_edges,
    ) => {
        for (const node of unselected_nodes) {
            this.update_node(node);
        }
        for (const edge of unselected_edges) {
            this.update_edge(edge);
        }
        for (const edge of selected_edges) {
            const elem = edge.element;
            const parent = elem.parentElement;
            parent.removeChild(elem);
            this.update_edge(edge);
            parent.appendChild(elem);
        }
        for (const node of selected_nodes) {
            const elem = node.element;
            const parent = elem.parentElement;
            parent.removeChild(elem);
            this.update_node(node);
            parent.appendChild(elem);
        }
    };

}


window.addEventListener("DOMContentLoaded", () => {

    const NODE_NUMBER_VALUES = [
        {key: "totalShares", label: "total shares"},
        {key: "totalHoldingsMillionDollar", label: "total holdings $"},
        {key: "pagerank", label: "graph PageRank"},
        {key: "hub", label: "graph hub"},
        {key: "authority", label: "graph authority"},
        {key: "hubOrAuthority", label: "max(graph hub, authority)"},
        {key: "x", label: "x position"},
        {key: "y", label: "y position"},
    ];

    const CONTROLS = {
        "radius": {
            type: "select", value: "totalHoldingsMillionDollar",
            label: "radius", title: "what value determines the radius of the spheres",
            choices: NODE_NUMBER_VALUES,
        },
        "min_radius": {
            type: "number", value: 10, min: 1, max: 1000,
            label: "min radius", title: "the smallest possible radius"
        },
        "max_radius": {
            type: "number", value: 50, min: 1, max: 1000,
            label: "max radius", title: "the largest possible radius"
        }
    };

    const diagram = new Diagram(document.getElementById("network"));
    window.diagram = diagram;
    let network = null;

    fetch("graph.json")
        .then(response => response.json())
        .then(data => {
            network = new Network(data.nodes, data.edges);
            window.network = network;
            const node = network.node_map["61322"];
            diagram.view.x = node.x;
            diagram.view.y = node.y;
            diagram.view.zoom = .2;
            diagram.set_network(network);
            on_control_change(CONTROLS["radius"], CONTROLS["radius"].value);
        });

    diagram.on_click = (x, y) => {
        if (!network)
            return;
        const node = network.node_at(x, y);
        console.log(diagram.view, node);
        if (node) {
            const nodes = network.traverse_nodes(node, 2);
            network.set_selected(nodes, 2);
        } else {
            network.set_selected([]);
        }
    };
    
    function update_controls() {
        const container = document.querySelector(".controls");
        for (const name of Object.keys(CONTROLS)) {
            const control = CONTROLS[name];
            control.name = name;
            if (!control.element) {
                if (control.type === "select") {
                    control.element = document.createElement("select");
                    control.element.setAttribute("name", control.name);
                    for (const opt of control.choices) {
                        const e = document.createElement("option");
                        e.setAttribute("value", opt.key);
                        if (opt.key === control.value)
                            e.setAttribute("selected", "selected");
                        e.innerText = opt.label;
                        control.element.appendChild(e);
                    }
                    control.element.addEventListener("change", e => {
                        on_control_change(control, control.element.value);
                    });
                } else {
                    control.element = document.createElement("input");
                    control.element.setAttribute("type", control.type);
                    control.element.setAttribute("name", control.name);
                    control.element.setAttribute("value", control.value);
                    if (control.min !== undefined)
                        control.element.setAttribute("min", control.min);
                    if (control.max !== undefined)
                        control.element.setAttribute("max", control.max);
                    control.element.addEventListener("change", e => {
                        on_control_change(control, control.element.value);
                    });
                }
                const wrapper = document.createElement("div");
                wrapper.setAttribute("class", "control");
                const label = document.createElement("div");
                const text = document.createTextNode(control.label);
                label.appendChild(text);
                wrapper.appendChild(label);
                wrapper.appendChild(control.element);
                container.appendChild(wrapper);
            }
        }
    }

    function on_control_change(control, value) {
        if (!network)
            return;

        if (control.type === "number")
            control.value = parseInt(value);
        else
            control.value = value;

        switch (control.name) {
            case "radius":
            case "min_radius":
            case "max_radius":
                network.update_radius(
                    CONTROLS["radius"].value,
                    CONTROLS["min_radius"].value,
                    CONTROLS["max_radius"].value,
                );
                diagram.update_all();
                break;
        }
    }
    
    update_controls();

});
