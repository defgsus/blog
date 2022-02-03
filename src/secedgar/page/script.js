class Network {

    constructor(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
        this.node_map = {};
        for (const n of this.nodes) {
            n.x *= 1000.;
            n.y *= 1000.;
            n.hidden = false;
            n.selected = false;
            n.alpha = .3;
            n.radius = 10. + n.hubOrAuthority * 30.;
            this.node_map[n.id] = n;
        }
        this.selected_nodes = [];
        this.on_selection_changed = (selected_nodes, unselected_nodes) => {};
    }

    node_at = (x, y) => {
        for (const n of this.nodes) {
            const r2 = (n.x - x) * (n.x - x) + (n.y - y) * (n.y - y);
            if (r2 < n.radius * n.radius)
                return n;
        }
    };

    set_selected = (nodes_or_ids) => {
        const selected_ids = new Set(nodes_or_ids.map(n => (
            typeof n === "object" ? n.id : n
        )));
        const
            selected_nodes = [],
            unselected_nodes = [];
        for (const node of this.selected_nodes) {
            if (!selected_ids.has(node.id)) {
                unselected_nodes.push(node);
                node.selected = false;
                node.alpha = .3;
            }
        }
        for (const node_id of selected_ids) {
            const node = this.node_map[node_id];
            node.selected = true;
            node.alpha = 1.;
            selected_nodes.push(node)
        }
        this.selected_nodes = selected_nodes;
        this.on_selection_changed(selected_nodes, unselected_nodes);
    };
}


class Diagram {
    constructor(element) {
        this.element = element;
        this.view = {
            x: 0,
            y: 0,
            scale: 10.
        };
        this.network = null;
        this.on_click = (x, y) => {};

        this.element.addEventListener("click", e => {
            const [x, y] = this.event_to_world_coords(e);
            this.on_click(x, y);
        });

        this._last_mouse_down = [0, 0];
        this._last_mouse_down_view = [0, 0];
        this.element.addEventListener("mousedown", e => {
            this._last_mouse_down = [e.clientX, e.clientY];
            this._last_mouse_down_view = [this.view.x, this.view.y];
        });
        this.element.addEventListener("mousemove", e => {
            if (!e.buttons)
                return;
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

    width = () => this.element.boundingBox.width();
    height = () => this.element.boundingBox.height();

    event_to_world_coords = (event) => {
        const
            bb = this.element.getBoundingClientRect(),
            elem_x = event.clientX - bb.left,
            elem_y = event.clientY - bb.top,
            x = (elem_x - bb.width/2) / this.view.zoom + this.view.x,
            y = (elem_y - bb.height/2) / this.view.zoom + this.view.y;
        return [x, y];
    };

    node_color = (node) => {
        return `rgba(${node.hub*255},${node.authority*255},0,${node.alpha})`;
    };

    set_network = (network) => {
        this.network = network;
        this.network.on_selection_changed = this.update_selection;

        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("class", "world-transform");
        for (const edge of this.network.edges) {
            const
                node_from = this.network.node_map[edge.from],
                node_to = this.network.node_map[edge.to];
            const elem = document.createElementNS("http://www.w3.org/2000/svg", "line");
            elem.setAttribute("class", `edge edge-${edge.id}`);
            elem.setAttribute("x1", `${node_from.x}`);
            elem.setAttribute("y1", `${node_from.y}`);
            elem.setAttribute("x2", `${node_to.x}`);
            elem.setAttribute("y2", `${node_to.y}`);
            elem.setAttribute("style", `stroke: ${this.node_color(node_to)}`);
            g.appendChild(elem);
        }
        for (const node of this.network.nodes) {
            const node_g = document.createElementNS("http://www.w3.org/2000/svg", "g");
            node_g.setAttribute("class", `node node-${node.id}`);
            let elem = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            node_g.appendChild(elem);
            elem.setAttribute("r", `${node.radius}`);
            elem.setAttribute("cx", `${node.x}`);
            elem.setAttribute("cy", `${node.y}`);
            elem.setAttribute("style", `fill: ${this.node_color(node)}`);
            elem = document.createElementNS("http://www.w3.org/2000/svg", "text");
            node_g.appendChild(elem);
            elem.setAttribute("x", `${node.x}`);
            elem.setAttribute("y", `${node.y}`);
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
        for (const elem of this.element.querySelectorAll(".node")) {
            const node = this.network.node_map[parseInt(elem.id.slice(5))];
            this.update_node(elem, node);
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

    update_node = (node, elem) => {
        if (!elem)
            elem = this.element.querySelector(`.node-${node.id}`);
        if (node.hidden)
            elem.classList.add("hidden");
        else
            elem.classList.remove("hidden");
        if (node.selected) {
            elem.classList.add("selected");
            const text = elem.querySelector("text");
            text.classList.remove("hidden");
            // TODO: bb is zero-sized
            const bb = text.getBoundingClientRect();
            text.setAttribute("transform", `translate(${-bb.width/2},0)`);
            text.setAttribute("style", `fill: ${this.node_color(node)}; stroke: black`);
        } else {
            elem.classList.remove("selected");
            elem.querySelector("text").classList.add("hidden");
        }
        const shape = elem.querySelector("circle");
        shape.setAttribute("r", `${node.radius}`);
        shape.setAttribute("cx", `${node.x}`);
        shape.setAttribute("cy", `${node.y}`);
        shape.setAttribute("style", `fill: ${this.node_color(node)}`);
    };

    update_selection = (selected_nodes, unselected_nodes) => {
        for (const node of unselected_nodes) {
            this.update_node(node);
        }
        for (const node of selected_nodes) {
            const elem = this.element.querySelector(`.node-${node.id}`);
            const parent = elem.parentElement;
            parent.removeChild(elem);
            this.update_node(node, elem);
            parent.appendChild(elem);
        }
    };

}


window.addEventListener("DOMContentLoaded", () => {

    const diagram = new Diagram(document.getElementById("network"));
    window.diagram = diagram;
    diagram.view = {
        x: 10270., y: 2250., zoom: 0.28,
    };
    let network = null;

    diagram.on_click = (x, y) => {
        if (!network)
            return;
        const node = network.node_at(x, y);
        console.log(diagram.view, node);
        if (node)
            network.set_selected([node.id]);
    };

    fetch("graph.json")
        .then(response => response.json())
        .then(data => {
            network = new Network(data.nodes, data.edges);
            diagram.set_network(network);
        });

});
