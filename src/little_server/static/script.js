
document.addEventListener("DOMContentLoaded", function () {

    const
        body = document.querySelector("body"),
        websocket_url = body.getAttribute("data-ws-url");

    let ws = new WebSocket(websocket_url);
    ws.onopen = on_ws_open;
    ws.onmessage = on_ws_message;

    let refresh_timeout = setTimeout(refresh, 300);
    let dom_cell_container = document.querySelector(".ls-cells");
    let dom_cell_elements = {};
    let dom_cell_hashes = {};

    function refresh() {
        /*fetch("cells/")
            .then(response => response.json())
            .then(response => render_cells(response["cells"]))
        ;*/

        //console.log(Math.random());

        if (refresh_timeout)
            clearTimeout(refresh_timeout);
        refresh_timeout = setTimeout(refresh, 1000)
    }

    function send_ws_message(name, data) {
        const message = {
            name: name,
            data: data,
        };
        ws.send(JSON.stringify(message));
    }

    function on_ws_open() {
        send_ws_message("dom-loaded");
    }

    function on_ws_message(event) {
        const
            message = JSON.parse(event.data),
            name = message.name,
            data = message.data;

        switch (name) {
            case "cell": render_cell(data); break;
            default:
                console.log("unhandled message", name, data);
        }
    }

    function render_cells(cells) {
        for (const cell of cells) {
            render_cell(cell);
        }
    }

    function render_cell(cell) {
        if (dom_cell_hashes[cell.name] === cell.hash)
            return;

        if (!dom_cell_elements[cell.name]) {
            const cell_elem = document.createElement("div");
            cell_elem.className = "ls-cell";
            dom_cell_container.appendChild(cell_elem);
            dom_cell_elements[cell.name] = cell_elem;
        }
        const cell_elem = dom_cell_elements[cell.name];

        let style = "";
        if (cell.column)
            style += `grid-column: ${cell.column};`;
        if (cell.row)
            style += `grid-row: ${cell.row};`;

        let css_classes = "ls-cell";
        if (cell.code)
            css_classes += " ls-cell-code";

        cell_elem.innerHTML = render_cell_html(cell);
        cell_elem.setAttribute("style", style);
        cell_elem.setAttribute("id", `ls-cell-${cell.name}`);
        cell_elem.className = css_classes;

        dom_cell_hashes[cell.name] = cell.hash;
        for (const b of cell_elem.querySelectorAll("button.action-button")) {
            b.onclick = on_action_click;
        }
    }

    function render_cell_html(cell) {
        let html = ``;

        if (cell.image) {
            html += `<img src="${cell.image}">`; // width="${cell.width}px" height="${cell.height}px">`;
        }

        if (cell.text) {
            for (const line of cell.text.split("\n"))
                html += `<p>${line}</p>`;
        }
        if (cell.code) {
            html += `<pre>${cell.code}</pre>`;
        }

        if (cell.actions) {
            for (const a of cell.actions) {
                html += `<button class="action-button" data-action="${a.id}">${a.name}</button>`;
            }
        }

        return html;
    }

    function on_action_click(event) {
        const action = event.target.getAttribute("data-action");
        fetch(`action/?a=${action}`, {method: "post"})
            .then(response => response.json())
            //.then(response => render_cells(response["cells"]))
        ;
    }
});