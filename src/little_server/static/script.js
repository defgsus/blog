
document.addEventListener("DOMContentLoaded", function () {

    const
        body = document.querySelector("body"),
        websocket_url = body.getAttribute("data-ws-url"),
        status_elem = document.querySelector(".ls-header .ls-header-status");

    let ws = null;
    let refresh_timeout = null;
    let dom_cell_container = document.querySelector(".ls-cells");
    let dom_cell_elements = {};
    let dom_cell_hashes = {};

    connect_ws();

    function connect_ws() {
        ws = new WebSocket(websocket_url);
        ws.onopen = on_ws_open;
        ws.onclose = on_ws_close;
        ws.onmessage = on_ws_message;
    }

    function try_reconnect_ws() {
        if (refresh_timeout)
            clearTimeout(refresh_timeout);
        refresh_timeout = setTimeout(connect_ws, 2000);
    }

    function send_ws_message(name, data) {
        const message = {
            name: name,
            data: data,
        };
        ws.send(JSON.stringify(message));
    }

    function on_ws_open() {
        status_elem.innerText = "connected";
        status_elem.classList.add("ls-connected");
        send_ws_message("dom-loaded");
    }

    function on_ws_close() {
        status_elem.innerText = "not connected";
        status_elem.classList.remove("ls-connected");
        try_reconnect_ws();
    }

    function on_ws_message(event) {
        const
            message = JSON.parse(event.data),
            name = message.name,
            data = message.data;

        switch (name) {
            case "cell": render_cell(data); break;
            case "log": cell_log(data); break;
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
        if (cell.log)
            css_classes += " ls-cell-log";
        if (cell.images && cell.images.length) {
            css_classes += " ls-cell-image";
            if (!cell.fit)
                css_classes += " ls-cell-scrollable";
        }

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

        if (cell.images) {
            for (const image of cell.images) {
                html += `<img src="${image}"`;
                if (cell.fit) {
                    html += `width="100%" height="${100 / cell.images.length}%"`;
                }
                html += `>`;
            }
        }

        if (cell.text) {
            for (const line of cell.text.split("\n"))
                html += `<p>${line}</p>`;
        }
        if (cell.code) {
            html += `<pre>${cell.code}</pre>`;
        }
        if (cell.log) {
            html += `<textarea contenteditable="false">${cell.log}</textarea>`;
        }

        if (cell.actions) {
            for (const a of cell.actions) {
                html += `<button class="action-button" data-action="${a.id}">${a.name}</button>`;
            }
        }

        return html;
    }

    function cell_log(data) {
        if (dom_cell_elements[data.name]) {
            const cell_elem = dom_cell_elements[data.name];
            const textarea = cell_elem.querySelector("textarea");
            if (textarea) {
                const at_bottom = textarea.scrollHeight - textarea.scrollTop <= textarea.clientHeight + 100;

                if (!textarea.textContent.endsWith("\n"))
                    textarea.textContent += "\n";
                textarea.textContent += data.log;
                if (at_bottom)
                    textarea.scrollTop = textarea.scrollHeight;
            }
        }
    }

    function on_action_click(event) {
        const action = event.target.getAttribute("data-action");
        send_ws_message("action", {name: action});
    }
});