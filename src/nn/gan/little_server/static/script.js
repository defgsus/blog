
document.addEventListener("DOMContentLoaded", function () {

    let refresh_timeout = setTimeout(refresh, 300);
    let dom_cell_container = document.querySelector(".cells");
    let dom_cell_elements = {};
    let dom_cell_hashes = {};

    function refresh() {
        fetch("cells/")
            .then(response => response.json())
            .then(response => render_cells(response["cells"]))
        ;

        //console.log(Math.random());

        if (refresh_timeout)
            clearTimeout(refresh_timeout);
        refresh_timeout = setTimeout(refresh, 1000)
    }

    function render_cells(cells) {
        for (const cell of cells) {
            if (dom_cell_hashes[cell.name] === cell.hash)
                continue;

            if (!dom_cell_elements[cell.name]) {
                const cell_elem = document.createElement("div");
                cell_elem.className = "cell";
                dom_cell_container.appendChild(cell_elem);
                dom_cell_elements[cell.name] = cell_elem;
            }
            const cell_elem = dom_cell_elements[cell.name];

            let html = ``;
            if (cell.image) {
                html += `<img src="${cell.image}?${cell.hash}">`;
            }
            if (cell.text) {
                html += `<p>${cell.text}</p>`;
            }

            cell_elem.innerHTML = html;
            dom_cell_hashes[cell.name] = cell.hash;
        }
    }
});