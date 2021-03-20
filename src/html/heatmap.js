
const heatmap_data___id__ = __data__;

let heatmap_filters___id__ = {x: "", y: ""};


function render_heatmap___id__() {
    const data = heatmap_data___id__;
    const num_colors = __num_colors__;
    
    let index_x = [], index_y = [];
    for (let x=0; x<data.labels_x.length; ++x) {
        index_x.push(x);
    }
    for (let y=0; y<data.labels_y.length; ++y) {
        index_y.push(y);
    }
    let min_val = 0., max_val = 0.;
    for (const y of index_y) {
        const row = data.matrix[y];
        for (const x of index_x) {
            let value = row[x];
            min_val = Math.min(min_val, value);
            max_val = Math.max(max_val, value);
        }
    }
    const color_factor = min_val === max_val ? 0. : (num_colors) / (max_val - min_val);

    let html = "";

    html += `<div class="hmlabelv"></div>`;
    for (const x of index_x) {
        const label = data.labels_x[x];
        html += `<div class="hmlabelv" title="${label}">${label}</div>`;
    }

    for (const y of index_y) {
        const row = data.matrix[y];

        const label = data.labels_y[y];
        html += `<div class="hmlabel" title="${label}">${label}</div>`;

        for (const x of index_x) {
            const
                value = row[x],
                col = Math.min(__num_colors__ - 1, Math.floor((value - min_val) * color_factor));

            const title = `${data.labels_x[x]} / ${data.labels_y[y]}: ${value}`;
            html += `<div class="hmc v${col}" title="${title}"></div>`;
        }
    }

    html += `<div class="hmlabelvb"></div>`;
    for (const x of index_x) {
        const label = data.labels_x[x];
        html += `<div class="hmlabelvb" title="${label}">${label}</div>`;
    }

    const elem = document.querySelector(".heatmap-__id__");
    elem.style = `grid-template-columns: __label_width__ repeat(${index_x.length}, 1fr);`;
    elem.innerHTML = html;
}

render_heatmap___id__();
