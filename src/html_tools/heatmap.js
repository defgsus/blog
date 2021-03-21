(function() {

    const
        heatmap_data___id__ = __data__,
        min_cells_x = __min_cells_x__,
        max_cells_y = __max_cells_y__;

    let heatmap_filters = __filters__;
    let render_timeout = null;

    for (const dim of "xy") {
        let elem = document.querySelector(".heatmap-filters-__id__ input.filter-" + dim);
        if (elem) {
            elem.addEventListener("input", function (e) {
                heatmap_filters[dim] = e.target.value;
                render_heatmap_lazy();
            });
        }
    }

    document.addEventListener("load", function () {
        render_heatmap();
    });

    function render_heatmap_lazy() {
        if (render_timeout)
            window.clearTimeout(render_timeout);

        render_timeout = window.setTimeout(render_heatmap, 250);
    }

    function filter_labels_to_index(labels, filter, is_x) {
        let index = [];
        for (let x=0; x < labels.length; ++x)
            index.push(x);

        if (filter && filter.length) {
            const filters = filter.split(",").map(function(f) { return new RegExp(f.trim()); });
            index = index.filter(function(i) {
                for (const f of filters) {
                    if (labels[i].match(f))
                        return true;
                }
            });
        }

        let need_extra_space = false;
        if (is_x) {
            if (index.length < min_cells_x)
                need_extra_space = true;
        } else {
            if (max_cells_y && index.length > max_cells_y) {
                index = index.slice(0, max_cells_y);
            }
        }

        return [index, need_extra_space];
    }

    function render_heatmap() {
        const data = heatmap_data___id__;
        const num_colors = __num_colors__;

        let index_x, index_y, need_extra_space;
        [index_y, need_extra_space] = filter_labels_to_index(data.labels_y, heatmap_filters.y);
        [index_x, need_extra_space] = filter_labels_to_index(data.labels_x, heatmap_filters.x, true);

        const render_index_x = need_extra_space
            ? index_x.concat([data.labels_x.length])
            : index_x;

        let min_val = 0., max_val = 0.;
        for (const y of index_y) {
            const row = data.matrix[y];
            for (const x of index_x) {
                let value = row[x];
                if (typeof value === "number" && !isNaN(value)) {
                    min_val = Math.min(min_val, value);
                    max_val = Math.max(max_val, value);
                }
            }
        }
        const color_factor = min_val === max_val ? 0. : (num_colors) / (max_val - min_val);

        let html = "";

        function render_x_labels(class_name) {
            html += `<div class="${class_name}"></div>`;
            for (const x of render_index_x) {
                const label = data.labels_x[x];
                if (label)
                    html += `<div class="${class_name}" title="${label}">${label}</div>`;
                else
                    html += `<div class="${class_name}"></div>`;
            }
        }
        render_x_labels("hmlabelv");

        for (const y of index_y) {
            const row = data.matrix[y];

            const label = data.labels_y[y];
            html += `<div class="hmlabel" title="${label}">${label}</div>`;

            for (const x of render_index_x) {
                if (x < data.labels_x.length) {
                    const
                        value = row[x],
                        title = `${data.labels_x[x]} / ${data.labels_y[y]}: ${value}`;

                    let col = "empty";
                    if (typeof value === "number" && !isNaN(value))
                        col = Math.min(__num_colors__ - 1, Math.floor((value - min_val) * color_factor));

                    html += `<div class="hmc hmc-${col}" title="${title}"></div>`;
                }
                else {
                    html += `<div class="hmc hmc-overlap"></div>`;
                }
            }
        }

        render_x_labels("hmlabelvb");

        let elem = document.querySelector(".heatmap-filters-__id__ .hm-dimensions");
        let text = `dimensions: ${data.labels_x.length} x ${data.labels_y.length}`;
        if (index_x.length !== data.labels_x.length || index_y.length !== data.labels_y.length)
            text += ` (display: ${index_x.length} x ${index_y.length})`;
        elem.textContent = text;

        elem = document.querySelector(".heatmap-__id__");
        let columns = `__label_width__ repeat(${index_x.length}, 1fr)`;
        if (need_extra_space)
            columns += ` ${min_cells_x - index_x.length + 1}fr`;
        elem.style = `grid-template-columns: ${columns}`;
        elem.innerHTML = html;
    }

    render_heatmap();

})();
