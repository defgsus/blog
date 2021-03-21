(function() {

    const
        heatmap_data___id__ = __data__,
        max_label_length = __max_label_length__,
        keep_label_front = __keep_label_front__,
        min_cells_x = __min_cells_x__,
        max_cells_x = __max_cells_x__,
        max_cells_y = __max_cells_y__;

    let heatmap_filters = __filters__;

    function get_filter_elem(dim, what, sub_element) {
        const axis_selector = `.heatmap-__id__ .heatmap-filters-axis-${dim}`;
        return document.querySelector(`${axis_selector} .heatmap-filter-cell-${what} ${sub_element || ""}`);
    }

    function get_filters(dim) {
        let elem = get_filter_elem(dim, "string", "input");
        if (elem)
            heatmap_filters[dim] = elem.value;
        elem = get_filter_elem(dim, "empty", "input");
        if (elem)
            heatmap_filters[`empty_${dim}`] = elem.checked;
        elem = get_filter_elem(dim, "page", "input");
        if (elem)
            heatmap_filters[`page_${dim}`] = Math.max(0, parseInt(elem.value) - 1);
    }

    for (const dim of "xy") {
        let elem = get_filter_elem(dim, "string", "input");
        if (elem)
            elem.addEventListener("input", render_heatmap_lazy);
        elem = get_filter_elem(dim, "empty", "input");
        if (elem)
            elem.addEventListener("change", render_heatmap_lazy);
        elem = get_filter_elem(dim, "page", "input");
        if (elem)
            elem.addEventListener("change", render_heatmap_lazy);
    }

    document.addEventListener("load", function () {
        render_heatmap();
    });

    let render_timeout = null;

    function render_heatmap_lazy(delay=250) {
        if (render_timeout)
            window.clearTimeout(render_timeout);

        render_timeout = window.setTimeout(render_heatmap, delay);
    }

    function filter_labels_to_index(labels, filter) {
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

        return index;
    }

    function filter_index_empty_cells(matrix, index_x, index_y) {
        if (!heatmap_filters.empty_y) {
            let new_index_y = [];
            for (const y of index_y) {
                const row = matrix[y];
                let has_number = false;
                for (const x of index_x) {
                    let value = row[x];
                    if (typeof value === "number" && !isNaN(value)) {
                        has_number = true;
                        break;
                    }
                }
                if (has_number)
                    new_index_y.push(y);
            }
            index_y = new_index_y;
        }
        if (!heatmap_filters.empty_x) {
            let index_x_has_number = {};
            for (const y of index_y) {
                const row = matrix[y];
                for (const x of index_x) {
                    let value = row[x];
                    if (typeof value === "number" && !isNaN(value))
                        index_x_has_number[x] = true;
                }
            }
            index_x = index_x.filter(function(x) { return index_x_has_number[x]; });
        }

        return [index_x, index_y];
    }

    function paginate(dim, index, per_page) {

        let cur_page = heatmap_filters[`page_${dim}`],
            num_pages = Math.ceil(index.length / per_page);

        get_filter_elem(dim, "page", ".heatmap-page-count").textContent = `${num_pages}`;
        const input_elem = get_filter_elem(dim, "page", "input");
        input_elem.setAttribute("max", num_pages);
        cur_page = Math.max(0, Math.min(cur_page, num_pages - 1));

        if (cur_page !== heatmap_filters[`page_${dim}`]) {
            input_elem.value = cur_page + 1;
            heatmap_filters[`page_${dim}`] = cur_page;
        }

        const page_offset = cur_page * per_page;
        return index.slice(page_offset, page_offset + per_page);
    }

    function limit_label_length(text) {
        if (text && text.length && text.length > max_label_length) {
            if (keep_label_front) {
                text = text.slice(text.length - max_label_length, text.length);
                text = ".." + text;
            }
            else {
                text = text.slice(0, max_label_length);
                text = text + "..";
            }
        }
        return text;
    }

    function render_heatmap() {
        const data = heatmap_data___id__;
        const num_colors = __num_colors__;

        get_filters("x");
        get_filters("y");
        console.log(heatmap_filters);

        let
            index_x = filter_labels_to_index(data.labels_x, heatmap_filters.x),
            index_y = filter_labels_to_index(data.labels_y, heatmap_filters.y);

        [index_x, index_y] = filter_index_empty_cells(data.matrix, index_x, index_y);

        const filtered_size = [index_x.length, index_y.length];

        index_x = paginate("x", index_x, max_cells_x);
        index_y = paginate("y", index_y, max_cells_y);

        const need_extra_space = (index_x.length < min_cells_x);

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
                    html += `<div class="${class_name}" title="${label}">${limit_label_length(label)}</div>`;
                else
                    html += `<div class="${class_name}"></div>`;
            }
        }
        render_x_labels("hmlabelv");

        for (const y of index_y) {
            const row = data.matrix[y];

            const label = data.labels_y[y];
            html += `<div class="hmlabel" title="${label}">${limit_label_length(label)}</div>`;

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

        let elem = document.querySelector(".heatmap-__id__ .heatmap-dimensions");
        if (elem) {
            let text = `dimensions: ${data.labels_x.length} x ${data.labels_y.length}`;
            if (filtered_size[0] !== data.labels_x.length || filtered_size[1] !== data.labels_y.length)
                text += `, filtered: ${filtered_size[0]} x ${filtered_size[1]}`;
            if (index_x.length !== data.labels_x.length || index_y.length !== data.labels_y.length)
                text += `, display: ${index_x.length} x ${index_y.length}`;
            elem.textContent = text;
        }

        elem = document.querySelector(".heatmap-__id__ .heatmap-grid");
        let columns = `__label_width__ repeat(${index_x.length}, 1fr)`;
        if (need_extra_space)
            columns += ` ${min_cells_x - index_x.length + 1}fr`;
        elem.style = `grid-template-columns: ${columns}`;
        elem.innerHTML = html;
    }

    render_heatmap();

})();
