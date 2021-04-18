
document.addEventListener("DOMContentLoaded", () => {

    const label_types = ["things", "adjectives", "numbers", "prefixes"];

    const elem_labels = document.querySelector(".image-search .labels");
    const elem_results = document.querySelector(".image-search .results");
    const elem_show_images = document.querySelector(".image-search .cb-show-images");

    let top_matches = null;
    let label_map = null;
    const selected_labels = {
        prefixes: 5,
        numbers: 0,
        adjectives: 10,
        things: 13,
    }
    let selected_label_idx = 0;

    function update_images() {
        if (!top_matches)
            return;
        const
            top_urls = top_matches[selected_label_idx],
            show_images = elem_show_images.checked;

        let html = ``;
        for (const url of top_urls) {
            html += `<div>`;
            if (show_images) {
                html += `<img src="${url[2]}" style="max-height: 512px; background: #aaa;">`;
            }
            html += `<p><b style="width: 3rem; display: inline-block">${Math.round(url[0] * 100) / 100}</b> `;
            html += `<a href="${url[2]}" title="${url[2]}" rel="noreferrer">${url[1]}</a>`;
            html += `</p></div>`;
        }
        elem_results.innerHTML = html;
    };

    function get_type_label_name(type, i) {
        let label = label_map[type][i]
        const is_selected = selected_labels[type] === i;
        if (type === "numbers")
            label = label[0];
        if (type === "things")
            label = label[label_map.numbers[selected_labels.numbers][1]];
        return label;
    }

    function update_labels() {
        const label_text = label_types
            .map(type => get_type_label_name(type, selected_labels[type]))
            .reverse()
            .join("/");

        selected_label_idx = label_map.index.indexOf(label_text);

        render_labels();
        if (top_matches)
            update_images();
    }

    function render_labels() {
        let labels_html = ``;
        let label_idx = 0;
        for (const type of label_types) {
            let html = `<select id="label-${type}" data-type="${type}" data-offset="${label_idx}">`;
            for (let i=0; i<label_map[type].length; ++i) {
                const is_selected = selected_labels[type] === i;
                const label = get_type_label_name(type, i);
                html += `<option value="${i}" ${is_selected ? "selected" : ""}>${label}</option>`;
                label_idx += 1;
            }
            html += `</select>`;

            labels_html = html + labels_html;
        }
        elem_labels.innerHTML = labels_html;
        for (const elem of document.querySelectorAll(".labels")) {
            elem.onchange = (e) => {
                const
                    type = e.target.getAttribute("data-type"),
                    value = parseInt(e.target.value);

                selected_labels[type] = value;
                update_labels();
            };
        }
    }

    elem_labels.innerHTML = `<i>loading...</i>`;
    fetch('/blog/assets/data/clip/label-features-4-map.json')
        .then(response => response.json())
        .then(data => {
            label_map = data;
            update_labels();
        });


    elem_results.innerHTML = `<i>loading...</i>`;
    fetch('/blog/assets/data/clip/top-matches-4.json')
        .then(response => response.json())
        .then(data => {
            top_matches = data;
            update_images();
        });

    elem_show_images.addEventListener("change", update_images);


});