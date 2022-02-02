window.addEventListener("DOMContentLoaded", () => {

    const container = document.getElementById("network");
    const network = new vis.Network(
        container,
        {},
        {
            autoResize: true,
            width: "100%",
            height: `${window.innerHeight-30}px`,
            layout: {
                improvedLayout: true,
                randomSeed: 23,
            },
            nodes: {
                shape: "dot",
                font: {
                    color: "#ccc",
                }
                //widthConstraint: {maximum: 25},
                //heightConstraint: {maximum: 25},
            },
            edges: {
                arrows: "to",
                arrowStrikethrough: false,
                color: {inherit: "to"},
                scaling: {
                    min: 2,
                    max: 10,
                }
            },
            physics: {
                barnesHut: {
                    springLength: 400,
                },
                stabilization: {
                    iterations: 300,
                }
            }
        },
    );
    //network.on("click", on_network_click);
    let full_network_data;
    const network_data = {
        nodes: new vis.DataSet(),
        edges: new vis.DataSet(),
    };
    network.setData(network_data);

    document.querySelector('button[name="stop"]').addEventListener("click", e => {
        network.stopSimulation();
    });

    fetch("../layouted.dot")
        .then(response => response.text())
        .then(dot_string => {
            try {
                //return vis.network.convertGephi(text)
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
            full_network_data = data;
            update_network();
        });

    function setup_data_pipes() {
        /* TODO: mhhh, function does not seem to be part of package */
        const node_pipe = vis.createNewDataPipeFrom(full_network_data.nodes)
            .filter(item => item.total_holdings_dollar > 10000000)
            .to(network_data.nodes);
    }

    function update_network() {
        const nodes = full_network_data.nodes.filter(n => (
            n.totalHoldingsDollar > 1000000 * 10000
        ));
        const node_new_id = {};
        for (let i=0; i<nodes.length; ++i) {
            node_new_id[nodes[i].id] = i;
        }
        //const node_set = new Set(nodes.map(n => n.id));

        network_data.nodes.add(
            nodes.map(n => {
                return {
                    ...n,
                    x: n.x * 1500,
                    y: n.y * 1500,
                    color: {
                        background: "crimson",
                        border: "gray",
                        highlight: "white",
                    },
                };
            })
        );

        network_data.edges.add(
            full_network_data.edges
                .filter(edge =>
                    node_new_id[edge.from] !== undefined && node_new_id[edge.to] !== undefined
                )
                .map(edge => {
                    const
                        new_from = node_new_id[edge.from],
                        new_to = node_new_id[edge.to];
                    return {
                        ...edge,
                        from: new_from,
                        to: new_to,
                    }
                })
        );

    }
});