"""

https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-69

"""
import json
import time
from typing import Union, Iterable, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
import igraph

from edgar import Edgar, Company, Filing
from nasdaq import NasdaqPublic


edgar = Edgar()
nasdaq = NasdaqPublic()


def get_all_transactions(company: Company) -> pd.DataFrame:
    transactions = []
    for f in tqdm(company.filings("4")):
        for i, t in enumerate(f.transactions()):
            t["id"] = f"{f.accessionNumber}-{i}"
            for key, value in f.reporter().items():
                t[f"reporter_{key}"] = value
            transactions.append(t)

    df = pd.DataFrame(transactions).set_index("id")
    df["date"] = pd.to_datetime(df["date"])
    return df


def store_transactions(company_name: str):
    c = edgar.get_company(company_name)
    df = get_all_transactions(c)
    df.to_csv(f"{company_name}.csv")
    print(df)


def main():


    c = edgar.get_company("blackrock")
    for f in c.filings():
        if f.form in ("3", "4", "5"):
            print("\n")
            print(f)
            if f.form == "4":
                # print(json.dumps(f.to_dict(), indent=2))
                print(f.issuer(), f.reporter(), f.signature())
                print(f.transactions())


def search_company(query: str):
    for cik, name, ticker, exchange in edgar.company_tickers_exchange["data"]:
        if query in name.lower():
            print(ticker, name)

#search_company("vanguard")


def to_int(x: Union[int, str]) -> int:
    if isinstance(x, str):
        if not x:
            return 0
        return int(x.replace(",", ""))
    elif isinstance(x, int):
        return x
    raise TypeError(f"Got '{type(x).__name__}'")


def get_path(data: Optional[dict], path: str):
    path = path.split(".")
    while path:
        if data is None:
            return None
        key = path.pop(0)
        data = data.get(key)
    return data


def get_nasdaq_holder_graph(
        filename: str,
        start_symbols: Iterable[Union[int, str]],
        scan_limit: int = 100,
        request_limit: int = 100,
        min_shares_dollar: int = 1_000_000,
        max_distance: int = 7,
        edge_weight_gte: float = 0.,
):
    todo = {
        (key, 0)
        for key in start_symbols
    }
    done = set()

    holdings_map = dict()
    profile_map = dict()
    vertex_map = dict()
    edges = dict()

    start_time = last_time = time.time()
    while todo:
        pick_entry = sorted(todo, key=lambda e: str(e[0]))[0]
        company_id, distance = pick_entry
        todo.remove(pick_entry)

        cur_time = time.time()
        if cur_time - last_time >= 1.:
            last_time = cur_time
            print(
                f"{cur_time - start_time:.0f} sec"
                f", todo {len(todo)}"
                f", done {len(done)}"
                f", edges {len(edges)}"
            )

        if isinstance(company_id, str):
            company_id = company_id.lower()

        holdings = holdings_map[company_id] = nasdaq.institutional_holdings(
            company_id, limit=request_limit,
        )
        done.add(company_id)

        if isinstance(company_id, str):
            company_id = company_id.lower()
            profile_map[company_id] = nasdaq.company_profile(company_id)

        if company_id not in vertex_map:
            vertex_map[company_id] = len(vertex_map)

        if distance > max_distance:
            continue

        try:
            if holdings.get("data") and holdings["data"].get("institutionPositions"):
                assert holdings["data"]["institutionPositions"]["table"]["headers"]["value"] == "Value ($1,000s)"

                for row in holdings["data"]["institutionPositions"]["table"]["rows"][:scan_limit]:
                    # can not map these to nasdaq endpoints :-(
                    if not row["url"]:
                        continue

                    held_company_id = row["url"].split("/")[3].lower()

                    shares_dollar = to_int(row["value"]) * 1000
                    if shares_dollar < min_shares_dollar:
                        continue

                    if (company_id, held_company_id) not in edges:
                        edges[(company_id, held_company_id)] = {
                            "shares": to_int(row["sharesHeld"]),
                            "sharesDollar": shares_dollar,
                        }

                    if distance <= max_distance and held_company_id not in done:
                        # print(len(todo), distance, company_id, held_company_id)
                        todo.add((held_company_id, distance + 1))

            if holdings.get("data") and holdings["data"].get("holdingsTransactions"):
                assert holdings["data"]["holdingsTransactions"]["table"]["headers"]["marketValue"] == "VALUE (IN 1,000S)"

                for row in holdings["data"]["holdingsTransactions"]["table"]["rows"][:scan_limit]:

                    shares_dollar = to_int(row["marketValue"][1:]) * 1000
                    if shares_dollar < min_shares_dollar:
                        continue

                    holder_company_id = int(row["url"].split("-")[-1])
                    if holder_company_id not in vertex_map:
                        vertex_map[holder_company_id] = len(vertex_map)

                    if (holder_company_id, company_id) not in edges:
                        edges[(holder_company_id, company_id)] = {
                            "shares": to_int(row["sharesHeld"]),
                            "sharesDollar": shares_dollar,
                        }

                    if holder_company_id not in done:
                        # print(len(todo), distance, holder_company_id, company_id)
                        todo.add((holder_company_id, distance + 1))

        except:
            print(json.dumps(holdings, indent=2))
            raise

    graph = igraph.Graph(directed=True)

    EMPTY = "none"
    for company_id in sorted(vertex_map, key=lambda id: vertex_map[id]):
        attributes = {
            "name": str(company_id),
            "symbol": str(company_id),
            "totalShares": 0,
            "totalHoldingsDollar": 0,
            "sector": EMPTY,
            "region": EMPTY,
            "industry": EMPTY,
        }

        if isinstance(company_id, str):
            attributes["type"] = "company"
            profile = profile_map[company_id]
            holdings = holdings_map[company_id]
            if profile["status"]["rCode"] != 400:
                try:
                    attributes.update({
                        "name": profile["data"]["CompanyName"]["value"],
                        "sector": profile["data"]["Sector"]["value"] or EMPTY,
                        "region": profile["data"]["Region"]["value"] or EMPTY,
                        "industry": profile["data"]["Industry"]["value"] or EMPTY,
                    })
                except:
                    raise ValueError(f"'{company_id}' in {profile}")

                value = get_path(holdings, "data.ownershipSummary.TotalHoldingsValue")
                if value:
                    assert value["label"].endswith("(millions)"), value["label"]
                    attributes["totalHoldingsDollar"] = to_int(value["value"][1:]) * 1_000_000

                rows = get_path(holdings, "data.activePositions.rows")
                if rows:
                    assert rows[-1]["positions"] == "Total Institutional Shares", rows
                    attributes["totalShares"] = to_int(rows[-1]["shares"])

        else:
            holdings = holdings_map[company_id]

            attributes.update({
                "type": "holder",
                "name": holdings["data"]["title"],
            })

            value = get_path(holdings, "data.positionStatistics.TotalMktValue")
            if value:
                assert value["label"].endswith("($MILLIONS)"), value["label"]
                attributes["totalHoldingsDollar"] = to_int(value["value"]) * 1_000_000

        attributes.update({
            "label": attributes["name"],
        })
        graph.add_vertex(**attributes)

    # -- convert edges to igraph --

    igraph_edges = []
    igraph_edge_attrs = {
        "weight": [],
        "shares": [],
        "sharesDollar": [],
    }
    for edge, attrs in edges.items():
        total_shares = graph.vs["totalShares"][vertex_map[edge[1]]]
        weight = attrs["shares"] / total_shares if total_shares else 0.01
        igraph_edges.append((vertex_map[edge[0]], vertex_map[edge[1]]))
        igraph_edge_attrs["weight"].append(max(0.0000001, weight))
        igraph_edge_attrs["shares"].append(attrs["shares"])
        igraph_edge_attrs["sharesDollar"].append(max(1, attrs["sharesDollar"]))
    graph.add_edges(igraph_edges, igraph_edge_attrs)

    hub_authority = np.round(np.array([
        graph.hub_score(),
        graph.authority_score()
    ]), 3)
    hub_authority[0] /= hub_authority[0].max() + 0.0001
    hub_authority[1] /= hub_authority[1].max() + 0.0001

    graph.vs["page_rank"] = graph.pagerank()
    graph.vs["hub"] = (hub_authority[0] + 0.0001).tolist()
    graph.vs["authority"] = (hub_authority[1] + 0.0001).tolist()
    graph.vs["hubOrAuthority"] = (np.max(hub_authority, axis=0, keepdims=True)[0] + 0.0001).tolist()

    filter_graph(graph, edge_weight_gte=edge_weight_gte, degree_gte=1)

    #print("blocs")
    #graph.vs["cohesive_blocks"] = graph.as_undirected().cohesive_blocks().membership
    #print("clusters")
    #graph.vs["cluster"] = graph.clusters().membership
    print("com1")
    graph.vs["community1"] = graph.community_label_propagation().membership
    print("com2")
    graph.vs["community2"] = graph.community_infomap().membership
    #print("com3")
    #graph.vs["community3"] = graph.community_spinglass(spins=10).membership
    print("--")

    layout = graph.layout_fruchterman_reingold()
    graph.vs["x"] = [c[0] for c in layout.coords]
    graph.vs["y"] = [c[1] for c in layout.coords]

    graph.write_dot(f"{filename}.dot")
    graph.write_gml(f"{filename}.gml")

    return graph


def filter_graph(
        graph: igraph.Graph,
        edge_weight_gte: float = 0.,
        degree_gte: int = 0,
) -> None:
    num_nodes, num_edges = len(graph.vs), len(graph.es)
    if edge_weight_gte:
        graph.delete_edges([
            i for i, w in enumerate(graph.es["weight"])
            if w < edge_weight_gte
        ])

    if degree_gte:
        graph.delete_vertices([
            i for i, (d_in, d_out) in enumerate(zip(graph.indegree(), graph.outdegree()))
            if d_in + d_out < degree_gte
        ])

    f_num_nodes, f_num_edges = len(graph.vs), len(graph.es)
    print(f"filtered {num_nodes}x{num_edges} -> {f_num_nodes}x{f_num_edges}")

#main()
#store_transactions("microsoft")

if 0:
    g = get_nasdaq_holder_graph(
        filename="graph-1b",
        start_symbols=[
            61322,  # vanguard
            #"vwagy",  # volkswagen ag adr
            #"ddaif",  # mercedes
            #"lmt",  # lockheed
            #"aldnf",
            #"TSLA", "AAPL", "MSFT", "AMZN", "AMD", "NVDA"
        ],
        max_distance=10,
        min_shares_dollar=1_000_000_000,
    )
    #graph.write_dot(f"{filename}.dot")
elif 1:
    g = get_nasdaq_holder_graph(
        #filename="graph-10m-10p",
        filename="layouted",
        start_symbols=[
            61322,  # vanguard
        ],
        max_distance=20,
        min_shares_dollar=10_000_000,
        edge_weight_gte=.1,
    )
elif 1:
    g: igraph.Graph = igraph.read("export.gml")
    layout = g.layout_fruchterman_reingold()
    print(layout)
    del g.vs["graphics"]
    g.vs["x"] = [c[0] for c in layout.coords]
    g.vs["y"] = [c[1] for c in layout.coords]
    g.write_dot("layouted.dot")

elif 1:
    get_nasdaq_holder_graph(
        filename="graph-vw",
        start_symbols=[
            "vwagy",  # volkswagen ag adr
        ],
        max_distance=1,
        min_value=1_000,
    )
#print(json.dumps(nasdaq.institutional_holdings(973119), indent=2))
