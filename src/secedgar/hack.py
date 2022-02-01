"""

https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-69

"""
import json
from typing import Union, Iterable

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


def get_nasdaq_holder_graph(
        filename: str,
        start_symbols: Iterable[Union[int, str]],
        scan_limit: int = 100,
        request_limit: int = 100,
        min_value: int = 1_000_000,
        max_distance: int = 7,
):
    todo = {
        (key, 0)
        for key in start_symbols
    }
    done = set()

    holdings_map = dict()
    profile_map = dict()
    vertex_map = dict()
    edge_set = set()
    edges = []
    edge_weights = []

    while todo:
        pick_entry = sorted(todo, key=lambda e: str(e[0]))[0]
        company_id, distance = pick_entry
        todo.remove(pick_entry)

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
            local_edges = []
            if holdings.get("data") and holdings["data"].get("institutionPositions"):
                for row in holdings["data"]["institutionPositions"]["table"]["rows"][:scan_limit]:
                    if not row["url"]:
                        continue
                    if int(row["value"].replace(",", "")) < min_value:
                        continue
                    held_company_id = row["url"].split("/")[3].lower()
                    local_edges.append((company_id, held_company_id, int(row["sharesHeld"].replace(",", ""))))

                    if distance <= max_distance and held_company_id not in done:
                        print(len(todo), distance, company_id, held_company_id)
                        todo.add((held_company_id, distance + 1))

            if holdings.get("data") and holdings["data"].get("holdingsTransactions"):
                for row in holdings["data"]["holdingsTransactions"]["table"]["rows"][:scan_limit]:
                    if not row["marketValue"] or int(row["marketValue"][1:].replace(",", "")) < min_value:
                        continue
                    holder_company_id = int(row["url"].split("-")[-1])
                    if holder_company_id not in vertex_map:
                        vertex_map[holder_company_id] = len(vertex_map)
                    local_edges.append((holder_company_id, company_id, int(row["sharesHeld"].replace(",", ""))))

                    if holder_company_id not in done:
                        print(len(todo), distance, holder_company_id, company_id)
                        todo.add((holder_company_id, distance + 1))

            for edge in local_edges:
                if edge[:2] in edge_set:
                    continue
                edge_set.add(edge[:2])

                for id in edge[:2]:
                    if id not in vertex_map:
                        vertex_map[id] = len(vertex_map)
                edges.append((vertex_map[edge[0]], vertex_map[edge[1]]))
                edge_weights.append(edge[2])

        except:
            print(json.dumps(holdings, indent=2))
            raise

    graph = igraph.Graph(directed=True)

    for company_id in sorted(vertex_map, key=lambda id: vertex_map[id]):
        attributes = {}
        if isinstance(company_id, str):
            profile = profile_map[company_id]
            holdings = holdings_map[company_id]
            if profile["status"]["rCode"] == 400:
                attributes.update({
                    "name": company_id,
                    "market_value": 0,
                })
            else:
                try:
                    attributes.update({
                        "name": profile["data"]["CompanyName"]["value"],
                        "sector": profile["data"]["Sector"]["value"],
                        "region": profile["data"]["Region"]["value"],
                        "industry": profile["data"]["Industry"]["value"],
                    })
                except:
                    raise ValueError(f"'{company_id}' in {profile}")

                value = int(((
                     (holdings["data"].get("ownershipSummary") or {})
                     .get("TotalHoldingsValue") or {}
                 ).get("value") or "$0").replace(",", "")[1:])
                attributes["market_value"] = value

        else:
            holdings = holdings_map[company_id]
            value = int(((
                (holdings["data"].get("positionStatistics") or {})
                .get("TotalMktValue") or {}
            ).get("value") or "0").replace(",", ""))
            attributes.update({
                "name": holdings["data"]["title"],
                "market_value": value,
            })
        attributes.update({
            "label": attributes["name"],
            "symbol": str(company_id),
            #"market_value": str(attributes["market_value"]) + ".0",
        })
        graph.add_vertex(**attributes)

    max_edge_weight = max(edge_weights)
    edge_weights = [round(e / max_edge_weight * 100, 1) for e in edge_weights]
    edge_weights = [max(1, e) for e in edge_weights]

    graph.add_edges(edges, {"weight": edge_weights})
    #print(graph)
    graph.write_dot(f"{filename}.dot")
    graph.write_gml(f"{filename}.gml")
    #graph.write_edgelist("graph.edge")

#print(json.dumps(holdings, indent=2))


#main()
#store_transactions("microsoft")

if 1:
    get_nasdaq_holder_graph(
        filename="graph-100m",
        start_symbols=[
            61322,  # vanguard
            #"vwagy",  # volkswagen ag adr
            #"ddaif",  # mercedes
            #"lmt",  # lockheed
            #"aldnf",
            #"TSLA", "AAPL", "MSFT", "AMZN", "AMD", "NVDA"
        ],
        max_distance=10,
        min_value=100_000,
    )
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