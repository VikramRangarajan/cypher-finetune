import datasets
import neo4j
from concurrent.futures import ThreadPoolExecutor
from cypherbench.metrics.provenance_subgraph_jaccard_similarity import get_ps_cypher
from tqdm import tqdm
from pathlib import Path
import json


def run_query(driver: neo4j.Driver, cypher, timeout=None):
    with driver.session(
        database="neo4j", default_access_mode=neo4j.READ_ACCESS
    ) as session:
        result = session.run(neo4j.Query(cypher, timeout=timeout))
        records = result.data()
        return records


def generate_cache():
    ds = datasets.load_dataset("megagonlabs/cypherbench", split="train")

    with open(Path.home() / "cypherbench" / "neo4j_info.json") as fin:
        neo4j_info = json.load(fin)

    train_graphs = neo4j_info["train_domains"]

    graph2conn = {}
    for graph in train_graphs:
        info = neo4j_info["full"][graph]
        uri = f"bolt://{info['host']}:{info['port']}"
        auth = (info["username"], info["password"])
        driver = neo4j.GraphDatabase.driver(
            uri=uri, auth=auth, max_connection_pool_size=100
        )
        graph2conn[graph] = driver

    paired_results = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_pairs = []

        for row in ds:
            gold_cypher = row["gold_cypher"]  # type: ignore
            ps_cypher = get_ps_cypher(gold_cypher)
            graph = row["graph"]  # type: ignore
            driver = graph2conn[graph]

            gold_future = executor.submit(run_query, driver, gold_cypher)
            ps_future = executor.submit(run_query, driver, ps_cypher)

            future_pairs.append((gold_future, ps_future))

        for gold_future, ps_future in tqdm(future_pairs):
            paired_results.append(
                (
                    gold_future.result(),
                    ps_future.result(),
                )
            )
