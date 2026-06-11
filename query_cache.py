import datasets
import neo4j
from concurrent.futures import ThreadPoolExecutor
from cypherbench.metrics.provenance_subgraph_jaccard_similarity import get_ps_cypher
import pickle
from tqdm import tqdm
from pathlib import Path
import json
import os
from typing import Any

CYPHERBENCH_DIR = Path(os.environ.get("CYPHERBENCH_DIR", Path.home() / "cypherbench"))
CACHE_PATH = CYPHERBENCH_DIR / "benchmark" / "train_cache.pkl"


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

    query_results = {}

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_pairs = []

        for row in ds:
            gold_cypher: str = row["gold_cypher"]  # type: ignore
            ps_cypher = get_ps_cypher(gold_cypher)
            graph: str = row["graph"]  # type: ignore
            driver = graph2conn[graph]

            gold_future = executor.submit(run_query, driver, gold_cypher)
            ps_future = executor.submit(run_query, driver, ps_cypher)

            future_pairs.append((gold_cypher, ps_cypher, gold_future, ps_future))

        for gold_cypher, ps_cypher, gold_future, ps_future in tqdm(future_pairs):
            query_results[gold_cypher] = {
                "gold_result": gold_future.result(),
                "gold_ps_result": ps_future.result(),
            }

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(query_results, f)


def get_cache() -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not CACHE_PATH.exists():
        print("Training dataset query cache not found. Generating now.")
        generate_cache()
        print("Done generating training query cache.")
    with open(CACHE_PATH, "rb") as f:
        cached_queries = pickle.load(f)
    return cached_queries


if __name__ == "__main__":
    get_cache()
