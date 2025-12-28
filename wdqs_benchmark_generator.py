"""
wdqs_benchmark_generator.py

SPARQL PoC generator for a balanced KG benchmark:
- Select N relations (PIDs) from wikidata_ontology.properties
- For each relation, query up to K entity->entity triples from WDQS
- Insert into triplets_db.triplets with a sample_id
- Emit a JSON report

Defaults requested:
  N_RELATIONS=50, TRIPLES_PER_RELATION=500, no domain/range constraints
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
from pymongo import MongoClient, UpdateOne


WDQS_ENDPOINT = "https://query.wikidata.org/sparql"


@dataclass(frozen=True)
class Config:
    mongo_uri: str
    ontology_db: str
    triplets_db: str
    sample_id: str

    n_relations: int
    triples_per_relation: int
    seed: int

    clear_existing: bool
    timeout_s: int
    max_retries: int
    sleep_base_s: float


def parse_args() -> Config:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mongo_uri", required=True, help="MongoDB URI (use the same Atlas/Atlas-local URI as Streamlit).")
    ap.add_argument("--ontology_db", default="wikidata_ontology")
    ap.add_argument("--triplets_db", default="triplets_db")
    ap.add_argument("--sample_id", required=True, help="Namespace for this build; used for visualization queries.")

    ap.add_argument("--n_relations", type=int, default=50)
    ap.add_argument("--triples_per_relation", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--clear_existing", action="store_true", help="Delete existing triplets for this sample_id first.")
    ap.add_argument("--timeout_s", type=int, default=55, help="Per-request timeout (seconds). Keep < 60 for WDQS.")
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--sleep_base_s", type=float, default=1.5)

    ns = ap.parse_args()
    return Config(
        mongo_uri=ns.mongo_uri,
        ontology_db=ns.ontology_db,
        triplets_db=ns.triplets_db,
        sample_id=ns.sample_id,
        n_relations=ns.n_relations,
        triples_per_relation=ns.triples_per_relation,
        seed=ns.seed,
        clear_existing=ns.clear_existing,
        timeout_s=ns.timeout_s,
        max_retries=ns.max_retries,
        sleep_base_s=ns.sleep_base_s,
    )


# ----------------------------
# Ontology DB helpers
# ----------------------------

def assert_ontology_ready(client: MongoClient, ontology_db: str) -> None:
    db = client[ontology_db]
    required = {"properties"}
    existing = set(db.list_collection_names())
    missing = required - existing
    if missing:
        raise RuntimeError(
            f"Ontology DB not ready. Missing {sorted(missing)} in '{ontology_db}'. "
            f"Run create_wikidata_ontology_db.py against this mongo_uri."
        )
    if db["properties"].count_documents({}) == 0:
        raise RuntimeError(
            f"Ontology DB '{ontology_db}.properties' is empty. Run create_wikidata_ontology_db.py."
        )


def load_allowed_pids(client: MongoClient, ontology_db: str) -> List[str]:
    db = client[ontology_db]
    # properties docs contain: property_id, label, ...
    pids = [d["property_id"] for d in db["properties"].find({}, {"_id": 0, "property_id": 1})]
    # Basic sanity: keep only "P..." form
    pids = [p for p in pids if isinstance(p, str) and p.startswith("P")]
    return pids


# ----------------------------
# WDQS query logic
# ----------------------------

def build_pid_query(pid: str, limit: int) -> str:
    # Entity-to-entity only:
    # - subject IRI in wd:Q...
    # - object IRI in wd:Q...
    #
    # We use wdt:PID direct claims (truthy-ish simplified view).
    return f"""
SELECT ?s ?o WHERE {{
  ?s wdt:{pid} ?o .
  FILTER(STRSTARTS(STR(?s), "http://www.wikidata.org/entity/Q"))
  FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
}}
LIMIT {limit}
""".strip()


def qid_from_iri(iri: str) -> Optional[str]:
    # "http://www.wikidata.org/entity/Q42" -> "Q42"
    if not iri:
        return None
    if "/entity/" in iri:
        tail = iri.rsplit("/entity/", 1)[-1]
        if tail.startswith("Q"):
            return tail
    return None


def wdqs_post(query: str, timeout_s: int, max_retries: int, sleep_base_s: float) -> Dict:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "WikonticBenchmarkPOC/1.0 (contact: local-dev)"
    }
    data = {"query": query}

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(WDQS_ENDPOINT, data=data, headers=headers, timeout=timeout_s)
            if resp.status_code == 200:
                return resp.json()

            # Retry on common transient codes
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"WDQS HTTP {resp.status_code}: {resp.text[:200]}")
                sleep = sleep_base_s * (2 ** attempt)
                time.sleep(sleep)
                continue

            # Non-retryable
            raise RuntimeError(f"WDQS HTTP {resp.status_code}: {resp.text[:500]}")
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            sleep = sleep_base_s * (2 ** attempt)
            time.sleep(sleep)

    raise RuntimeError(f"WDQS request failed after retries. Last error: {last_err}")


def fetch_triples_for_pid(pid: str, limit: int, cfg: Config) -> Tuple[List[Tuple[str, str, str]], Dict]:
    q = build_pid_query(pid, limit)
    t0 = time.time()
    data = wdqs_post(q, cfg.timeout_s, cfg.max_retries, cfg.sleep_base_s)
    dt = time.time() - t0

    rows = data.get("results", {}).get("bindings", [])
    out: List[Tuple[str, str, str]] = []

    for b in rows:
        s_iri = b.get("s", {}).get("value")
        o_iri = b.get("o", {}).get("value")
        s = qid_from_iri(s_iri)
        o = qid_from_iri(o_iri)
        if s and o:
            out.append((s, pid, o))

    meta = {
        "pid": pid,
        "returned_bindings": len(rows),
        "valid_qid_edges": len(out),
        "seconds": round(dt, 3),
    }
    return out, meta


# ----------------------------
# Mongo writing
# ----------------------------

def upsert_edges(client: MongoClient, triplets_db: str, sample_id: str, edges: List[Tuple[str, str, str]]) -> int:
    col = client[triplets_db]["triplets"]
    ops = []
    for (s, p, o) in edges:
        doc = {"subject": s, "relation": p, "object": o, "sample_id": sample_id}
        filt = {"subject": s, "relation": p, "object": o, "sample_id": sample_id}
        ops.append(UpdateOne(filt, {"$setOnInsert": doc}, upsert=True))

    if not ops:
        return 0
    res = col.bulk_write(ops, ordered=False)
    return res.upserted_count


def clear_sample(client: MongoClient, triplets_db: str, sample_id: str) -> int:
    col = client[triplets_db]["triplets"]
    res = col.delete_many({"sample_id": sample_id})
    return res.deleted_count


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cfg = parse_args()

    client = MongoClient(cfg.mongo_uri)
    assert_ontology_ready(client, cfg.ontology_db)

    allowed_pids = load_allowed_pids(client, cfg.ontology_db)
    if len(allowed_pids) < cfg.n_relations:
        raise RuntimeError(f"Ontology only contains {len(allowed_pids)} PIDs; cannot sample {cfg.n_relations}.")

    rng = random.Random(cfg.seed)
    chosen_pids = rng.sample(allowed_pids, cfg.n_relations)

    if cfg.clear_existing:
        deleted = clear_sample(client, cfg.triplets_db, cfg.sample_id)
        print(f"Cleared {deleted} existing triplets for sample_id={cfg.sample_id}")

    report = {
        "sample_id": cfg.sample_id,
        "mongo_uri": cfg.mongo_uri,
        "ontology_db": cfg.ontology_db,
        "triplets_db": cfg.triplets_db,
        "n_relations": cfg.n_relations,
        "triples_per_relation": cfg.triples_per_relation,
        "seed": cfg.seed,
        "pids": chosen_pids,
        "per_pid": [],
        "totals": {},
    }

    all_edges: List[Tuple[str, str, str]] = []
    total_inserted = 0
    total_edges = 0
    failures: List[Dict] = []

    for i, pid in enumerate(chosen_pids, start=1):
        print(f"[{i}/{len(chosen_pids)}] Fetching {cfg.triples_per_relation} triples for {pid} ...")
        try:
            edges, meta = fetch_triples_for_pid(pid, cfg.triples_per_relation, cfg)
            total_edges += len(edges)
            all_edges.extend(edges)

            ins = upsert_edges(client, cfg.triplets_db, cfg.sample_id, edges)
            total_inserted += ins

            meta["inserted"] = ins
            report["per_pid"].append(meta)
        except Exception as e:
            failures.append({"pid": pid, "error": str(e)})
            print(f"  ERROR for {pid}: {e}")

        # Small courtesy pause to reduce WDQS load
        time.sleep(0.3)

    report["totals"] = {
        "edges_collected": total_edges,
        "edges_upserted": total_inserted,
        "failed_pids": len(failures),
    }
    report["failures"] = failures

    # Relation histogram on collected edges
    hist: Dict[str, int] = {}
    for (_, p, _) in all_edges:
        hist[p] = hist.get(p, 0) + 1
    report["relation_histogram"] = dict(sorted(hist.items(), key=lambda x: x[1], reverse=True))

    out_path = f"wdqs_benchmark_report__{cfg.sample_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDONE")
    print(json.dumps(report["totals"], indent=2))
    print(f"Report written to: {out_path}")
    print(f"Now visualize by loading sample_id='{cfg.sample_id}' from triplets_db.triplets")


if __name__ == "__main__":
    main()
