"""
Utilities for building a balanced KG benchmark from Wikidata (WDQS).

- Validates the Wikidata ontology DB produced by create_wikidata_ontology_db.py
- Samples or validates relations (PIDs) from the ontology
- Fetches entity-to-entity triples per PID from WDQS with retry/backoff
- Upserts into triplets_db.triplets with a sample_id namespace

The TripleSource abstraction isolates where triples come from so we can swap
WDQS for a dump-based reader later without rewriting the orchestration code.
"""

from __future__ import annotations

import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import statistics

import requests
from pymongo import MongoClient, UpdateOne

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"


# ----------------------------
# Config and helpers
# ----------------------------


@dataclass(frozen=True)
class BenchmarkRunConfig:
    mongo_uri: str
    ontology_db: str
    triplets_db: str
    sample_id: str

    relation_source: str  # "ontology_pids" | "custom_pids"
    custom_pids: List[str]

    n_relations: int
    triples_per_relation: int
    seed: int
    balancing_mode: str  # "per_relation_cap" | "uniform_relations"

    clear_existing: bool
    timeout_s: int
    max_retries: int
    sleep_base_s: float


class OntologyHelper:
    """
    Minimal ontology accessor for wikidata_ontology DB created by create_wikidata_ontology_db.py.
    """

    REQUIRED_COLLS = {"properties"}

    def __init__(self, client: MongoClient, db_name: str):
        self.db = client[db_name]

    def assert_ready(self) -> None:
        existing = set(self.db.list_collection_names())
        missing = self.REQUIRED_COLLS - existing
        if missing:
            raise RuntimeError(
                f"Ontology DB not ready. Missing collections: {sorted(missing)}. "
                "Run create_wikidata_ontology_db.py first."
            )

        if self.db["properties"].count_documents({}) == 0:
            raise RuntimeError(
                "Ontology DB has no properties. Run create_wikidata_ontology_db.py to populate it."
            )

    def list_property_ids(self) -> List[str]:
        pids = [
            d["property_id"]
            for d in self.db["properties"].find({}, {"_id": 0, "property_id": 1})
        ]
        return [p for p in pids if isinstance(p, str) and p.startswith("P")]

    def validate_pids(self, pids: Sequence[str]) -> Tuple[List[str], List[str]]:
        allowed = set(self.list_property_ids())
        ok = [p for p in pids if p in allowed]
        bad = [p for p in pids if p not in allowed]
        return ok, bad


# ----------------------------
# Triple source abstraction
# ----------------------------


class TripleSource:
    """
    Interface for fetching triples per PID. Implementations return:
      edges: list[(subject_qid, pid, object_qid)]
      meta: dict with debug/telemetry (seconds, raw bindings count, etc.)
    """

    def fetch_edges(self, pid: str, limit: int) -> Tuple[List[Tuple[str, str, str]], Dict]:
        raise NotImplementedError


def _qid_from_iri(iri: str) -> Optional[str]:
    if not iri:
        return None
    if "/entity/" in iri:
        tail = iri.rsplit("/entity/", 1)[-1]
        if tail.startswith("Q"):
            return tail
    return None


def _build_pid_query(pid: str, limit: int) -> str:
    # Entity-to-entity only, using wdt:PID direct claims
    return f"""
SELECT ?s ?o WHERE {{
  ?s wdt:{pid} ?o .
  FILTER(STRSTARTS(STR(?s), "http://www.wikidata.org/entity/Q"))
  FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
}}
LIMIT {limit}
""".strip()


class TripleSourceWDQS(TripleSource):
    def __init__(self, timeout_s: int, max_retries: int, sleep_base_s: float):
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.sleep_base_s = sleep_base_s

    def _post(self, query: str) -> Dict:
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "WikonticBenchmarkPOC/1.0 (local streamlit)",
        }
        data = {"query": query}

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    WDQS_ENDPOINT,
                    data=data,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"WDQS HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(self.sleep_base_s * (2 ** attempt))
                    continue

                raise RuntimeError(f"WDQS HTTP {resp.status_code}: {resp.text[:500]}")
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                time.sleep(self.sleep_base_s * (2 ** attempt))

        raise RuntimeError(f"WDQS request failed after retries. Last error: {last_err}")

    def fetch_edges(self, pid: str, limit: int) -> Tuple[List[Tuple[str, str, str]], Dict]:
        q = _build_pid_query(pid, limit)
        t0 = time.time()
        data = self._post(q)
        dt = time.time() - t0

        rows = data.get("results", {}).get("bindings", [])
        edges: List[Tuple[str, str, str]] = []
        for b in rows:
            s = _qid_from_iri(b.get("s", {}).get("value"))
            o = _qid_from_iri(b.get("o", {}).get("value"))
            if s and o:
                edges.append((s, pid, o))

        meta = {
            "pid": pid,
            "returned_bindings": len(rows),
            "valid_qid_edges": len(edges),
            "seconds": round(dt, 3),
        }
        return edges, meta


# ----------------------------
# Mongo helpers
# ----------------------------


def _upsert_edges(client: MongoClient, triplets_db: str, sample_id: str, edges: Iterable[Tuple[str, str, str]]) -> int:
    col = client[triplets_db]["triplets"]
    ops = []
    for (s, p, o) in edges:
        doc = {"subject": s, "relation": p, "object": o, "sample_id": sample_id}
        filt = doc.copy()
        ops.append(UpdateOne(filt, {"$setOnInsert": doc}, upsert=True))
    if not ops:
        return 0
    res = col.bulk_write(ops, ordered=False)
    return res.upserted_count


def _clear_sample(client: MongoClient, triplets_db: str, sample_id: str) -> int:
    col = client[triplets_db]["triplets"]
    res = col.delete_many({"sample_id": sample_id})
    return res.deleted_count


def load_edges_for_sample(
    client: MongoClient, triplets_db: str, sample_id: str, limit: int
) -> List[Tuple[str, str, str]]:
    col = client[triplets_db]["triplets"]
    docs = list(
        col.find(
            {"sample_id": sample_id},
            {"_id": 0, "subject": 1, "relation": 1, "object": 1},
        ).limit(int(limit))
    )
    return [(d["subject"], d["relation"], d["object"]) for d in docs]


# ----------------------------
# Metrics
# ----------------------------


def relation_metrics_from_edges(edges: List[Tuple[str, str, str]]) -> Dict:
    counts = Counter(p for _, p, _ in edges)
    if not counts:
        return {
            "num_relations": 0,
            "total_edges": 0,
            "min_per_relation": 0,
            "max_per_relation": 0,
            "mean_per_relation": 0,
            "stdev_per_relation": 0,
        }

    vals = list(counts.values())
    return {
        "num_relations": len(vals),
        "total_edges": sum(vals),
        "min_per_relation": min(vals),
        "max_per_relation": max(vals),
        "mean_per_relation": statistics.mean(vals),
        "stdev_per_relation": statistics.pstdev(vals),
    }


def graph_connectivity_metrics(edges: List[Tuple[str, str, str]]) -> Dict:
    """
    Computes weak connectivity on an undirected projection (common for KG quick checks).
    """
    nodes = set()
    adj: Dict[str, set] = {}
    for s, _, o in edges:
        nodes.add(s)
        nodes.add(o)
        adj.setdefault(s, set()).add(o)
        adj.setdefault(o, set()).add(s)

    num_nodes = len(nodes)
    num_edges = len(edges)
    if num_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "largest_component": 0,
            "avg_degree": 0,
            "edge_density": 0,
        }

    seen = set()
    comp_sizes: List[int] = []

    for node in nodes:
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nei in adj.get(cur, []):
                if nei not in seen:
                    seen.add(nei)
                    q.append(nei)
        comp_sizes.append(size)

    avg_degree = (2 * num_edges) / num_nodes
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "components": len(comp_sizes),
        "largest_component": max(comp_sizes) if comp_sizes else 0,
        "avg_degree": avg_degree,
        "edge_density": density,
    }


# ----------------------------
# Core runner
# ----------------------------


def _select_relations(
    relation_source: str,
    allowed_pids: List[str],
    n_relations: int,
    custom_pids: Sequence[str],
    seed: int,
) -> List[str]:
    if relation_source == "custom_pids":
        cleaned = [p.strip() for p in custom_pids if p.strip()]
        if not cleaned:
            raise RuntimeError("relation_source=custom_pids but no PIDs were provided.")
        unique = []
        seen = set()
        for p in cleaned:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        return unique

    if n_relations > len(allowed_pids):
        raise RuntimeError(f"Requested {n_relations} relations but ontology only has {len(allowed_pids)}.")
    rng = random.Random(seed)
    return rng.sample(allowed_pids, n_relations)


def _apply_balancing(
    edges: List[Tuple[str, str, str]],
    mode: str,
    per_relation_cap: int,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    """
    per_relation_cap: cap per relation (already limited by WDQS limit)
    uniform_relations: enforce equal cap when possible (drop extras)
    """
    if len(edges) <= per_relation_cap:
        return edges
    # If more than cap, sample for stability/repeatability
    return rng.sample(edges, per_relation_cap)


def run_benchmark(cfg: BenchmarkRunConfig, triple_source: Optional[TripleSource] = None) -> Dict:
    client = MongoClient(cfg.mongo_uri)
    ontology = OntologyHelper(client, cfg.ontology_db)
    ontology.assert_ready()

    allowed_pids = ontology.list_property_ids()
    if not allowed_pids:
        raise RuntimeError("Ontology properties collection is empty.")

    if cfg.relation_source == "custom_pids":
        ok, bad = ontology.validate_pids(cfg.custom_pids)
        if bad:
            raise RuntimeError(f"Some custom PIDs are not present in ontology: {bad}")
        chosen_pids = _select_relations("custom_pids", allowed_pids, cfg.n_relations, ok, cfg.seed)
    else:
        chosen_pids = _select_relations("ontology_pids", allowed_pids, cfg.n_relations, [], cfg.seed)

    triple_source = triple_source or TripleSourceWDQS(
        timeout_s=cfg.timeout_s, max_retries=cfg.max_retries, sleep_base_s=cfg.sleep_base_s
    )
    rng = random.Random(cfg.seed)

    if cfg.clear_existing:
        _clear_sample(client, cfg.triplets_db, cfg.sample_id)

    report: Dict = {
        "sample_id": cfg.sample_id,
        "mongo_uri": cfg.mongo_uri,
        "ontology_db": cfg.ontology_db,
        "triplets_db": cfg.triplets_db,
        "relation_source": cfg.relation_source,
        "custom_pids": cfg.custom_pids,
        "chosen_pids": chosen_pids,
        "n_relations": cfg.n_relations,
        "triples_per_relation": cfg.triples_per_relation,
        "balancing_mode": cfg.balancing_mode,
        "seed": cfg.seed,
        "per_pid": [],
        "failures": [],
        "totals": {},
        "relation_histogram": {},
    }

    total_collected = 0
    total_upserted = 0
    all_edges: List[Tuple[str, str, str]] = []

    for pid in chosen_pids:
        try:
            edges_raw, meta = triple_source.fetch_edges(pid, cfg.triples_per_relation)
            edges_balanced = _apply_balancing(
                edges_raw,
                mode=cfg.balancing_mode,
                per_relation_cap=cfg.triples_per_relation,
                rng=rng,
            )

            inserted = _upsert_edges(client, cfg.triplets_db, cfg.sample_id, edges_balanced)

            total_collected += len(edges_balanced)
            total_upserted += inserted
            all_edges.extend(edges_balanced)

            meta.update(
                {
                    "pid": pid,
                    "used_edges": len(edges_balanced),
                    "inserted": inserted,
                }
            )
            report["per_pid"].append(meta)
        except Exception as e:
            report["failures"].append({"pid": pid, "error": str(e)})

        # Friendly pause between calls
        time.sleep(0.3)

    hist: Dict[str, int] = {}
    for (_, p, _) in all_edges:
        hist[p] = hist.get(p, 0) + 1

    report["relation_histogram"] = dict(sorted(hist.items(), key=lambda x: x[1], reverse=True))
    report["totals"] = {
        "edges_collected": total_collected,
        "edges_upserted": total_upserted,
        "failed_pids": len(report["failures"]),
    }
    return report
