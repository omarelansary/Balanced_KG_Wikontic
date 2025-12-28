"""
benchmark_builder.py

Proof-of-concept: Build a balanced KG dataset (triplets) and store it in MongoDB
so it can be visualized by the existing Wikontic Streamlit app.

Key features:
- Choose domain and range (entity type IDs like Q5) or label-based lookup
- Choose relation source:
    (A) Wikontic ontology PIDs (wikidata_ontology.properties)
    (B) User-specified PID list (validated against ontology)
- Balance policies:
    - per_relation_cap: cap N triples per relation
    - uniform_relations: equal N per relation (if possible)
- Save to triplets_db.triplets with a sample_id so Streamlit can visualize
- Provider abstraction for future datasets (Wikidata dump now is stubbed; can be implemented later)

This file does NOT modify Wikontic’s existing workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple

from pymongo import MongoClient, UpdateOne


# ----------------------------
# Mongo / Ontology helpers
# ----------------------------

@dataclass(frozen=True)
class MongoCfg:
    mongo_uri: str
    ontology_db: str = "wikidata_ontology"
    triplets_db: str = "triplets_db"


class OntologyStore:
    """
    Accesses wikidata_ontology DB created by create_wikidata_ontology_db.py.

    We validate readiness by checking required collections and that properties exist.
    """
    REQUIRED_COLLS = {"entity_types", "entity_type_aliases", "properties", "property_aliases"}

    def __init__(self, client: MongoClient, db_name: str):
        self.db = client[db_name]

    def assert_ready(self) -> None:
        existing = set(self.db.list_collection_names())
        missing = self.REQUIRED_COLLS - existing
        if missing:
            raise RuntimeError(
                f"Ontology DB not ready. Missing collections: {sorted(missing)}. "
                f"Run create_wikidata_ontology_db.py against this Mongo URI."
            )
        n_props = self.db["properties"].count_documents({})
        if n_props == 0:
            raise RuntimeError(
                "Ontology DB 'properties' is empty. Run create_wikidata_ontology_db.py."
            )

    def list_property_ids(self) -> List[str]:
        # properties documents contain property_id + label (per your DB)
        return [d["property_id"] for d in self.db["properties"].find({}, {"_id": 0, "property_id": 1})]

    def validate_pids(self, pids: List[str]) -> Tuple[List[str], List[str]]:
        allowed = set(self.list_property_ids())
        ok = [p for p in pids if p in allowed]
        bad = [p for p in pids if p not in allowed]
        return ok, bad

    def validate_type_ids(self, type_ids: List[str]) -> Tuple[List[str], List[str]]:
        # entity_types contains entity_type_id (e.g., Q5) and label
        allowed = set(
            d["entity_type_id"]
            for d in self.db["entity_types"].find({}, {"_id": 0, "entity_type_id": 1})
        )
        ok = [t for t in type_ids if t in allowed]
        bad = [t for t in type_ids if t not in allowed]
        return ok, bad

    def resolve_type_labels_to_ids(self, labels: List[str]) -> Dict[str, Optional[str]]:
        """
        If user provides type labels instead of IDs, try exact match in entity_types.label.
        (You can expand later to alias matching.)
        """
        out = {}
        for lab in labels:
            doc = self.db["entity_types"].find_one({"label": lab}, {"_id": 0, "entity_type_id": 1})
            out[lab] = doc["entity_type_id"] if doc else None
        return out


class TripletsStore:
    """
    Writes final benchmark triplets to triplets_db.triplets.
    Matches the schema used by the structured/dynamic aligners (subject, relation, object, sample_id).
    """
    def __init__(self, client: MongoClient, db_name: str, triplets_collection: str = "triplets"):
        self.db = client[db_name]
        self.col = self.db[triplets_collection]

    def clear_sample(self, sample_id: str) -> int:
        res = self.col.delete_many({"sample_id": sample_id})
        return res.deleted_count

    def upsert_triplets(self, triplets: List[Dict], sample_id: str) -> int:
        ops = []
        for t in triplets:
            doc = {
                "subject": t["subject"],
                "relation": t["relation"],
                "object": t["object"],
                "sample_id": sample_id,
            }
            filt = {
                "subject": doc["subject"],
                "relation": doc["relation"],
                "object": doc["object"],
                "sample_id": sample_id,
            }
            ops.append(UpdateOne(filt, {"$setOnInsert": doc}, upsert=True))
        if not ops:
            return 0
        res = self.col.bulk_write(ops, ordered=False)
        return res.upserted_count


# ----------------------------
# Triplets Provider (Wikidata now, extensible later)
# ----------------------------

class TripletsProvider:
    """
    Interface to yield candidate triples (subject, relation, object) with optional types.
    For a real Wikidata-scale benchmark, implement a dump reader here.
    """

    def iter_triplets(
        self,
        allowed_pids: set[str],
        domain_types: set[str],
        range_types: set[str],
        limit: int,
    ) -> Iterable[Dict]:
        raise NotImplementedError


class CSVTripletsProvider(TripletsProvider):
    """
    Proof-of-concept provider: reads pre-extracted triples from a CSV file.

    Expected columns:
      subject, relation, object, (optional) subject_type, object_type

    - subject_type/object_type enable domain/range filtering without extra Wikidata lookup.
    - If those are missing, domain/range filtering cannot be enforced in this POC.
    """

    def __init__(self, path: str):
        self.path = path

    def iter_triplets(
        self,
        allowed_pids: set[str],
        domain_types: set[str],
        range_types: set[str],
        limit: int,
    ) -> Iterable[Dict]:
        n = 0
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"subject", "relation", "object"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"CSV missing required columns {sorted(required)}")

            for row in reader:
                rel = (row.get("relation") or "").strip()
                if rel not in allowed_pids:
                    continue

                t = {
                    "subject": (row.get("subject") or "").strip(),
                    "relation": rel,
                    "object": (row.get("object") or "").strip(),
                    "subject_type": (row.get("subject_type") or "").strip(),
                    "object_type": (row.get("object_type") or "").strip(),
                }

                if not t["subject"] or not t["object"]:
                    continue

                # If types are present, enforce domain/range
                if domain_types and t["subject_type"] and t["subject_type"] not in domain_types:
                    continue
                if range_types and t["object_type"] and t["object_type"] not in range_types:
                    continue

                yield t
                n += 1
                if n >= limit:
                    break


# ----------------------------
# Balancing
# ----------------------------

def balance_triplets(
    triplets: List[Dict],
    mode: str,
    per_relation_cap: int,
    seed: int,
) -> List[Dict]:
    """
    Balancing modes:
      - per_relation_cap: for each relation, sample up to per_relation_cap
      - uniform_relations: sample exactly per_relation_cap from each relation (drop relations with fewer)
    """
    rng = random.Random(seed)
    by_rel: Dict[str, List[Dict]] = {}
    for t in triplets:
        by_rel.setdefault(t["relation"], []).append(t)

    out: List[Dict] = []
    if mode == "per_relation_cap":
        for r, items in by_rel.items():
            if len(items) <= per_relation_cap:
                out.extend(items)
            else:
                out.extend(rng.sample(items, per_relation_cap))
    elif mode == "uniform_relations":
        # keep only relations with enough support, then take exactly cap from each
        for r, items in by_rel.items():
            if len(items) >= per_relation_cap:
                out.extend(rng.sample(items, per_relation_cap))
    else:
        raise ValueError(f"Unknown balance mode: {mode}")

    rng.shuffle(out)
    return out


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mongo_uri", required=True, help="MongoDB URI (Atlas/Atlas local with Search enabled is OK).")
    ap.add_argument("--ontology_db", default="wikidata_ontology")
    ap.add_argument("--triplets_db", default="triplets_db")
    ap.add_argument("--sample_id", required=True, help="Namespace for this build; Streamlit uses this to query/visualize.")

    # Domains/Ranges (types)
    ap.add_argument("--domain_type_ids", default="", help="Comma-separated entity type IDs (e.g., Q5,Q6256).")
    ap.add_argument("--range_type_ids", default="", help="Comma-separated entity type IDs.")
    ap.add_argument("--domain_type_labels", default="", help="Comma-separated labels to resolve via ontology entity_types.label.")
    ap.add_argument("--range_type_labels", default="", help="Comma-separated labels to resolve via ontology entity_types.label.")

    # Relation source
    ap.add_argument("--relation_source", choices=["ontology_pids", "custom_pids"], required=True)
    ap.add_argument("--custom_pids", default="", help="Comma-separated PIDs if relation_source=custom_pids")

    # Provider
    ap.add_argument("--provider", choices=["csv"], default="csv")
    ap.add_argument("--csv_path", default="", help="Path to CSV with columns subject,relation,object,(optional)subject_type,object_type")

    # Sampling / balancing
    ap.add_argument("--candidate_limit", type=int, default=1_000_000, help="Max candidates to read from provider")
    ap.add_argument("--balance_mode", choices=["per_relation_cap", "uniform_relations"], default="per_relation_cap")
    ap.add_argument("--per_relation_cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)

    # Safety
    ap.add_argument("--clear_existing", action="store_true", help="Delete existing triplets for sample_id before insert.")
    ap.add_argument("--dry_run", action="store_true", help="Do not write to DB; print summary only.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    client = MongoClient(args.mongo_uri)
    ontology = OntologyStore(client, args.ontology_db)
    ontology.assert_ready()

    # --- Resolve/validate types ---
    domain_ids = [x.strip() for x in args.domain_type_ids.split(",") if x.strip()]
    range_ids = [x.strip() for x in args.range_type_ids.split(",") if x.strip()]

    if args.domain_type_labels:
        mapping = ontology.resolve_type_labels_to_ids([x.strip() for x in args.domain_type_labels.split(",") if x.strip()])
        unresolved = [k for k, v in mapping.items() if v is None]
        if unresolved:
            raise RuntimeError(f"Unresolved domain type labels: {unresolved}. Use --domain_type_ids or add exact labels.")
        domain_ids.extend([v for v in mapping.values() if v])

    if args.range_type_labels:
        mapping = ontology.resolve_type_labels_to_ids([x.strip() for x in args.range_type_labels.split(",") if x.strip()])
        unresolved = [k for k, v in mapping.items() if v is None]
        if unresolved:
            raise RuntimeError(f"Unresolved range type labels: {unresolved}. Use --range_type_ids or add exact labels.")
        range_ids.extend([v for v in mapping.values() if v])

    ok_dom, bad_dom = ontology.validate_type_ids(domain_ids)
    ok_rng, bad_rng = ontology.validate_type_ids(range_ids)
    if bad_dom:
        raise RuntimeError(f"Invalid domain type IDs: {bad_dom}")
    if bad_rng:
        raise RuntimeError(f"Invalid range type IDs: {bad_rng}")

    domain_types = set(ok_dom)
    range_types = set(ok_rng)

    # --- Resolve/validate PIDs ---
    if args.relation_source == "ontology_pids":
        pids = ontology.list_property_ids()
    else:
        pids = [x.strip() for x in args.custom_pids.split(",") if x.strip()]
        if not pids:
            raise RuntimeError("relation_source=custom_pids but --custom_pids is empty.")
        ok, bad = ontology.validate_pids(pids)
        if bad:
            raise RuntimeError(f"Some custom PIDs are not in ontology properties: {bad}")
        pids = ok

    allowed_pids = set(pids)

    # --- Provider ---
    if args.provider == "csv":
        if not args.csv_path:
            raise RuntimeError("--provider csv requires --csv_path")
        provider = CSVTripletsProvider(args.csv_path)
    else:
        raise RuntimeError("Unsupported provider (POC only implements csv).")

    # --- Collect candidates ---
    candidates: List[Dict] = list(
        provider.iter_triplets(
            allowed_pids=allowed_pids,
            domain_types=domain_types,
            range_types=range_types,
            limit=args.candidate_limit,
        )
    )

    if not candidates:
        raise RuntimeError(
            "No candidates found after filtering. Check:\n"
            "- your CSV has subject/relation/object\n"
            "- relation IDs match (e.g., P31)\n"
            "- if you used domain/range, ensure subject_type/object_type columns exist and match Q-IDs"
        )

    # --- Balance ---
    balanced = balance_triplets(
        candidates,
        mode=args.balance_mode,
        per_relation_cap=args.per_relation_cap,
        seed=args.seed,
    )

    # --- Summary ---
    rel_counts: Dict[str, int] = {}
    for t in balanced:
        rel_counts[t["relation"]] = rel_counts.get(t["relation"], 0) + 1

    summary = {
        "sample_id": args.sample_id,
        "ontology_db": args.ontology_db,
        "triplets_db": args.triplets_db,
        "relation_source": args.relation_source,
        "num_allowed_pids": len(allowed_pids),
        "domain_types": sorted(domain_types),
        "range_types": sorted(range_types),
        "candidate_limit": args.candidate_limit,
        "candidates_after_filter": len(candidates),
        "balance_mode": args.balance_mode,
        "per_relation_cap": args.per_relation_cap,
        "balanced_triplets": len(balanced),
        "relations_in_output": len(rel_counts),
        "top_relations": sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:20],
        "seed": args.seed,
    }

    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("Dry-run enabled: not writing to DB.")
        return

    # --- Write to DB for Streamlit visualization ---
    store = TripletsStore(client, args.triplets_db, "triplets")
    if args.clear_existing:
        deleted = store.clear_sample(args.sample_id)
        print(f"Cleared {deleted} existing triplets for sample_id={args.sample_id}")

    inserted = store.upsert_triplets(balanced, args.sample_id)
    print(f"Inserted (upserted) {inserted} new triplets into {args.triplets_db}.triplets for sample_id={args.sample_id}")


if __name__ == "__main__":
    main()
