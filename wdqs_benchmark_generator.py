"""
CLI wrapper around the WDQS benchmark builder (same logic as the Streamlit page).

Example:
    python wdqs_benchmark_generator.py ^
        --mongo_uri "mongodb://localhost:63819/?directConnection=true" ^
        --ontology_db wikidata_ontology ^
        --triplets_db triplets_db ^
        --sample_id wdqs_cli_demo ^
        --n_relations 5 ^
        --triples_per_relation 50 ^
        --relation_source ontology_pids
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from utils.wdqs_benchmark import BenchmarkRunConfig, run_benchmark

DEFAULT_MONGO = "mongodb://localhost:63819/?directConnection=true"


def _split_pids(raw: str) -> List[str]:
    if not raw:
        return []
    parts = raw.replace("\n", ",").replace(" ", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def parse_args() -> Tuple[BenchmarkRunConfig, str]:
    ap = argparse.ArgumentParser(description="Generate a balanced KG benchmark from Wikidata WDQS.")

    ap.add_argument("--mongo_uri", default=DEFAULT_MONGO, help="MongoDB URI (same used by Streamlit).")
    ap.add_argument("--ontology_db", default="wikidata_ontology")
    ap.add_argument("--triplets_db", default="triplets_db")
    ap.add_argument("--sample_id", required=True, help="Namespace for this build; used in triplets.sample_id.")

    ap.add_argument("--relation_source", choices=["ontology_pids", "custom_pids"], default="ontology_pids")
    ap.add_argument(
        "--custom_pids",
        default="",
        help="Comma/newline/space separated PIDs if relation_source=custom_pids (validated against ontology).",
    )
    ap.add_argument("--n_relations", type=int, default=50, help="Only used for relation_source=ontology_pids.")

    ap.add_argument("--triples_per_relation", type=int, default=500)
    ap.add_argument("--balancing_mode", choices=["per_relation_cap", "uniform_relations"], default="per_relation_cap")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--clear_existing", action="store_true", help="Delete any existing docs for this sample_id first.")
    ap.add_argument("--timeout_s", type=int, default=55, help="Per-WDQS request timeout (must stay under 60s).")
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--sleep_base_s", type=float, default=1.5, help="Base sleep for exponential backoff.")

    ap.add_argument(
        "--output",
        default="",
        help="Optional path for the JSON report (default: wdqs_benchmark_report__<sample_id>.json)",
    )

    args = ap.parse_args()
    custom_pids = _split_pids(args.custom_pids)

    cfg = BenchmarkRunConfig(
        mongo_uri=args.mongo_uri,
        ontology_db=args.ontology_db,
        triplets_db=args.triplets_db,
        sample_id=args.sample_id,
        relation_source=args.relation_source,
        custom_pids=custom_pids,
        n_relations=args.n_relations,
        triples_per_relation=args.triples_per_relation,
        seed=args.seed,
        balancing_mode=args.balancing_mode,
        clear_existing=args.clear_existing,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        sleep_base_s=args.sleep_base_s,
    )

    return cfg, args.output


def main() -> None:
    cfg, output_override = parse_args()
    report = run_benchmark(cfg)

    out_path = Path(output_override or f"wdqs_benchmark_report__{cfg.sample_id}.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTotals:")
    print(json.dumps(report["totals"], indent=2))
    print(f"\nRelations used ({len(report['chosen_pids'])}): {', '.join(report['chosen_pids'])}")
    print(f"Report written to: {out_path}")
    print(f"sample_id='{cfg.sample_id}' now available in {cfg.triplets_db}.triplets")


if __name__ == "__main__":
    main()
