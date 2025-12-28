# Benchmark Builder (WDQS PoC)
from __future__ import annotations

import json
from typing import List

import streamlit as st
from pymongo import MongoClient

from utils.visualization import visualize_knowledge_graph
from utils.wdqs_benchmark import (
    BenchmarkRunConfig,
    graph_connectivity_metrics,
    load_edges_for_sample,
    relation_metrics_from_edges,
    run_benchmark,
)


st.set_page_config(page_title="Benchmark Builder", layout="wide")
st.title("Benchmark Builder (Wikidata WDQS PoC)")
st.caption(
    "Build a balanced KG benchmark directly from Wikidata (SPARQL) using the existing ontology PIDs. "
    "Text → LLM extraction remains unchanged; this is a separate path for benchmark generation."
)

DEFAULT_MONGO = "mongodb://localhost:63819/?directConnection=true"

# Hold last report for download/preview
if "benchmark_report" not in st.session_state:
    st.session_state["benchmark_report"] = None
if "benchmark_sample_id" not in st.session_state:
    st.session_state["benchmark_sample_id"] = "wdqs_poc_50x500"


with st.form("benchmark_builder_form"):
    st.subheader("1) Source & Storage")
    col1, col2 = st.columns(2)
    with col1:
        mongo_uri = st.text_input("Mongo URI", value=DEFAULT_MONGO)
        ontology_db = st.text_input("Ontology DB name", value="wikidata_ontology")
    with col2:
        triplets_db = st.text_input("Triplets DB name", value="triplets_db")
        sample_id = st.text_input("sample_id namespace", value=st.session_state["benchmark_sample_id"])

    st.subheader("2) Relation selection")
    relation_source = st.radio(
        "Relation source",
        options=["ontology_pids", "custom_pids"],
        format_func=lambda x: "Ontology PIDs (sampled)" if x == "ontology_pids" else "Custom PIDs (validated)",
        horizontal=True,
    )
    custom_pid_text = ""
    if relation_source == "ontology_pids":
        n_relations = st.number_input(
            "Number of relations (sampled from ontology)",
            min_value=1,
            max_value=2000,
            value=50,
            step=1,
        )
    else:
        n_relations = 50  # ignored for custom; kept for config completeness
        custom_pid_text = st.text_area(
            "Custom PIDs (comma or newline separated)",
            height=120,
            placeholder="P31\nP279\nP17",
            help="Each PID must exist in wikidata_ontology.properties. n_relations is ignored for this mode.",
        )

    st.subheader("3) Sampling & balancing")
    colA, colB, colC = st.columns(3)
    with colA:
        triples_per_relation = st.number_input(
            "Triples per relation (cap K)",
            min_value=1,
            max_value=2000,
            value=500,
            step=50,
            help="Keep this moderate (< ~500) to avoid WDQS timeouts; no paging is used in this PoC.",
        )
    with colB:
        balancing_mode = st.selectbox(
            "Balancing mode",
            options=["per_relation_cap", "uniform_relations"],
            help="per_relation_cap: cap K per relation. uniform_relations: aim for exactly K per relation where available.",
        )
    with colC:
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=13, step=1)

    st.subheader("4) WDQS/runtime controls")
    colX, colY, colZ = st.columns(3)
    with colX:
        timeout_s = st.number_input("WDQS request timeout (seconds)", min_value=10, max_value=59, value=55, step=1)
    with colY:
        max_retries = st.number_input("Max retries on 429/5xx/timeouts", min_value=0, max_value=10, value=4, step=1)
    with colZ:
        clear_existing = st.checkbox("Clear existing triplets for this sample_id first", value=True)

    submitted = st.form_submit_button("Build benchmark")

if submitted:
    custom_pids: List[str] = []
    if relation_source == "custom_pids":
        custom_pids = [p.strip() for p in custom_pid_text.replace("\n", ",").split(",") if p.strip()]

    cfg = BenchmarkRunConfig(
        mongo_uri=mongo_uri.strip(),
        ontology_db=ontology_db.strip(),
        triplets_db=triplets_db.strip(),
        sample_id=sample_id.strip(),
        relation_source=relation_source,
        custom_pids=custom_pids,
        n_relations=int(n_relations),
        triples_per_relation=int(triples_per_relation),
        seed=int(seed),
        balancing_mode=balancing_mode,
        clear_existing=bool(clear_existing),
        timeout_s=int(timeout_s),
        max_retries=int(max_retries),
        sleep_base_s=1.5,
    )

    try:
        with st.spinner("Running WDQS benchmark build..."):
            report = run_benchmark(cfg)
        st.session_state["benchmark_report"] = report
        st.session_state["benchmark_sample_id"] = sample_id

        st.success("Benchmark build complete.")
        st.json(report.get("totals", {}))

        report_bytes = json.dumps(report, indent=2).encode("utf-8")
        st.download_button(
            "Download build report (JSON)",
            data=report_bytes,
            file_name=f"wdqs_benchmark_report__{cfg.sample_id}.json",
            mime="application/json",
        )

        with st.expander("Per-relation details"):
            st.dataframe(report.get("per_pid", []))
    except Exception as e:
        st.error(f"Build failed: {e}")

st.divider()

# Visualization
st.header("Visualize a built sample from MongoDB")
vis_mongo_uri = st.text_input(
    "Mongo URI for visualization",
    value=mongo_uri,
    key="vis_mongo_uri",
    help="Typically the same URI used above.",
)
vis_triplets_db = st.text_input("Triplets DB name", value=triplets_db, key="vis_triplets_db")
vis_sample_id = st.text_input(
    "sample_id to load",
    value=st.session_state.get("benchmark_sample_id", sample_id),
    key="vis_sample_id",
)
vis_limit = st.number_input(
    "Max edges to load for visualization",
    min_value=50,
    max_value=10_000,
    value=2000,
    step=250,
    key="vis_limit",
)

if st.button("Load and visualize", key="visualize_btn"):
    try:
        client = MongoClient(vis_mongo_uri)
        edges = load_edges_for_sample(client, vis_triplets_db, vis_sample_id, vis_limit)
        if not edges:
            st.warning("No triplets found for this sample_id.")
        else:
            highlight = {s for s, _, _ in edges} | {o for _, _, o in edges}
            st.success(f"Loaded {len(edges)} edges (showing up to {vis_limit}).")
            visualize_knowledge_graph(edges, highlight_entities=highlight)
    except Exception as e:
        st.error(f"Visualization failed: {e}")

st.divider()

# Metrics (balance + connectivity)
st.header("Graph stats (balance & connectivity)")
metrics_limit = st.number_input(
    "Max edges to load for metrics",
    min_value=100,
    max_value=50_000,
    value=5_000,
    step=500,
    key="metrics_limit",
    help="Use a manageable sample to avoid long runs; metrics use an undirected projection for weak connectivity.",
)

if st.button("Compute stats", key="metrics_btn"):
    try:
        client = MongoClient(vis_mongo_uri)
        edges = load_edges_for_sample(client, vis_triplets_db, vis_sample_id, metrics_limit)
        if not edges:
            st.warning("No triplets found for this sample_id (cannot compute stats).")
        else:
            rel_stats = relation_metrics_from_edges(edges)
            conn_stats = graph_connectivity_metrics(edges)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Relation balance")
                st.json(rel_stats)
            with col2:
                st.subheader("Connectivity (weak, undirected projection)")
                st.json(conn_stats)

            st.caption(
                "Metrics: per-relation counts (min/max/mean/stdev) and weak components/density are common graph diagnostics. "
                "References: Newman, 'Networks: An Introduction' (OUP 2010); Barabási, 'Network Science' (2016)."
            )
    except Exception as e:
        st.error(f"Stat computation failed: {e}")

st.divider()

if st.session_state.get("benchmark_report"):
    with st.expander("Last build report (full)"):
        st.json(st.session_state["benchmark_report"])
