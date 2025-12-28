# --- File: 0_KG_Extraction.py ---
import streamlit as st
from pyvis.network import Network
# import networkx as nx
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from utils.structured_inference_with_db import StructuredInferenceWithDB
from utils.structured_aligner import Aligner
from utils.openai_utils import LLMTripletExtractor
from pymongo import MongoClient
import uuid
import logging
import sys
import base64

# Configure logging
logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger('KGExtraction')
logger.setLevel(logging.INFO)


# Ensure the same user_id across all pages
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
logger.info(f"User ID: {user_id}")

st.set_page_config(
    page_title="Wikontic",
    page_icon="media/wikotic-wo-text.png",
    layout="wide"
)

WIKIDATA_ONTOLOGY_DB_NAME = "wikidata_ontology"
TRIPLETS_DB_NAME = "triplets_db"
# --- Mongo Setup ---
_ = load_dotenv(find_dotenv())
mongo_client = MongoClient(os.getenv("MONGO_URI"))
ontology_db = mongo_client.get_database(WIKIDATA_ONTOLOGY_DB_NAME)
triplets_db = mongo_client.get_database(TRIPLETS_DB_NAME)

# --- Extractor Setup ---
# extractor = LLMTripletExtractor(model=selected_model)
aligner = Aligner(ontology_db=ontology_db, triplets_db=triplets_db)
# inference_with_db = StructuredInferenceWithDB(extractor=extractor, aligner=aligner, triplets_db=triplets_db)


def fetch_related_triplets(entities):
    collection = triplets_db.get_collection('triplets')
    query = {"$or": [
                {"subject": {"$in": entities}},
                {"object": {"$in": entities}}
            ],
            "sample_id": user_id
        }
    results = collection.find(query, {"_id": 0, "subject": 1, "relation": 1, "object": 1})
    return [(doc["subject"], doc["relation"], doc["object"]) for doc in results]



# --- Visualize ---
def visualize_knowledge_graph(triplets, highlight_entities=None):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for s, r, o in triplets:
        for node in [s, o]:
            if node not in added_nodes:
                net.add_node(node, label=node,
                             color="#B2CD9C" if node in highlight_entities else "#C7C8CC")
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        # graph_container.components.v1.html(f.read(), height=600, scrolling=True)
        # with expanded_kg_container:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)


def visualize_initial_knowledge_graph(initial_triplets):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)

    for t in initial_triplets:
        print(t)
        s, r, o = t['subject'], t['relation'], t['object']
        net.add_node(s, label=s, color="#B2CD9C")
        net.add_node(o, label=o, color="#B2CD9C")
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name
    with open(html_path, "r", encoding="utf-8") as f:
        # graph_container.components.v1.html(f.read(), height=600, scrolling=True)
        # with initial_kg_container:
        st.components.v1.html(f.read(), height=600, scrolling=True)

    os.remove(html_path)

# --- UI ---
with open("media/wikontic.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

# Embed in header using HTML + Markdown
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
        <h1 style="margin: 0;">KG Extraction + Visualization</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "KG Construction Mode",
    ["From Text (LLM)", "From Triplets (Dataset)"]
)

if mode == "From Text (LLM)":
    st.info("Extract triplets from unstructured text using LLMs and visualize the knowledge graph.")

    model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
    selected_model = st.selectbox("Choose a model for KG extraction:", model_options, index=0)

    input_text = st.text_area("Enter Text", placeholder="Paste your text here...")
    trigger = st.button("Extract and Visualize")

    if trigger:
        if not input_text:
            st.warning("Please enter a text to extract KG.")
        elif not selected_model:
            st.warning("Please select a model for KG extraction.")
        else:
            extractor = LLMTripletExtractor(model=selected_model)
            inference_with_db = StructuredInferenceWithDB(extractor=extractor, aligner=aligner, triplets_db=triplets_db)
            initial_triplets, final_triplets, filtered_triplets, ontology_filtered_triplets = inference_with_db.extract_triplets_with_ontology_filtering(input_text, sample_id=user_id)
            if len(initial_triplets) > 0:
                aligner.add_initial_triplets(initial_triplets, sample_id=user_id)
            if len(final_triplets) > 0:
                aligner.add_triplets(final_triplets, sample_id=user_id)
            if len(filtered_triplets) > 0:
                aligner.add_filtered_triplets(filtered_triplets, sample_id=user_id)
            if len(ontology_filtered_triplets) > 0:
                aligner.add_ontology_filtered_triplets(ontology_filtered_triplets, sample_id=user_id)
            print("Initial triplets: ", initial_triplets)
            print()
            print("Refined triplets: ", final_triplets)
            print()
            print("filtered_triplets: ", filtered_triplets)
            print()
            print("ontology_filtered_triplets: ", ontology_filtered_triplets)
            # insert_triplets_to_neo4j(refined_triplets)
            new_entities = {t["subject"] for t in final_triplets} | {t["object"] for t in final_triplets}
            subgraph = fetch_related_triplets(list(new_entities))
            # st.session_state.kg = nx.DiGraph()
            # for s, r, o in subgraph:
                # st.session_state.kg.add_edge(s, o, label=r, highlight=s in new_entities or o in new_entities)
            st.success(f"✅ Extracted {len(final_triplets)} triplets and visualized {len(subgraph)} related ones.")

            col1, col2 = st.columns(2)
            
            with col1:
                # initial_kg_container = st.empty()
                # with initial_kg_container:
                    st.subheader("Extracted Triplets")
                    visualize_initial_knowledge_graph(initial_triplets)

            with col2:
                # expanded_kg_container = st.empty()
                # with expanded_kg_container:
                    st.subheader("Expanded KG Subgraph")
                    visualize_knowledge_graph(subgraph, highlight_entities=new_entities)
else:
    st.info("Upload pre-existing triplets (e.g., Wikidata QID/PID/QID) and visualize as a KG. No LLM extraction is used.")

    # --- Upload ---
    uploaded = st.file_uploader(
        "Upload Triplets File",
        type=["csv", "jsonl"],
        help="CSV must have columns: subject,relation,object. JSONL must have keys: subject,relation,object per line."
    )

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        max_rows = st.number_input("Max rows to ingest", min_value=1, max_value=5_000_000, value=50_000, step=1000)
    with colB:
        preview_rows = st.number_input("Preview rows", min_value=1, max_value=200, value=20, step=5)
    with colC:
        st.caption("Tip: Start with a small file (e.g., 1k–10k triplets) to validate schema and visualization.")

    # --- Optional: Clear previous run for this sample_id ---
    clear_before_insert = st.checkbox("Clear previous triplets for this run (sample_id)", value=True)

    ingest = st.button("Ingest Triplets and Visualize")

    def _normalize_row(row: dict) -> dict:
        """Normalize keys and trim whitespace."""
        s = str(row.get("subject", "")).strip()
        r = str(row.get("relation", "")).strip()
        o = str(row.get("object", "")).strip()
        return {"subject": s, "relation": r, "object": o}

    def _validate_row(t: dict) -> bool:
        """Basic validation: non-empty and looks like Wikidata IDs if you use Q/P IDs."""
        if not t["subject"] or not t["relation"] or not t["object"]:
            return False
        # Optional: enforce Q/P pattern (comment out if you ingest non-Wikidata datasets)
        # if not (t["subject"].startswith("Q") and t["relation"].startswith("P") and t["object"].startswith("Q")):
        #     return False
        return True

    def _load_triplets(file) -> list[dict]:
        import json
        import pandas as pd

        name = file.name.lower()
        triplets = []

        if name.endswith(".csv"):
            df = pd.read_csv(file)
            expected = {"subject", "relation", "object"}
            if not expected.issubset(set(df.columns)):
                raise ValueError(f"CSV must contain columns {sorted(expected)}, but got {list(df.columns)}")

            for _, row in df.iterrows():
                t = _normalize_row(row.to_dict())
                if _validate_row(t):
                    triplets.append(t)
                if len(triplets) >= max_rows:
                    break

        elif name.endswith(".jsonl"):
            # Stream JSONL line by line
            for line in file.getvalue().decode("utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                t = _normalize_row(obj)
                if _validate_row(t):
                    triplets.append(t)
                if len(triplets) >= max_rows:
                    break
        else:
            raise ValueError("Unsupported file type.")

        return triplets

    if uploaded is not None:
        try:
            # Preview without ingesting
            triplets_preview = _load_triplets(uploaded)
            st.subheader("Preview (normalized, validated)")
            st.write(triplets_preview[: int(preview_rows)])
            st.caption(f"Previewing {min(len(triplets_preview), int(preview_rows))} / {len(triplets_preview)} rows (after validation).")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    if ingest:
        if uploaded is None:
            st.warning("Please upload a CSV or JSONL file first.")
        else:
            try:
                # Re-read (uploader stream may have been consumed)
                uploaded.seek(0)
                triplets = _load_triplets(uploaded)

                if not triplets:
                    st.warning("No valid triplets found after parsing/validation.")
                else:
                    triplets_col = triplets_db.get_collection("triplets")

                    if clear_before_insert:
                        triplets_db.get_collection("triplets").delete_many({"sample_id": user_id})

                    # Insert in batches
                    BATCH = 5000
                    docs = []
                    inserted = 0

                    for t in triplets:
                        docs.append({
                            "subject": t["subject"],
                            "relation": t["relation"],
                            "object": t["object"],
                            "sample_id": user_id
                        })
                        if len(docs) >= BATCH:
                            triplets_col.insert_many(docs, ordered=False)
                            inserted += len(docs)
                            docs = []
                    if docs:
                        triplets_col.insert_many(docs, ordered=False)
                        inserted += len(docs)

                    st.success(f"✅ Ingested {inserted} triplets into triplets_db.triplets (sample_id={user_id}).")

                    # Build a small graph from the ingested edges for display
                    # Use first N edges to keep visualization responsive
                    N_SHOW = min(200, len(triplets))
                    extracted_edges = [(t["subject"], t["relation"], t["object"]) for t in triplets[:N_SHOW]]

                    # Highlight entities from ingested sample
                    new_entities = {t["subject"] for t in triplets[:N_SHOW]} | {t["object"] for t in triplets[:N_SHOW]}

                    # Expand via your existing function (if it reads from MongoDB, it can now retrieve these)
                    subgraph = fetch_related_triplets(list(new_entities))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Ingested Triplets (sample view)")
                        visualize_initial_knowledge_graph([
                            {"subject": s, "relation": r, "object": o, "subject_type": "", "object_type": ""}
                            for (s, r, o) in extracted_edges
                        ])

                    with col2:
                        st.subheader("Expanded KG Subgraph")
                        visualize_knowledge_graph(subgraph, highlight_entities=new_entities)

            except Exception as e:
                st.error(f"Ingestion failed: {e}")


st.subheader("Visualize an existing KG from DB")

sample_id_in = st.text_input("sample_id to visualize", value="bench_poc_001")
load_btn = st.button("Load and visualize")

if load_btn:
    triplets_col = triplets_db.get_collection("triplets")

    # Load a small sample for display
    docs = list(triplets_col.find(
        {"sample_id": sample_id_in},
        {"_id": 0, "subject": 1, "relation": 1, "object": 1}
    ).limit(500))

    if not docs:
        st.warning("No triplets found for this sample_id.")
    else:
        edges = [(d["subject"], d["relation"], d["object"]) for d in docs]
        highlight_entities = {d["subject"] for d in docs} | {d["object"] for d in docs}

        # Optional: expand using your existing helper (if it queries the DB)
        subgraph = fetch_related_triplets(list(highlight_entities))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loaded triplets (sample)")
            visualize_knowledge_graph(edges, highlight_entities=highlight_entities)
        with col2:
            st.subheader("Expanded subgraph")
            visualize_knowledge_graph(subgraph, highlight_entities=highlight_entities)

                        