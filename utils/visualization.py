"""
Shared visualization helpers to avoid importing Streamlit pages (which can have side effects).
"""

import os
import tempfile
from typing import Iterable, Optional, Set, Tuple

import streamlit as st
from pyvis.network import Network


def visualize_knowledge_graph(
    triplets: Iterable[Tuple[str, str, str]],
    highlight_entities: Optional[Set[str]] = None,
    height: str = "600px",
) -> None:
    """
    Render a directed graph with optional highlighted nodes.
    Mirrors the implementation used on the KG Extraction page without pulling that page in.
    """
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    highlight_entities = highlight_entities or set()
    added_nodes = set()

    for s, r, o in triplets:
        for node in (s, o):
            if node not in added_nodes:
                net.add_node(
                    node,
                    label=node,
                    color="#B2CD9C" if node in highlight_entities else "#C7C8CC",
                )
                added_nodes.add(node)
        net.add_edge(s, o, label=r, color="#000000")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name

    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(html_path)

