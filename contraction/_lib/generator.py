import json
import random
from pathlib import Path
from typing import Tuple

import networkx as nx

from contraction._lib.color import Color


def _get_color() -> str:
    colors = [color for color in Color]
    pick = random.randint(0, len(Color) - 1)
    return colors[pick].value


def generate_graph(n_nodes: int, n_edges: int, n_colors: int) -> nx.Graph:
    max_n_colors = len(Color)
    if n_colors > max_n_colors:
        raise ValueError(f"Maximum number of colors is {max_n_colors}. Received {n_colors}")

    if n_nodes < 1:
        raise ValueError("Generated graph must have at least 1 node")

    G = nx.Graph()
    for node_id in range(n_nodes):
        node_name = str(node_id)
        color = _get_color()
        G.add_node(node_name, color=color)

    for _ in range(n_edges):
        node_name_a = str(random.randint(0, n_nodes - 1))
        node_name_b = str(random.randint(0, n_nodes - 1))
        G.add_edge(node_name_a, node_name_b)

    return G


def load_graph(filepath: Path) -> Tuple[nx.Graph, int]:
    G = nx.Graph()
    with filepath.open() as f:
        graph_data = json.load(f)
        n_contractions = graph_data['contractions']

        for node_data in graph_data['nodes']:
            name = str(node_data['id'])
            color = node_data['color']
            G.add_node(name, color=color)

        for node_data in graph_data['nodes']:
            name = str(node_data['id'])
            edges = [(name, str(edge)) for edge in node_data['edges']]
            G.add_edges_from(edges)

    return G, n_contractions
