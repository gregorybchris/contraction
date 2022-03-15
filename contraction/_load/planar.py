import random

import networkx as nx

from contraction._gen.color import Color


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
        node = str(node_id)
        color = _get_color()
        G.add_node(node, color=color)

    for _ in range(n_edges):
        node_a = str(random.randint(0, n_nodes - 1))
        node_b = str(random.randint(0, n_nodes - 1))
        G.add_edge(node_a, node_b)

    return G
