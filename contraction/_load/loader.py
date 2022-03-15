import json
from pathlib import Path
from typing import Tuple

import networkx as nx


def load_graph(filepath: Path) -> Tuple[nx.Graph, int]:
    G = nx.Graph()
    with filepath.open() as f:
        graph_data = json.load(f)
        n_contractions = graph_data['contractions']

        for node_data in graph_data['nodes']:
            name = str(node_data['id'])
            color = node_data['color']
            id = f"{name}-{color}"
            G.add_node(name, color=color, id=id)

        for node_data in graph_data['nodes']:
            name = str(node_data['id'])
            edges = [(name, str(edge)) for edge in node_data['edges']]
            G.add_edges_from(edges)

    return G, n_contractions
