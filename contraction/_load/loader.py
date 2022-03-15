import json
from pathlib import Path
from typing import Tuple

import networkx as nx


def load_graph(filepath: Path) -> Tuple[nx.Graph, int]:
    G = nx.Graph()
    with filepath.open() as f:
        graph_data = json.load(f)
        n_contractions = graph_data['contractions']

        for data in graph_data['nodes']:
            node = str(data['id'])
            color = data['color']
            id = f"{node}-{color}"
            G.add_node(node, color=color, id=id)

        for data in graph_data['nodes']:
            node = str(data['id'])
            edges = [(node, str(edge)) for edge in data['edges']]
            G.add_edges_from(edges)

    return G, n_contractions
