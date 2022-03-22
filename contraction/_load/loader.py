import json
from pathlib import Path

import networkx as nx


def load_graph_from_json(filepath: Path) -> nx.Graph:
    G = nx.Graph()
    with filepath.open() as f:
        graph_data = json.load(f)
        n_contractions = graph_data['contractions']
        G.graph['contractions'] = n_contractions

        for data in graph_data['nodes']:
            node = str(data['id'])
            color = data['color']
            id = f"{node}-{color}"
            G.add_node(node, color=color, id=id)

        for data in graph_data['nodes']:
            node = str(data['id'])
            edges = [(node, str(edge)) for edge in data['edges']]
            G.add_edges_from(edges)

    return G


def load_graph_from_gml(filepath: Path) -> nx.Graph:
    G: nx.Graph = nx.read_gml(filepath)
    for node in G:
        color = G.nodes[node]['color']
        id = f"{node}-{color}"
        G.nodes[node]['id'] = id
    return G
