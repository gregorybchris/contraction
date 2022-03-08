import networkx as nx


def contract(G: nx.Graph, node_name: str, node_color: str) -> nx.Graph:
    if node_name not in G:
        raise ValueError(f"No node {node_name} in graph")

    # children_edges = G[node_name]
    # children = list(children_edges.keys())
    # for child in children:
    #     grandchildren_edges = G[child]
    #     grandchildren = list(grandchildren_edges.keys())

    return G
