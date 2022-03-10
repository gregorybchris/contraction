import networkx as nx

from contraction._lib.color import Color


def contract(G: nx.Graph, name: str, color: Color) -> None:
    if name not in G:
        raise ValueError(f"No node {name} in graph")

    # Update contraction root to have new color
    G.nodes[name]['color'] = color

    nodes_to_remove = set()
    edges_to_add = set()
    for child_name in G[name]:
        if G.nodes[child_name]['color'] == color:
            # Keep track of same-colored children for removal
            nodes_to_remove.add(child_name)

            # Keep track of children of same-colored children for adding edges
            for grandchild_name in G[child_name]:
                if grandchild_name != name:
                    edges_to_add.add((name, grandchild_name))

    # Add edges between contraction root and children of same-colored children
    G.add_edges_from(edges_to_add)

    # Remove same-colored children
    G.remove_nodes_from(nodes_to_remove)
