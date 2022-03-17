from queue import Queue
from typing import List, Optional

import networkx as nx


def contract(G: nx.Graph, node: str, color: str, mutate: bool = False) -> nx.Graph:
    if node not in G:
        raise ValueError(f"No node {node} in graph")

    if not mutate:
        G = G.copy()

    # Update contraction root to have new color
    G.nodes[node]['color'] = color

    nodes_to_remove = set()
    edges_to_add = set()
    for child_node in G[node]:
        if G.nodes[child_node]['color'] == color:
            # Keep track of same-colored children for removal
            nodes_to_remove.add(child_node)

            # Keep track of children of same-colored children for adding edges
            for grandchild_node in G[child_node]:
                if grandchild_node != node:
                    edges_to_add.add((node, grandchild_node))

    # Add edges between contraction root and children of same-colored children
    G.add_edges_from(edges_to_add)

    # Remove same-colored children
    G.remove_nodes_from(nodes_to_remove)

    return G


def get_markov_blanket(G: nx.Graph, node: str) -> List[str]:
    markov_blanket = set()
    markov_blanket.add(node)
    for child_node in G[node]:
        markov_blanket.add(node)
        for grandchild_node in G[child_node]:
            markov_blanket.add(grandchild_node)
    return list(markov_blanket)


def get_nodes(G: nx.Graph, markov_root: Optional[str] = None) -> List[str]:
    if markov_root is None:
        return [node for node in G]
    else:
        return get_markov_blanket(G, markov_root)


def order_nodes_by_centrality(G: nx.Graph, nodes: List[str], power: int = 2) -> List[str]:
    scores = {}
    queue = Queue()
    for node in nodes:
        score = 0
        visited_set = set()
        queue.put((node, 0))
        while not queue.empty():
            current_node, distance = queue.get()
            if current_node not in visited_set:
                score += distance**power
                visited_set.add(current_node)
                for child_node in G[current_node]:
                    queue.put((child_node, distance + 1))
        scores[node] = score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    sorted_nodes, _ = zip(*sorted_scores)
    return list(sorted_nodes)


def order_nodes_by_degree(G: nx.Graph, nodes: List[str]) -> List[str]:
    scores = [(node, len(G[node])) for node in nodes]
    sorted_scores = sorted(scores, key=lambda x: -x[1])
    sorted_nodes, _ = zip(*sorted_scores)
    return list(sorted_nodes)


def get_colors(G: nx.Graph, node: str, by_frequency: bool = False) -> List[str]:
    if not by_frequency:
        return list({G.nodes[child_node]['color'] for child_node in G[node]})

    frequencies = {}
    for child_node in G[node]:
        color = G.nodes[child_node]['color']
        if color not in frequencies:
            frequencies[color] = 0
        frequencies[color] += 1

    sorted_frequencies = sorted(frequencies.items(), key=lambda x: -x[1])
    sorted_nodes, _ = zip(*sorted_frequencies)
    return list(sorted_nodes)


def get_n_non_singular_colors(G: nx.Graph) -> int:
    frequencies = {}
    n_non_singular = 0
    for _, data in G.nodes(data=True):
        color = data['color']
        if color not in frequencies:
            frequencies[color] = 0
        frequencies[color] += 1

        if frequencies[color] == 2:
            n_non_singular += 1

    return n_non_singular
