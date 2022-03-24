from queue import Queue
from typing import Iterator, List, Optional, Tuple

import networkx as nx

from contraction._solve.types import Contraction
from contraction._train.contraction_model import ContractionModel
from contraction._train.contraction_dataset import ContractionDataset


def contract(G: nx.Graph, contraction: Contraction, mutate: bool = False) -> nx.Graph:
    node, color = contraction

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

    # Decrement max contractions
    G.graph['contractions'] -= 1

    return G


def _iter_markov_blanket(G: nx.Graph, node: str) -> Iterator[str]:
    markov_blanket = set()

    markov_blanket.add(node)
    yield node

    for child_node in G[node]:
        if child_node not in markov_blanket:
            yield child_node
        markov_blanket.add(child_node)
        for grandchild_node in G[child_node]:
            if grandchild_node not in markov_blanket:
                yield grandchild_node
            markov_blanket.add(grandchild_node)


def _iter_nodes(G: nx.Graph, markov_root: Optional[str] = None) -> Iterator[str]:
    if markov_root is None:
        for node in G:
            yield node
    else:
        for node in _iter_markov_blanket(G, markov_root):
            yield node


def _iter_nodes_by_centrality(G: nx.Graph, nodes: Iterator[str], power: int = 2) -> Iterator[str]:
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
    for node, _ in sorted(scores.items(), key=lambda x: x[1]):
        yield node


def _iter_nodes_by_degree(G: nx.Graph, nodes: Iterator[str]) -> Iterator[str]:
    scores = [(node, len(G[node])) for node in nodes]
    for node, _ in sorted(scores, key=lambda x: -x[1]):
        yield node


def iter_contractions(
    G: nx.Graph,
    last_contraction: Optional[Contraction] = None,
    order_by: Optional[str] = None,
    contraction_nodes: Optional[List[str]] = None,
) -> Iterator[Contraction]:
    markov_root = None if last_contraction is None else last_contraction[0]
    nodes = _iter_nodes(G, markov_root=markov_root)
    if contraction_nodes is not None:
        nodes = [node for node in contraction_nodes if node in G]

    if order_by is None:
        pass
    if order_by == 'centrality':
        nodes = _iter_nodes_by_centrality(G, nodes)
    elif order_by == 'degree':
        nodes = _iter_nodes_by_degree(G, nodes)
    else:
        raise ValueError("Invalid order_by type")

    for node in nodes:
        for color in _iter_neighbor_colors(G, node):
            if color != G.nodes[node]['color']:
                yield node, color


def iter_contractions_by_graph_size(
    G: nx.Graph,
    contractions: Iterator[Contraction],
) -> Iterator[Tuple[Contraction, nx.Graph]]:
    contracted_graphs = []
    for contraction in contractions:
        contracted = contract(G, contraction)
        contracted_graphs.append((contraction, contracted))

    for pair in sorted(contracted_graphs, key=lambda x: len(x[1])):
        yield pair


def iter_contractions_with_model(
    G: nx.Graph,
    contractions: Iterator[Contraction],
    model: ContractionModel,
) -> Iterator[Tuple[Contraction, nx.Graph]]:
    model.eval()
    predictions = []
    for contraction in contractions:
        contracted = contract(G, contraction)
        if len(contracted) == 1:
            predictions.append((contraction, 0, contracted))
            continue
        data = ContractionDataset.graph_to_data(contracted)
        prediction = model(data)
        predictions.append((contraction, prediction, contracted))

    for contraction, _, contracted in sorted(predictions, key=lambda x: x[1]):
        yield contraction, contracted


def _iter_neighbor_colors(G: nx.Graph, node: str) -> Iterator[str]:
    color_set = set()
    for child_node in G[node]:
        color = G.nodes[child_node]['color']
        if color not in color_set:
            color_set.add(color)
            yield color


def _get_neighbor_colors_by_freq(G: nx.Graph, node: str) -> Iterator[str]:
    frequencies = {}
    for child_node in G[node]:
        color = G.nodes[child_node]['color']
        if color not in frequencies:
            frequencies[color] = 0
        frequencies[color] += 1

    for node, _ in sorted(frequencies.items(), key=lambda x: -x[1]):
        yield node


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


def get_n_graph_colors(G: nx.Graph) -> int:
    graph_colors = {G.nodes[node_id]['color'] for node_id in G.nodes}
    return len(graph_colors)
