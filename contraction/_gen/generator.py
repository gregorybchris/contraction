import json
from pathlib import Path
from typing import List, Optional

import networkx as nx

from contraction._gen.contraction import Contraction
from contraction._gen.ops import contract
# from contraction._gen.ops import get_markov_blanket
# from contraction._gen.ops import get_nodes_by_centrality
from contraction._gen.ops import get_nodes_by_degree
from contraction._gen.ops import get_colors
from contraction._gen.ops import get_n_non_singular_colors


def generate_data(
    G: nx.Graph,
    solution_dirpath: Path,
    max_contractions: Optional[int] = None,
    require_shortest: bool = True,
    zip_graphs: bool = True,
) -> List[Contraction]:
    return _generate_data(G, solution_dirpath, [], max_contractions, require_shortest, zip_graphs)


def _generate_data(
    G: nx.Graph,
    solution_dirpath: Path,
    parent_path: List[Contraction],
    max_contractions: Optional[int],
    require_shortest: bool,
    zip_graphs: bool,
) -> Optional[List[Contraction]]:
    if len(G) == 1:
        return []

    graph_colors = {G.nodes[node_id]['color'] for node_id in G.nodes}
    if max_contractions is not None and len(graph_colors) > max_contractions + 1:
        return None

    # If the number of colors that have more than one node is greater
    # than the number of allowed contractions, then there is no solution.
    if max_contractions is not None and get_n_non_singular_colors(G) > max_contractions:
        return None

    last_contracted_node = parent_path[-1][0] if len(parent_path) > 0 else None
    # nodes = get_markov_blanket(G, node=last_contracted_node)
    # nodes = get_nodes_by_centrality(G, markov_root=None)
    # nodes = get_nodes_by_centrality(G, markov_root=last_contracted_node)
    # nodes = get_nodes_by_degree(G, markov_root=None)
    nodes = get_nodes_by_degree(G, markov_root=last_contracted_node)

    best_path = None
    for node in nodes:
        # Get colors of node neighbors
        colors = get_colors(G, node, by_frequency=False)
        for color in colors:
            G_c = contract(G, node, color)
            contraction = node, color
            path = parent_path.copy()
            path.append(contraction)
            child_max_contractions = None if max_contractions is None else max_contractions - 1
            child_path = _generate_data(G_c, solution_dirpath, path, child_max_contractions, require_shortest,
                                        zip_graphs)
            if child_path is not None:
                new_path = child_path.copy()
                new_path.insert(0, (node, color))
                if require_shortest:
                    if best_path is None or len(child_path) + 1 < len(best_path):
                        best_path = new_path
                else:
                    _save_graph(G, solution_dirpath, new_path, zip=zip_graphs)
                    return new_path

    if best_path is not None:
        _save_graph(G, solution_dirpath, best_path, zip=zip_graphs)
    return best_path


def _save_graph(
    G: nx.Graph,
    solution_dirpath: Path,
    solution: List[Contraction],
    zip: bool = True,
) -> None:
    graph_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='id')

    # Save graph data
    extension = 'gml.gz' if zip else 'gml'
    graph_filepath = solution_dirpath / f'graph-{graph_hash}.{extension}'
    nx.write_gml(G, graph_filepath)

    # Save graph solution
    solution_filepath = solution_dirpath / f'solution-{graph_hash}.json'
    contractions = []
    for node, color in solution:
        contraction = {
            'node': node,
            'color': color,
        }
        contractions.append(contraction)
    solution_record = {
        'contractions': contractions,
    }
    with solution_filepath.open(mode='w') as f:
        json.dump(solution_record, f, indent=2)
