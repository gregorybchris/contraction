import json
from pathlib import Path
from typing import List, Optional

import networkx as nx

from contraction._solve.contraction import Contraction
from contraction._solve.ops import contract
from contraction._solve.ops import get_nodes_by_degree
from contraction._solve.ops import get_colors
from contraction._solve.ops import get_n_non_singular_colors


class Solver:
    def __init__(self, solutions_dirpath: Path, zip_graphs: bool = True):
        self._solutions_dirpath = solutions_dirpath
        self._zip_graphs = zip_graphs

    def solve(self, G: nx.Graph, graph_id: str, max_contractions: Optional[int] = None) -> List[Contraction]:
        return self._solve(G, graph_id, [], max_contractions)

    def _solve(
        self,
        G: nx.Graph,
        graph_id: str,
        parent_path: List[Contraction],
        max_contractions: Optional[int],
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
        nodes = get_nodes_by_degree(G, markov_root=last_contracted_node)

        for node in nodes:
            # Get colors of node neighbors
            colors = get_colors(G, node, by_frequency=False)
            for color in colors:
                G_c = contract(G, node, color)
                path = parent_path.copy()
                path.append((node, color))
                child_max_contractions = None if max_contractions is None else max_contractions - 1
                child_path = self._solve(G_c, graph_id, path, child_max_contractions)
                if child_path is not None:
                    new_path = child_path.copy()
                    new_path.insert(0, (node, color))
                    self._save_graph(G, graph_id, new_path)
                    return new_path

        return None

    def _save_graph(self, G: nx.Graph, graph_id: str, solution: List[Contraction]) -> None:
        graph_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='id')

        # Save graph data
        extension = 'gml.gz' if self._zip_graphs else 'gml'
        solution_dirpath = self._solutions_dirpath / f'graph-{graph_id}'
        solution_dirpath.mkdir(exist_ok=True, parents=True)
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
