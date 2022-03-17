import json
from pathlib import Path
from typing import Iterator, List, Optional

import networkx as nx

from contraction._solve.contraction import Contraction
from contraction._solve import ops


class Solver:
    def __init__(self, solutions_dirpath: Optional[Path] = None, zip_graphs: bool = True):
        self._solutions_dirpath = solutions_dirpath
        self._zip_graphs = zip_graphs

    def solve(self, G: nx.Graph, graph_id: str, max_contractions: Optional[int] = None) -> Optional[List[Contraction]]:
        solution = self._solve(G, graph_id, max_contractions, None)
        return None if solution is None else solution

    def _solve(
        self,
        G: nx.Graph,
        graph_id: str,
        max_contractions: Optional[int],
        last_contraction: Optional[Contraction],
    ) -> Optional[List[Contraction]]:
        if len(G) == 1:
            return []

        if max_contractions is not None and ops.get_n_graph_colors(G) > max_contractions + 1:
            return None

        if max_contractions is not None and ops.get_n_non_singular_colors(G) > max_contractions:
            return None

        for contraction in self._iter_contractions(G, last_contraction):
            child_graph = ops.contract(G, contraction)
            child_max_contractions = None if max_contractions is None else max_contractions - 1
            child_solution = self._solve(child_graph, graph_id, child_max_contractions, contraction)
            if child_solution is not None:
                solution = child_solution.copy()
                solution.insert(0, contraction)
                self._save_solution(G, graph_id, solution)
                return solution

        return None

    def _iter_contractions(self, G: nx.Graph, last_contraction: Optional[Contraction]) -> Iterator[Contraction]:
        markov_root = None if last_contraction is None else last_contraction[0]
        nodes = ops.get_nodes(G, markov_root=markov_root)
        nodes = ops.order_nodes_by_centrality(G, nodes)

        for node in nodes:
            colors = ops.get_neighbor_colors(G, node)
            for color in colors:
                yield node, color

    def _save_solution(self, G: nx.Graph, graph_id: str, solution: List[Contraction]) -> None:
        if self._solutions_dirpath is None:
            return

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
