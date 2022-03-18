import json
from pathlib import Path
from typing import List, Optional

import networkx as nx

from contraction._solve.solution import Contraction, Solution
from contraction._solve import ops


class Solver:
    def __init__(self, solutions_dirpath: Optional[Path] = None, zip_graphs: bool = True):
        self._solutions_dirpath = solutions_dirpath
        self._zip_graphs = zip_graphs

    def solve(self, G: nx.Graph, graph_id: str, max_contractions: int) -> Optional[List[Contraction]]:
        solution = self._solve(G, graph_id, max_contractions, None)
        return None if solution is None else solution.to_list()

    def _solve(
        self,
        G: nx.Graph,
        graph_id: str,
        max_contractions: int,
        last_contraction: Optional[Contraction],
    ) -> Optional[Solution]:
        if len(G) == 1:
            return Solution()

        if ops.get_n_graph_colors(G) > max_contractions + 1:
            return None

        if ops.get_n_non_singular_colors(G) > max_contractions:
            return None

        contractions = list(ops.iter_contractions(G, last_contraction=last_contraction))
        contractions = ops.order_contractions_by_graph_size(G, contractions)
        for contraction in contractions:
            contracted = ops.contract(G, contraction)
            child_solution = self._solve(contracted, graph_id, max_contractions - 1, contraction)
            if child_solution is not None:
                solution = child_solution.push_front(contraction)
                self._save_solution(G, graph_id, solution)
                return solution

        return None

    def _save_solution(self, G: nx.Graph, graph_id: str, solution: Solution) -> None:
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
        for node, color in solution.to_list():
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
