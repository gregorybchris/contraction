import json
from pathlib import Path
from typing import List, Optional

import networkx as nx
import torch

from contraction._solve.color import Color
from contraction._train.contraction_model import ContractionModel
from contraction._solve.solution import Contraction, Solution
from contraction._solve import ops


class Solver:
    def __init__(self, solutions_dirpath: Optional[Path] = None, zip_graphs: bool = True):
        self._solutions_dirpath = solutions_dirpath
        self._zip_graphs = zip_graphs

        self._model = self._load_model(solutions_dirpath)

    def _load_model(self, solutions_dirpath: Optional[Path]):
        if solutions_dirpath is None:
            raise ValueError("Must pass solutions_dirpath to load model")
        model_filepath = solutions_dirpath.parent / 'models' / 'model.pt'
        model = ContractionModel(len(Color))
        model.load_state_dict(torch.load(model_filepath))
        return model

    def solve(self, G: nx.Graph, graph_id: str) -> Optional[List[Contraction]]:
        solution = self._solve(G, graph_id, None)
        return None if solution is None else solution.to_list()

    def _solve(
        self,
        G: nx.Graph,
        graph_id: str,
        last_contraction: Optional[Contraction],
    ) -> Optional[Solution]:
        if ops.has_one_color(G):
            return Solution()

        if ops.get_n_graph_colors(G) > G.graph['contractions'] + 1:
            return None

        if ops.get_n_non_singular_colors(G) > G.graph['contractions']:
            return None

        contractions = ops.iter_contractions(G, last_contraction=last_contraction, order_by='centrality')
        # contractions = ops.iter_contractions_by_graph_size(G, contractions)
        # contractions = ops.iter_contractions_with_model(G, contractions, self._model)

        for contraction in contractions:
            # for contraction, contracted in contractions:
            contracted = ops.contract(G, contraction)
            child_solution = self._solve(contracted, graph_id, contraction)
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
        solution_dirpath = self._solutions_dirpath / graph_id
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
        print(f"Saved partial solution to {solution_filepath}")
