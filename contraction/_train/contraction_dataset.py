import json
import re
from pathlib import Path
from typing import Iterator, Union

import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

from contraction._solve.color import Color


class ContractionDataset(Dataset):
    def __init__(
        self,
        data_dirpath: Union[str, Path],
    ):
        super().__init__()
        self._data_dirpath = Path(data_dirpath)
        self._graph_filepaths = list(self._iter_graph_filepaths())

    def _iter_graph_filepaths(self) -> Iterator[Path]:
        for graph_dirpath in self._data_dirpath.iterdir():
            for solution_filepath in graph_dirpath.iterdir():
                if solution_filepath.name.startswith('graph'):
                    yield solution_filepath

    def __len__(self) -> int:
        return len(self._graph_filepaths)

    def __getitem__(self, idx: int) -> Data:
        # Get and transform graph
        graph_filepath = self._graph_filepaths[idx]
        G: nx.Graph = nx.read_gml(graph_filepath)
        colors = [c.value for c in Color]
        for node in G:
            color = G.nodes[node]['color']
            G.nodes[node]['color'] = colors.index(color)

        # Get number of contractions needed
        pattern = re.compile(r"^graph-(.*).gml")
        match = pattern.match(str(graph_filepath.name))
        graph_hash = match.group(1)
        solution_filepath = graph_filepath.parent / f'solution-{graph_hash}.json'

        with solution_filepath.open() as f:
            solution = json.load(f)
            n_contractions = len(solution['contractions'])

        # Populate Data object
        data = from_networkx(G, group_node_attrs=['color'])
        data.y = float(n_contractions)
        return data
