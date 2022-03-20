import json
import re
from pathlib import Path
from typing import Iterator, Union

import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

from contraction._solve.color import Color


class ContractionDataset(Dataset):
    TRAIN_SPLIT = 'train'
    TEST_SPLIT = 'test'

    def __init__(
        self,
        data_dirpath: Union[str, Path],
        split: str = 'train',
    ):
        super().__init__()
        self._data_dirpath = Path(data_dirpath)
        self._split = split

        assert split in [self.TRAIN_SPLIT, self.TEST_SPLIT]

        self._graph_filepaths = list(self._iter_graph_filepaths())

    def _iter_graph_filepaths(self) -> Iterator[Path]:
        for graph_dirpath in self._data_dirpath.iterdir():

            pattern = re.compile(r"^graph-([0-9]+)-([0-9]+)")
            match = pattern.match(str(graph_dirpath.name))
            if match is None:
                continue
            graph_id_level = int(match.group(2))

            # Don't train on odd levels
            if self._split == self.TRAIN_SPLIT and graph_id_level % 2 == 1:
                continue

            # Don't test on even levels
            if self._split == self.TEST_SPLIT and graph_id_level % 2 == 0:
                continue

            for solution_filepath in graph_dirpath.iterdir():
                if solution_filepath.name.startswith('graph'):
                    yield solution_filepath

    def __len__(self) -> int:
        return len(self._graph_filepaths)

    @classmethod
    def graph_to_data(cls, G: nx.Graph) -> Data:
        G = G.copy()
        colors = [c.value for c in Color]
        for node in G:
            color = G.nodes[node]['color']
            G.nodes[node]['color'] = colors.index(color)

        return from_networkx(G, group_node_attrs=['color'])

    def __getitem__(self, idx: int) -> Data:
        # Get and transform graph
        graph_filepath = self._graph_filepaths[idx]
        G: nx.Graph = nx.read_gml(graph_filepath)

        # Get number of contractions needed
        pattern = re.compile(r"^graph-(.*).gml")
        match = pattern.match(str(graph_filepath.name))
        graph_hash = match.group(1)
        solution_filepath = graph_filepath.parent / f'solution-{graph_hash}.json'

        with solution_filepath.open() as f:
            solution = json.load(f)
            n_contractions = len(solution['contractions'])

        # Populate Data object
        data = self.graph_to_data(G)
        data.y = float(n_contractions)
        return data
