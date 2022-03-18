from pathlib import Path
from typing import Union

import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx


class ContractionDataset(Dataset):
    def __init__(
        self,
        data_dirpath: Union[str, Path],
    ):
        super().__init__()
        self._data_dirpath = Path(data_dirpath)
        self._graph_dirpaths = list(self._data_dirpath.iterdir())

    def __len__(self) -> int:
        return len(self._graph_dirpaths)

    def __getitem__(self, idx: int) -> Data:
        graph_dirpath = self._graph_dirpaths[0]
        # TODO: Read each graph from a level into its own Data object
        for filepath in graph_dirpath.iterdir():
            if filepath.name.startswith('graph'):
                G = nx.read_gml(filepath)
                # data = from_networkx(G, group_node_attrs=['color'])
                data = from_networkx(G)
                return data
