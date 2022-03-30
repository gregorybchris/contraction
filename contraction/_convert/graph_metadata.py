import csv
import functools
from pathlib import Path
from typing import Dict

from contraction._convert.graph_id import make_graph_id


class GraphMetadata:
    def __init__(self, max_contractions: int, n_colors: int, solution_n_nodes: int):
        self.max_contractions = max_contractions
        self.n_colors = n_colors
        self.solution_n_nodes = solution_n_nodes

    @classmethod
    def get(cls, graph_id: str) -> 'GraphMetadata':
        metadata_map = cls._load_metadata_map()
        if graph_id not in metadata_map:
            raise ValueError(f"No metadata for graph_id {graph_id}")
        return metadata_map[graph_id]

    @classmethod
    @functools.lru_cache
    def _load_metadata_map(cls) -> Dict[str, 'GraphMetadata']:
        metadata_filepath = Path(__file__).parent / 'metadata' / 'metadata.csv'
        column_types = [str, str, str, int, int, int]
        metadata = {}
        with open(metadata_filepath) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                row = [column_type(value) for value, column_type in zip(row, column_types)]
                graph_id = make_graph_id(row[0], row[1], row[2])
                metadata[graph_id] = GraphMetadata(row[3], row[4], row[5])
        return metadata
