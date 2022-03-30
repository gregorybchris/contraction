import csv
import functools
from pathlib import Path
from typing import Dict, Iterator, Optional

from contraction._convert.graph_id import make_graph_id, parse_graph_id
from contraction._convert.graph_category import GraphCategory


class GraphMetadata:
    def __init__(
        self,
        graph_id: str,
        max_contractions: int,
        n_colors: int,
        solution_n_nodes: int,
    ):
        self.graph_id = graph_id
        self.category, self.group, self.level = parse_graph_id(graph_id)
        self.max_contractions = max_contractions
        self.n_colors = n_colors
        self.solution_n_nodes = solution_n_nodes

    @classmethod
    def get(cls, graph_id: str) -> 'GraphMetadata':
        metadata_map = cls._load_metadata_map()
        if graph_id not in metadata_map:
            raise ValueError(f"No metadata found for graph_id {graph_id}")
        return metadata_map[graph_id]

    @classmethod
    def iterator(
        cls,
        category: Optional[GraphCategory] = None,
        group: Optional[str] = None,
        level: Optional[str] = None,
    ) -> Iterator['GraphMetadata']:
        metadata_map = cls._load_metadata_map()
        for _, graph_metadata in metadata_map.items():
            if category is not None and graph_metadata.category != category:
                continue
            if group is not None and graph_metadata.group != group:
                continue
            if level is not None and graph_metadata.level != level:
                continue
            yield graph_metadata

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
                metadata[graph_id] = GraphMetadata(graph_id, row[3], row[4], row[5])
        return metadata
