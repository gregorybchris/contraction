import re
from typing import Tuple, Union

from contraction._convert.graph_category import GraphCategory


def make_graph_id(category: Union[GraphCategory, str], group: str, level: str) -> str:
    if isinstance(category, str):
        category = GraphCategory.from_str(category)
    return f"{category.value}-{group}-{level}"


def parse_graph_id(graph_id: str) -> Tuple[GraphCategory, str, str]:
    pattern = re.compile(r"^([a-z]+)-([0-9x]+)-([0-9]+)$")
    match = pattern.match(graph_id)
    if match is None:
        raise ValueError(f"Invalid graph ID: {graph_id}")
    category = GraphCategory.from_str(match.group(1))
    group = match.group(2)
    level = match.group(3)
    return category, group, level
