import re
from typing import Tuple


def parse_graph_id(graph_id: str) -> Tuple[int, int]:
    pattern = re.compile(r"^([0-9]+)-([0-9]+)$")
    match = pattern.match(graph_id)
    if match is None:
        raise ValueError(f"Invalid graph ID: {graph_id}")
    group = int(match.group(1))
    level = int(match.group(2))
    return group, level


def make_graph_id(group: int, level: int) -> str:
    return f"{group}-{level}"
