from contraction._convert.graph_id import parse_graph_id


def get_max_contractions(graph_id: str):
    group, level = parse_graph_id(graph_id)
    max_contractions = [
        [1, 2, 2, 2, 2, 3],
        [1, 2, 2, 3, 3, 4],
        [2, 3, 3, 3, 4, 5],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 4],
        [4, 4, 4, 4, 5, 5],
        [3, 3, 3, 3, 3, 3],
        [3, 3, 5, 4, 5, 5],
        [3, 3, 6, 9, 9, 7],
        [4, 5, 8, 8, 9, 7],
        [4, 6, 6, 6, 7, 11],
        [4, 4, 4, 5, 5, 6],
        [5, 6, 5, 7, 5, 6],
        [6, 5, 5, 9, 10, 9],
        [3, 5, 6, 5, 5, 5],
    ]
    if level < 1 or level > len(max_contractions[0]):
        raise ValueError(f"No contraction metadata for level {level}")
    if group < 1 or group > len(max_contractions):
        raise ValueError(f"No contraction metadata for group {group}")
    return max_contractions[group - 1][level - 1]


def get_n_colors(graph_id: str):
    group, level = parse_graph_id(graph_id)
    n_colors = [
        [2, 3, 3, 3, 3, 3],
        [2, 2, 3, 3, 3, 3],
        [3, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 3],
        [3, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [3, 3, 3, 3, 3, 3],
        [3, 3, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 4],
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
        [2, 3, 4, 4, 4, 4],
    ]
    if level < 1 or level > len(n_colors[0]):
        raise ValueError(f"No color metadata for level {level}")
    if group < 1 or group > len(n_colors):
        raise ValueError(f"No color metadata for group {group}")
    return n_colors[group - 1][level - 1]
