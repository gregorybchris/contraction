import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

from contraction._lib.color import Color
from contraction._lib.display import Display
from contraction._lib.generator import load_graph
from contraction._lib.ops import contract

Contraction = Tuple[str, Color]


def apply_contractions(
    G: nx.Graph,
    contractions: List[Contraction],
) -> None:
    display = Display()
    display.draw_graph(G)
    for name, color in contractions:
        G = contract(G, name, color)
        display.draw_graph(G)
    display.show()


def run():
    graph_id = '2-2'

    data_dirpath = Path(__file__).parent.parent.parent / 'data'
    graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'

    G, _ = load_graph(graph_filepath)

    training_dirpath = data_dirpath / 'training' / f'graph-{graph_id}'
    training_dirpath.mkdir(exist_ok=True, parents=True)
    path_map = {}
    path = generate_data(G, training_dirpath, path_map)
    print(f"Final path: {path}")

    # display = Display()
    # display.draw_graph(G)
    # display.show()

    contractions = [
        ('1', Color.GREEN),
        # ('0', Color.RED),
    ]

    apply_contractions(G, contractions)


def generate_data(
    G: nx.Graph,
    data_dirpath: Path,
    path_map: Dict[str, List[Contraction]],
    depth: int = 0,
) -> List[Contraction]:
    indent = '\t' * depth
    # indent = '\t' * (depth + 1)

    print(f"{indent}Generating data at depth {depth}")

    graph_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='color')
    if len(G) == 1:
        best_path = []
        print(f"{indent}Best path: {best_path} at depth {depth}")
        path_map[graph_hash] = best_path
        _save_graph(G, data_dirpath, graph_hash, best_path)
        return best_path

    if graph_hash in path_map:
        return path_map[graph_hash]

    colors = {G.nodes[node_id]['color'] for node_id in G}
    best_path = []
    for name in G.nodes:
        for color in colors:
            print(f"{indent}• {name}->{color}")
            if G.nodes[name]['color'] == color:
                print(f"{indent}Same color")
                continue

            G_c = contract(G, name, Color.from_str(color))
            child_path = generate_data(G_c, data_dirpath, path_map, depth=depth + 1)

            if len(best_path) == 0 or len(child_path) + 1 < len(best_path):
                print(f"{indent}Found better child path at depth {depth + 1}")
                print(f"{indent}From child path:", child_path)
                best_path = child_path.copy()
                best_path.insert(0, (name, Color.from_str(color)))
                print(f"{indent}New best child path:", best_path)
            else:
                print(f"{indent}Didn't find better child path")

    print(f"{indent}Best path: {best_path} at depth {depth}")
    path_map[graph_hash] = best_path
    _save_graph(G, data_dirpath, graph_hash, best_path)
    return best_path


def _save_graph(
    G: nx.Graph,
    data_dirpath: Path,
    graph_hash: str,
    path: List[Contraction],
) -> None:
    # Save graph data
    graph_filepath = data_dirpath / f'graph-{graph_hash}.gml'
    nx.write_gml(G, graph_filepath)

    # Save graph metadata
    meta_filepath = data_dirpath / f'meta-{graph_hash}.json'
    metadata = {
        'contractions': [[name, str(color)] for name, color in path],
    }
    with meta_filepath.open(mode='w') as f:
        json.dump(metadata, f)
