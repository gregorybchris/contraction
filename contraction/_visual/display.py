from typing import List, Optional

import networkx as nx

from contraction._gen.color import Color
from contraction._gen.contraction import Contraction
from contraction._gen.ops import contract

HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MATPLOTLIB = False


class Display:
    def __init__(self, seed: Optional[int] = None, iterations: int = 50):
        self._figure_count = 0
        self._seed = seed
        self._iterations = iterations

    def draw_graph(self, G: nx.Graph, title: Optional[str] = None) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to draw graphs")

        if len(G) == 0:
            raise ValueError("Graph is empty")

        figure_title = title if title is not None else self._figure_count
        plt.figure(figure_title)
        plt.box(on=False)

        nodes = G.nodes(data=True)
        names, datas = zip(*nodes)
        colors = [Color.str_to_hex(data['color']) for data in datas]
        pos = nx.spring_layout(G, iterations=self._iterations, seed=self._seed)
        nx.draw_networkx_edges(G, pos, width=7, alpha=0.7, edge_color='#2c3e50')
        nx.draw_networkx_nodes(G, pos, nodelist=names, node_size=600, node_color=colors)
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=18)

        self._figure_count += 1

    def show(self) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to draw graphs")

        plt.show()

    def apply_contractions(self, G: nx.Graph, contractions: List[Contraction]) -> None:
        self.draw_graph(G, "Step 0: Initial")
        for name, color in contractions:
            G = contract(G, name, color)
            self.draw_graph(G, title=f"Step {self._figure_count}: {name} -> {color.value}")
        self.show()
