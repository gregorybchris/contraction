import math
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

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

        figure_title = title if title is not None else self._figure_count
        plt.figure(figure_title)
        plt.box(on=False)

        self._draw_graph(G)
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

    def draw_graph_grid(self, G: nx.Graph, contractions: List[Contraction]) -> None:
        px = 1 / plt.rcParams['figure.dpi']
        n_contractions = len(contractions)
        # grid_size = math.ceil(math.sqrt(n_contractions + 1))
        grid_rows = 2
        grid_cols = math.ceil((n_contractions + 1) / grid_rows)
        figure, axes = plt.subplots(figsize=(1200 * px, 800 * px))
        figure.tight_layout()
        axes.autoscale_view(tight=True)
        # axes.set_axis_off()
        subplot_id = int(f"{grid_rows}{grid_cols}{1}")
        plt.subplot(subplot_id, title="Step 0: Initial")
        pos = self._draw_graph(G)
        self._figure_count += 1
        for i, (name, color) in enumerate(contractions):
            G = contract(G, name, color)
            subplot_id = int(f"{grid_rows}{grid_cols}{i + 2}")
            plt.subplot(subplot_id, title=f"Step {self._figure_count}: {name} -> {color.value}")
            pos = self._draw_graph(G, pos=pos)
            self._figure_count += 1
        plt.tight_layout()
        self.show()

    def _draw_graph(self, G: nx.Graph, pos: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        if len(G) == 0:
            raise ValueError("Graph is empty")

        nodes = G.nodes(data=True)
        names, datas = zip(*nodes)
        colors = [Color.str_to_hex(data['color']) for data in datas]
        pos = nx.spring_layout(G, iterations=self._iterations, seed=self._seed, pos=pos)
        nx.draw_networkx_edges(G, pos, width=6, alpha=0.7, edge_color='#2c3e50')
        nx.draw_networkx_nodes(G, pos, nodelist=names, node_size=400, node_color=colors)
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=10)
        return pos
