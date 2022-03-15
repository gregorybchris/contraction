import math
from typing import Dict, List, Optional, Tuple

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

    def draw_graph(
        self,
        G: nx.Graph,
        title: Optional[str] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to draw graphs")

        figure_title = title if title is not None else self._figure_count
        plt.figure(figure_title)
        plt.box(on=False)

        self._draw_graph(G, highlight_edges=highlight_edges)
        self._figure_count += 1

    def show(self) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to draw graphs")

        plt.show()

    def apply_contractions(self, G: nx.Graph, contractions: List[Contraction]) -> None:
        self.draw_graph(G, "Step 0: Initial")
        for node, color in contractions:
            G = contract(G, node, color)
            self.draw_graph(G, title=f"Step {self._figure_count}: {node} -> {color}")
        self.show()

    def draw_graph_grid(
        self,
        G: nx.Graph,
        contractions: List[Contraction],
        title: Optional[str] = None,
    ) -> None:
        n_contractions = len(contractions)
        grid_rows = 2
        grid_cols = math.ceil((n_contractions + 1) / grid_rows)

        px = 1 / plt.rcParams['figure.dpi']
        figure, axes = plt.subplots(figsize=(1200 * px, 800 * px))
        figure.canvas.set_window_title(title)
        figure.tight_layout()
        axes.autoscale_view(tight=True)

        arrow = u"\u279E"

        pos = None
        for node, color in contractions:
            subplot_id = int(f"{grid_rows}{grid_cols}{self._figure_count + 1}")
            plt.subplot(subplot_id, title=f"Step {self._figure_count + 1}: ({node}) {arrow} {color}")
            pos = self._draw_graph(G, pos=pos, highlight_edges=None)
            self._figure_count += 1
            G = contract(G, node, color)

        subplot_id = int(f"{grid_rows}{grid_cols}{self._figure_count + 1}")
        plt.subplot(subplot_id, title=f"Step {self._figure_count + 1}: Solved")
        self._draw_graph(G, pos=pos, highlight_edges=None)

        plt.tight_layout()
        self.show()

    def _draw_graph(
        self,
        G: nx.Graph,
        pos: Optional[Dict[str, np.ndarray]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, np.ndarray]:
        if len(G) == 0:
            raise ValueError("Graph is empty")

        highlight_edges = [] if highlight_edges is None else highlight_edges
        # highlight_nodes = [node for node in G][:2]
        # highlight_edges = [(highlight_nodes[0], highlight_nodes[1])] if len(G) > 1 else []

        nodes = G.nodes(data=True)
        nodes, datas = zip(*nodes)
        colors = [Color.str_to_hex(data['color']) for data in datas]
        pos = nx.spring_layout(G, iterations=self._iterations, seed=self._seed, pos=pos)
        nx.draw_networkx_edges(G, pos, width=6, alpha=0.4, edge_color='#535c68')
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=6, alpha=0.8, edge_color='#2f3640')

        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=400, node_color=colors)
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=10)
        return pos