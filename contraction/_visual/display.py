import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from contraction._solve.color import Color
from contraction._solve.contraction import Contraction
from contraction._solve.ops import contract

HAS_MATPLOTLIB = True
try:
    import matplotlib
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

    def _get_highlight_edges(self, G: nx.Graph, node: str, color: str) -> List[Tuple[str, str]]:
        highlight_edges = []
        for child_node in G[node]:
            if G.nodes[child_node]['color'] == color:
                highlight_edges.append((node, child_node))
        return highlight_edges

    def draw_graph_grid(
        self,
        G: nx.Graph,
        contractions: List[Contraction],
        title: Optional[str] = None,
    ) -> None:
        n_contractions = len(contractions)
        grid_rows = 2 if n_contractions > 1 else 1
        grid_cols = math.ceil((n_contractions + 1) / grid_rows)

        px = 1 / plt.rcParams['figure.dpi']
        figure, axes = plt.subplots(figsize=(1200 * px, 800 * px))
        figure.canvas.set_window_title(title)
        figure.tight_layout()
        axes.autoscale_view(tight=True)
        plt.rc('axes.spines', bottom=False, top=False)
        plt.rc('axes', edgecolor='#E0E0E0', linewidth=5)
        plt.rc('axes', titlecolor='#404040', titleweight='bold')

        arrow = u"\u279E"

        pos = None
        highlight_edges = None
        for node, color in contractions:
            subplot_id = int(f"{grid_rows}{grid_cols}{self._figure_count + 1}")
            plt.subplot(subplot_id, title=f"Step {self._figure_count + 1}: ({node}) {arrow} {color}")
            highlight_edges = self._get_highlight_edges(G, node, color)
            pos = self._draw_graph(G, pos=pos, highlight_edges=highlight_edges)
            self._figure_count += 1
            G = contract(G, node, color)

        subplot_id = int(f"{grid_rows}{grid_cols}{self._figure_count + 1}")
        plt.subplot(subplot_id, title=f"Step {self._figure_count + 1}: Solved")
        self._draw_graph(G, pos=pos)

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

        nodes = G.nodes(data=True)
        nodes, datas = zip(*nodes)
        colors = [Color.str_to_hex(data['color']) for data in datas]

        pos = nx.spring_layout(G, iterations=self._iterations, seed=self._seed, pos=pos)

        nx.draw_networkx_edges(G, pos, width=4, alpha=0.5, edge_color='#535c68')
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=6, alpha=0.8, edge_color='#2f3640')

        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=400, node_color=colors)
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=9, font_weight='bold')

        return pos
