import networkx as nx

from contraction._lib.color import Color

HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MATPLOTLIB = False


class Display:
    def __init__(self):
        self._figure_count = 0

    def draw_graph(self, G: nx.Graph) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to draw graphs")

        if len(G) == 0:
            raise ValueError("Graph is empty")

        plt.figure(self._figure_count)
        plt.box(on=False)

        nodes = G.nodes(data=True)
        names, datas = zip(*nodes)
        colors = [Color.str_to_hex(data['color']) for data in datas]
        pos = nx.spring_layout(G, iterations=10, seed=0)
        nx.draw_networkx_edges(G, pos, width=7, alpha=0.7, edge_color='#2c3e50')
        nx.draw_networkx_nodes(G, pos, nodelist=names, node_size=600, node_color=colors)
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=18)

        self._figure_count += 1

    def show(self) -> None:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to draw graphs")

        plt.show()
