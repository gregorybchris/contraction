from pprint import pprint

import networkx as nx

from contraction._lib.color import Color
from contraction._lib.display import draw_graphs
from contraction._lib.generator import generate_graph
from contraction._lib.ops import contract


def _get_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node('A', color=Color.RED)
    G.add_node('B', color=Color.BLUE)
    G.add_node('C', color=Color.GREEN)
    G.add_node('D', color=Color.BLUE)
    G.add_node('E', color=Color.RED)
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'C')
    return G


def run():
    G = _get_graph()
    # G = generate_graph(10, 7, 6)

    G1 = G.copy()
    pprint(list(G.adjacency()))
    contract(G, 'A', Color.BLUE)
    # contract(G, '0', Color.BLUE)
    pprint(list(G.adjacency()))
    draw_graphs([G1, G])
