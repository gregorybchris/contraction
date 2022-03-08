import networkx as nx

from contraction._lib.display import draw_graph
from contraction._lib.ops import contract
from contraction._lib.color import Color


def run():
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
    draw_graph(G)
    print(G)
    G_c = contract(G, 'A', Color.BLUE)
    print(G_c)
    # draw_graph(G_c)
