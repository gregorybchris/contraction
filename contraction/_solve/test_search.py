from typing import Iterator, List

# from contraction._solve.search import BreadthFirstSearch
# from contraction._solve.search import DepthFirstSearch
# from contraction._solve.search import BeamSearch
from contraction._solve.search import BeamStackSearch


class Tree:
    def __init__(self, data: int, children: List['Tree']):
        self.data = data
        self.children = {child.data: child for child in children}

    def __repr__(self) -> str:
        return f"{self.data}: {list(self.children.values())}"


class SearchNode:
    def __init__(self, tree: Tree, solution: List[int]):
        self.tree = tree
        self.solution = solution


# class TreeSearch(DepthFirstSearch[SearchNode, int]):
# class TreeSearch(BreadthFirstSearch[SearchNode, int]):
# class TreeSearch(BeamSearch[SearchNode, int, int]):
class TreeSearch(BeamStackSearch[SearchNode, int, int]):
    def iter_edges(self, search_node: SearchNode) -> Iterator[int]:
        for data in search_node.tree.children:
            yield data

    def is_solution(self, search_node: SearchNode) -> bool:
        return search_node.tree.data == 7

    def apply_edge(self, search_node: SearchNode, data: int) -> SearchNode:
        new_tree = search_node.tree.children[data]
        new_solution = search_node.solution.copy()
        new_solution.append(data)
        return SearchNode(new_tree, new_solution)

    def evaluate(self, search_node: SearchNode) -> int:
        return abs(7 - search_node.tree.data)


if __name__ == '__main__':
    tree = Tree(1, [
        Tree(2, [
            Tree(4, []),
            Tree(5, []),
        ]),
        Tree(3, [
            Tree(6, []),
            Tree(7, []),
        ]),
    ])

    root_node = SearchNode(tree, [tree.data])
    tree_search = TreeSearch()
    # result = tree_search.search(root_node)
    result = tree_search.search(root_node, beam_width=1)
    if result is None:
        print("No solution found")
    else:
        print(result.solution)
