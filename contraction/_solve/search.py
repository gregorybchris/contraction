from abc import abstractmethod, ABC
from typing import Generic, Iterator, Optional, TypeVar
from queue import Queue

NodeType = TypeVar('NodeType')
EdgeType = TypeVar('EdgeType')
ValueType = TypeVar('ValueType')


class Search(ABC, Generic[NodeType, EdgeType]):
    @abstractmethod
    def search(self, node: NodeType) -> Optional[NodeType]:
        pass

    @abstractmethod
    def iter_edges(self, node: NodeType) -> Iterator[EdgeType]:
        pass

    @abstractmethod
    def is_solution(self, node: NodeType) -> bool:
        pass

    @abstractmethod
    def apply_edge(self, node: NodeType, edge: EdgeType) -> NodeType:
        pass


class HeuristicSearch(Search, Generic[NodeType, EdgeType, ValueType]):
    @abstractmethod
    def evaluate(self, node: NodeType) -> ValueType:
        pass


class DepthFirstSearch(Search, Generic[NodeType, EdgeType]):
    def search(self, node: NodeType) -> Optional[NodeType]:
        if self.is_solution(node):
            return node

        for edge in self.iter_edges(node):
            next_node = self.apply_edge(node, edge)
            solution = self.search(next_node)
            if solution is not None:
                return solution
        return None


class BreadthFirstSearch(Search, Generic[NodeType, EdgeType]):
    def search(self, root: NodeType) -> Optional[NodeType]:
        queue = Queue()
        queue.put(root)

        while not queue.empty():
            node = queue.get()
            if self.is_solution(node):
                return node

            for edge in self.iter_edges(node):
                next_node = self.apply_edge(node, edge)
                queue.put(next_node)
        return None


class BeamSearch(HeuristicSearch, Generic[NodeType, EdgeType, ValueType]):
    def search(self, root: NodeType, beam_width: int, objective: str = 'min') -> Optional[NodeType]:
        if objective not in ['min', 'max']:
            raise ValueError(f"Invalid objective {objective}, must be 'min' or 'max'")

        queue = Queue()
        queue.put(root)

        while not queue.empty():
            node = queue.get()
            if self.is_solution(node):
                return node

            beam = []
            for edge in self.iter_edges(node):
                next_node = self.apply_edge(node, edge)
                value = self.evaluate(next_node)
                beam.append((next_node, value))

            reverse_sort = objective == 'max'
            sorted_beam = sorted(beam, key=lambda x: x[1], reverse=reverse_sort)
            for next_node, value in sorted_beam[:beam_width]:
                queue.put(next_node)

        return None


class BeamStackSearch(HeuristicSearch, Generic[NodeType, EdgeType, ValueType]):
    def search(self, node: NodeType, beam_width: int, objective: str = 'min') -> Optional[NodeType]:
        if self.is_solution(node):
            return node

        beam = []
        for edge in self.iter_edges(node):
            next_node = self.apply_edge(node, edge)
            value = self.evaluate(next_node)
            beam.append((next_node, value))

        reverse_sort = objective == 'max'
        sorted_beam = sorted(beam, key=lambda x: x[1], reverse=reverse_sort)
        for next_node, value in sorted_beam[:beam_width]:
            solution = self.search(next_node, beam_width=beam_width, objective=objective)
            if solution is not None:
                return solution
        return None
