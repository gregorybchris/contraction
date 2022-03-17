from typing import List, Optional, Tuple

Contraction = Tuple[str, str]


class ContractionPath(list):
    def __init__(self, initial_list: Optional[List[Contraction]] = None):
        super().__init__()
        self._list = [] if initial_list is None else initial_list

    def push_front(self, contraction: Contraction) -> 'ContractionPath':
        new_list = self._list.copy()
        new_list.insert(0, contraction)
        return ContractionPath(initial_list=new_list)

    def push_back(self, contraction: Contraction) -> 'ContractionPath':
        new_list = self._list.copy()
        new_list.append(contraction)
        return ContractionPath(initial_list=new_list)

    def to_list(self):
        return self._list
