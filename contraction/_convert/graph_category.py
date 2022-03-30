from enum import Enum


class GraphCategory(Enum):
    CHALLENGE = 'challenge'
    DAILY = 'daily'
    KAMI = 'kami'
    USER = 'user'

    @classmethod
    def from_str(cls, category_str: str) -> 'GraphCategory':
        return cls[category_str.upper()]
