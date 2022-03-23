from enum import Enum
from typing import Tuple


class Color(Enum):
    BLACK = 'black'
    BLUE = 'blue'
    GRAY = 'gray'
    GREEN = 'green'
    ORANGE = 'orange'
    PINK = 'pink'
    PURPLE = 'purple'
    RED = 'red'
    YELLOW = 'yellow'

    @classmethod
    def to_hex(cls, color: 'Color') -> str:
        hex_map = {
            cls.BLACK: '#2c3e50',
            cls.BLUE: '#2980b9',
            cls.GRAY: '#b2bec3',
            cls.GREEN: '#27ae60',
            cls.ORANGE: '#e67e22',
            cls.PINK: '#ff7979',
            cls.PURPLE: '#8e44ad',
            cls.RED: '#e84118',
            cls.YELLOW: '#f1c40f',
        }
        if color in hex_map:
            return hex_map[color]
        raise ValueError(f"Unknown color {color}")

    @classmethod
    def to_rgb(cls, color: 'Color') -> Tuple[int, int, int]:
        rgb_map = {
            cls.BLACK: (52, 73, 94),
            cls.BLUE: (52, 152, 219),
            cls.GRAY: (178, 190, 195),
            cls.GREEN: (39, 174, 96),
            cls.ORANGE: (230, 126, 34),
            cls.PINK: (253, 121, 168),
            cls.PURPLE: (155, 89, 182),
            cls.RED: (232, 65, 24),
            cls.YELLOW: (241, 196, 15),
        }
        if color in rgb_map:
            return rgb_map[color]
        raise ValueError(f"Unknown color {color}")

    @classmethod
    def from_str(cls, color_str: str) -> 'Color':
        return cls[color_str.upper()]

    @classmethod
    def str_to_hex(cls, color_str: str) -> str:
        return cls.to_hex(cls.from_str(color_str))
