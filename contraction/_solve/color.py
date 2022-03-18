from enum import Enum


class Color(Enum):
    BLACK = 'black'
    BLUE = 'blue'
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
            cls.GREEN: '#27ae60',
            cls.ORANGE: '#e67e22',
            cls.PINK: '#ff7979',
            cls.PURPLE: '#8e44ad',
            cls.RED: '#e74c3c',
            cls.YELLOW: '#f1c40f',
        }
        if color in hex_map:
            return hex_map[color]
        raise ValueError(f"Unknown color {color}")

    @classmethod
    def from_str(cls, color_str: str) -> 'Color':
        return cls[color_str.upper()]

    @classmethod
    def str_to_hex(cls, color_str: str) -> str:
        return cls.to_hex(cls.from_str(color_str))
