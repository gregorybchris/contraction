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
        if color == cls.BLACK:
            return '#2c3e50'
        elif color == cls.BLUE:
            return '#3498db'
        elif color == cls.GREEN:
            return '#2ecc71'
        elif color == cls.ORANGE:
            return '#e67e22'
        elif color == cls.PINK:
            return '#ff7979'
        elif color == cls.PURPLE:
            return '#9b59b6'
        elif color == cls.RED:
            return '#e74c3c'
        elif color == cls.WHITE:
            return '#bdc3c7'
        elif color == cls.YELLOW:
            return '#f1c40f'
        else:
            raise ValueError(f"Unknown color {color}")

    @classmethod
    def from_str(cls, color_str: str) -> 'Color':
        return cls[color_str.upper()]

    @classmethod
    def str_to_hex(cls, color_str: str) -> str:
        return cls.to_hex(cls.from_str(color_str))
