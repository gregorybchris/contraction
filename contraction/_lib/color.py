from enum import Enum


class Color(Enum):
    BLACK = 'black'
    BLUE = 'blue'
    GREEN = 'green'
    ORANGE = 'orange'
    PURPLE = 'purple'
    RED = 'red'
    WHITE = 'white'
    YELLOW = 'yellow'

    @classmethod
    def to_hex(cls, color: 'Color') -> str:
        match color:
            case cls.BLACK:
                return '#2c3e50'
            case cls.BLUE:
                return '#3498db'
            case cls.GREEN:
                return '#2ecc71'
            case cls.ORANGE:
                return '#e67e22'
            case cls.PURPLE:
                return '#9b59b6'
            case cls.RED:
                return '#e74c3c'
            case cls.WHITE:
                return '#bdc3c7'
            case cls.YELLOW:
                return '#f1c40f'
            case _:
                raise ValueError(f"Unknown color {color}")

    @classmethod
    def from_str(cls, color_str: str) -> 'Color':
        return cls[color_str.upper()]

    @classmethod
    def str_to_hex(cls, color_str: str) -> str:
        return cls.to_hex(cls.from_str(color_str))
