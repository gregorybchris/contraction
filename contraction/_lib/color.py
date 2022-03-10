from enum import Enum


class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'
    ORANGE = 'orange'
    PURPLE = 'purple'

    @classmethod
    def to_hex(cls, color: 'Color') -> str:
        match color:
            case cls.RED:
                return '#e74c3c'
            case cls.GREEN:
                return '#2ecc71'
            case cls.BLUE:
                return '#3498db'
            case cls.YELLOW:
                return '#f1c40f'
            case cls.ORANGE:
                return '#e67e22'
            case cls.PURPLE:
                return '#9b59b6'
            case _:
                raise ValueError(f"Unknown color {color}")
