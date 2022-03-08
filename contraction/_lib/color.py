from enum import Enum


class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

    @classmethod
    def get_hex(cls, color: 'Color') -> str:
        match color:
            case cls.RED:
                return "#e74c3c"
            case cls.GREEN:
                return "#2ecc71"
            case cls.BLUE:
                return "#3498db"
            case _:
                raise ValueError(f"Unknown color {color}")
