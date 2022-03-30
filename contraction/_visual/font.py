import functools
import io
from enum import Enum
from urllib import request

HAS_PILLOW = True
try:
    from PIL import ImageFont
except ImportError:
    HAS_PILLOW = False


@functools.lru_cache
def _download_font(font_url: str) -> bytes:
    return request.urlopen(font_url).read()


class Font(Enum):
    ROBOTO = 'roboto'

    @classmethod
    def _get_url(cls, font: 'Font') -> str:
        url_map = {
            cls.ROBOTO: "https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Bold.ttf?raw=true",
        }
        if font in url_map:
            return url_map[font]
        raise ValueError(f"Unknown font {font}")

    @classmethod
    def pillow(cls, font: 'Font', *args, **kwargs) -> ImageFont:
        url = cls._get_url(font)
        font_data = _download_font(url)
        with io.BytesIO(font_data) as f:
            return ImageFont.truetype(f, *args, **kwargs, layout_engine=ImageFont.LAYOUT_BASIC)
