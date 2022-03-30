from pathlib import Path

import click

from contraction._convert.graph_category import GraphCategory


class ClickPath(click.Path):
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


class ClickGraphCategory(click.Path):
    def convert(self, value, param, ctx):
        return GraphCategory.from_str(super().convert(value, param, ctx))
