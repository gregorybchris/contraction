import click
import time
from pathlib import Path

from contraction._convert.convert import convert_image, save_graph
from contraction._convert.graph_id import make_graph_id
from contraction._load.loader import load_graph_from_gml, load_graph_from_json
from contraction._solve.solver import Solver
from contraction._visual.display import Display
from contraction._train.training import train_model


class ClickPath(click.Path):
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
def generate(data_dirpath: Path) -> None:
    solutions_dirpath = data_dirpath / 'solutions'
    solver = Solver(solutions_dirpath, zip_graphs=False)
    for group in [1, 2, 3, 4, 5, 6, 7, 8]:
        for level in [1, 2, 3, 4, 5, 6]:
            graph_id = f'{group}-{level}'
            print("\nProcessing graph: ", graph_id)

            graph_filepath = data_dirpath / 'level-graphs' / f'graph-{graph_id}.gml'
            G = load_graph_from_gml(graph_filepath)

            graph_filepath = data_dirpath / 'level-graphs' / f'graph-{graph_id}.gml'
            G = load_graph_from_gml(graph_filepath)

            start_time = time.time()
            solution = solver.solve(G, graph_id)
            end_time = time.time() - start_time
            print(f"Solution: {solution}")
            print(f"Processed in {end_time}s")


@cli.command()
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--display-steps/--no-display-steps', default=False)
@click.option('--from-json/--from-gml', default=False)
def solve(graph_id: str, data_dirpath: Path, display_steps: bool, from_json: bool) -> None:
    if from_json:
        graph_filepath = data_dirpath / '_old' / 'old-graphs' / f'graph-{graph_id}.json'
        G = load_graph_from_json(graph_filepath)
    else:
        graph_filepath = data_dirpath / 'level-graphs' / f'graph-{graph_id}.gml'
        G = load_graph_from_gml(graph_filepath)

    solutions_dirpath = data_dirpath / 'solutions'
    solver = Solver(solutions_dirpath, zip_graphs=True)
    start_time = time.time()
    solution = solver.solve(G, graph_id)
    end_time = time.time() - start_time
    print(f"Solution: {solution}")
    print(f"Processed in {end_time}s")

    if display_steps:
        display = Display(seed=0, iterations=250)
        display.draw_graph_grid(G, solution, title=f"Graph: {graph_id}")


@cli.command(name='display')
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--from-json/--from-gml', default=False)
def display_graph(graph_id: str, data_dirpath: Path, from_json: bool) -> None:
    if from_json:
        graph_filepath = data_dirpath / '_old' / 'old-graphs' / f'graph-{graph_id}.json'
        G = load_graph_from_json(graph_filepath)
    else:
        graph_filepath = data_dirpath / 'level-graphs' / f'graph-{graph_id}.gml'
        G = load_graph_from_gml(graph_filepath)

    display = Display(seed=0)
    display.draw_graph(G)
    display.show()


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--save/--no-save', default=True)
@click.option('--plot/--no-plot', default=True)
def train(data_dirpath: Path, save: bool, plot: bool) -> None:
    train_model(data_dirpath, save, plot)


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--debug/--no-debug', default=False)
def convert(data_dirpath: Path, debug: bool) -> None:
    images_dirpath = data_dirpath / 'level-images'
    graphs_dirpath = data_dirpath / 'level-graphs'
    debug_dirpath = data_dirpath / 'level-debug' if debug else None
    for group in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        for level in [1, 2, 3, 4, 5, 6]:
            graph_id = make_graph_id(group, level)
            print(graph_id)
            G = convert_image(graph_id, images_dirpath, debug_dirpath)
            save_graph(G, graph_id, graphs_dirpath)
