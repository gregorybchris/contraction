import click
import time
from pathlib import Path

from contraction._load.loader import load_graph
from contraction._solve.solver import Solver
from contraction._visual.display import Display


class ClickPath(click.Path):
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--require-shortest/--no-require-shortest', default=False)
def generate(data_dirpath: Path, require_shortest: bool):
    solutions_dirpath = data_dirpath / 'training'
    solver = Solver(solutions_dirpath, zip_graphs=True)
    for group in [1, 2, 3, 4, 5, 6, 7]:
        for level in [1, 2, 3, 4, 5, 6]:
            graph_id = f'{group}-{level}'
            print("\nProcessing graph: ", graph_id)
            graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'
            G, max_contractions = load_graph(graph_filepath)

            start_time = time.time()
            solution = solver.solve(G, graph_id, max_contractions)
            end_time = time.time() - start_time
            print(f"Solution: {[(node, color) for node, color in solution]}")
            print(f"Processed in {end_time}s")


@cli.command()
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--display-steps/--no-display-steps', default=False)
def solve(graph_id: str, data_dirpath: Path, display_steps: bool):
    graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'
    G, max_contractions = load_graph(graph_filepath)

    solutions_dirpath = data_dirpath / 'training'
    solver = Solver(solutions_dirpath, zip_graphs=True)
    start_time = time.time()
    solution = solver.solve(G, graph_id, max_contractions)
    end_time = time.time() - start_time
    print(f"Solution: {[(node, color) for node, color in solution]}")
    print(f"Processed in {end_time}s")

    if display_steps:
        display = Display(seed=0, iterations=250)
        display.draw_graph_grid(G, solution, title=f"Graph: {graph_id}")


@cli.command()
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
def display(graph_id: str, data_dirpath: Path):
    graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'
    G, _ = load_graph(graph_filepath)
    display = Display(seed=0)
    display.draw_graph(G)
    display.show()
