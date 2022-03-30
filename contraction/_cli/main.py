import click
import time
from pathlib import Path
from typing import Optional

from contraction._cli.parsing import ClickGraphCategory, ClickPath
from contraction._convert.convert import convert_image, save_graph
from contraction._convert.graph_category import GraphCategory
from contraction._convert.graph_metadata import GraphMetadata
from contraction._load.loader import load_graph
from contraction._solve.solver import Solver
from contraction._visual.display import Display
from contraction._train.training import train_model


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--debug/--no-debug', default=False)
@click.option('--category', type=ClickGraphCategory(), default=None)
@click.option('--group', type=str, default=None)
@click.option('--level', type=str, default=None)
def convert(
    data_dirpath: Path,
    debug: bool,
    category: Optional[GraphCategory],
    group: Optional[str],
    level: Optional[str],
) -> None:
    images_dirpath = data_dirpath / 'level-images'
    graphs_dirpath = data_dirpath / 'level-graphs'
    debug_dirpath = data_dirpath / 'level-debug' if debug else None
    for graph_metadata in GraphMetadata.iterator(category=category, group=group, level=level):
        graph_id = graph_metadata.graph_id
        print(f"Converting graph {graph_id}")
        G = convert_image(graph_id, images_dirpath, debug_dirpath)
        if G is not None:
            save_graph(G, graph_id, graphs_dirpath, zip_graph=False)


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--display-steps/--no-display-steps', default=False)
@click.option('--save-steps/--no-save-steps', default=False)
@click.option('--from-json/--from-gml', default=False)
@click.option('--category', type=ClickGraphCategory(), default=None)
@click.option('--group', type=str, default=None)
@click.option('--level', type=str, default=None)
def solve(
    data_dirpath: Path,
    display_steps: bool,
    save_steps: bool,
    from_json: bool,
    category: Optional[GraphCategory],
    group: Optional[str],
    level: Optional[str],
) -> None:
    for graph_metadata in GraphMetadata.iterator(category=category, group=group, level=level):
        graph_id = graph_metadata.graph_id
        print(f"Solving graph {graph_id}")

        solution_n_nodes = graph_metadata.solution_n_nodes
        if solution_n_nodes > 1:
            print(f"Solution with {solution_n_nodes} nodes not yet supported")
            return None

        G = load_graph(data_dirpath, graph_id, from_json=from_json)

        solutions_dirpath = data_dirpath / 'solutions'
        solver = Solver(solutions_dirpath, zip_graphs=False)
        start_time = time.time()
        solution = solver.solve(G, graph_id)
        end_time = time.time() - start_time

        if solution is None:
            print("No solution found")
            print(f"Processed in {end_time}s")
            return

        print(f"Solution: {solution}")
        print(f"Processed in {end_time}s")

        if display_steps or save_steps:
            display = Display(seed=0, iterations=250)
            display.draw_graph_grid(G, solution, title=f"Graph: {graph_id}")
            if save_steps:
                solution_image_filepath = data_dirpath / 'level-solution-images' / f'solution-{graph_id}.png'
                print(f"Saved solution image to {solution_image_filepath}")
                display.save(solution_image_filepath)
            if display_steps:
                display.show()


@cli.command(name='display')
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--from-json/--from-gml', default=False)
def display_graph(graph_id: str, data_dirpath: Path, from_json: bool) -> None:
    G = load_graph(data_dirpath, graph_id, from_json=from_json)
    display = Display(seed=0)
    display.draw_graph(G)
    display.show()


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--save/--no-save', default=True)
@click.option('--plot/--no-plot', default=True)
@click.option('--epochs', 'n_epochs', default=100)
def train(data_dirpath: Path, save: bool, plot: bool, n_epochs: int) -> None:
    train_model(data_dirpath, save, plot, n_epochs)
