import click
import time
from pathlib import Path

from contraction._load.loader import load_graph
from contraction._gen.generator import generate_data
from contraction._visual.display import Display

# def run() -> None:
#     parser = argparse.ArgumentParser()
#     subparsers = parser.add_subparsers()

#     generate_parser = subparsers.add_parser('generate', help="Generate data", aliases=['gen'])
#     generate_parser.add_argument('--data', dest='data_dirpath', required=True, help="Data folder.")
#     generate_parser.set_defaults(func=generate)

#     display_parser = subparsers.add_parser('display', help="Display", aliases=['dis'])
#     display_parser.add_argument('level', help="Level ID to display.")
#     generate_parser.add_argument('--data', dest='data_dirpath', required=True, help="Data folder.")
#     display_parser.add_argument('--solution', help="Display full solution.")
#     generate_parser.set_defaults(func=display)

#     return parser.parse_args()


class ClickPath(click.Path):
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
def generate(data_dirpath: Path):
    for group in [1, 2, 3, 4, 5, 6, 7]:
        for level in [1, 2, 3, 4, 5, 6]:
            graph_id = f'{group}-{level}'
            print("\nProcessing graph: ", graph_id)
            graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'
            G, max_contractions = load_graph(graph_filepath)

            training_dirpath = data_dirpath / 'training' / f'graph-{graph_id}'
            training_dirpath.mkdir(exist_ok=True, parents=True)
            start_time = time.time()
            path = generate_data(G, training_dirpath, max_contractions, require_shortest=False, zip_graphs=True)
            end_time = time.time() - start_time
            print(f"Final path: {[(name, color.value) for name, color in path]}")
            print(f"Processed in {end_time}s")


@cli.command()
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
@click.option('--require-shortest/--no-require-shortest', default=False)
@click.option('--display-steps/--no-display-steps', default=False)
def solve(graph_id: str, data_dirpath: Path, require_shortest: bool, display_steps: bool):
    graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'
    G, max_contractions = load_graph(graph_filepath)

    graph_dirpath = data_dirpath / 'training' / f'graph-{graph_id}'
    graph_dirpath.mkdir(exist_ok=True, parents=True)
    start_time = time.time()
    path = generate_data(G, graph_dirpath, max_contractions, require_shortest=require_shortest, zip_graphs=True)
    end_time = time.time() - start_time
    print(f"Final path: {[(name, color.value) for name, color in path]}")
    print(f"Processed in {end_time}s")

    if display_steps:
        display = Display(seed=0)
        display.apply_contractions(G, path)


@cli.command()
@click.option('--graph-id', required=True)
@click.option('--data', 'data_dirpath', type=ClickPath(exists=True, file_okay=False, resolve_path=True), required=True)
def display(graph_id: str, data_dirpath: Path):
    graph_filepath = data_dirpath / 'graphs' / f'graph-{graph_id}.json'
    G, _ = load_graph(graph_filepath)
    display = Display(seed=0)
    display.draw_graph(G)
    display.show()
