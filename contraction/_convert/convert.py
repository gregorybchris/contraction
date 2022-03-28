from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np

from contraction._convert.puzzle_metadata import (get_max_contractions, get_n_colors, get_n_node_solution)
from contraction._convert import convert_utils as utils
from contraction._solve.color import Color
from contraction._visual.font import Font

HAS_PILLOW = True
try:
    from PIL import Image, ImageDraw
except ImportError:
    HAS_PILLOW = False


def convert_image(graph_id: str, images_dirpath: Path, debug_dirpath: Optional[Path]) -> Optional[nx.Graph]:
    if not HAS_PILLOW:
        raise ImportError("Pillow is required to convert images to graphs")

    n_node_solution = get_n_node_solution(graph_id)
    if n_node_solution > 1:
        print(f"Solution with {n_node_solution} nodes not yet supported")
        return None

    image_filename = f'kami-{graph_id}.png'
    image_filepath = images_dirpath / image_filename
    image = Image.open(image_filepath)

    n_colors = get_n_colors(graph_id)
    max_contractions = get_max_contractions(graph_id)

    image_array = np.asarray(image)

    centers = utils.get_centers(image.size)
    box_radius = image.size[0] // 100
    samples = utils.get_samples(image_array, centers, box_radius=box_radius)
    mask = utils.get_mask(image_array, centers, box_radius=box_radius)

    labels = utils.get_labels(samples, n_colors, mask)
    color_map = utils.get_color_map(samples, labels, mask)

    G = utils.construct_graph(labels, color_map, max_contractions, mask)
    utils.simplify_graph(G)
    utils.rename_graph_nodes(G)

    if debug_dirpath is not None:
        debug_image = get_debug_image(G, image_array, labels, centers, color_map, box_radius, mask)
        save_debug_image(debug_image, debug_dirpath, graph_id)

    return G


def save_graph(G: nx.Graph, graph_id: str, graphs_dirpath: Path, zip_graph: bool = True) -> None:
    extension = 'gml.gz' if zip_graph else 'gml'
    output_filepath = graphs_dirpath / f'graph-{graph_id}.{extension}'
    print(f"Graph saved to {output_filepath}")
    nx.write_gml(G, output_filepath)


def get_debug_image(
    G: nx.Graph,
    image_array: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    color_map: Dict[int, Color],
    box_radius: int,
    mask: np.ndarray,
) -> Image:
    if not HAS_PILLOW:
        raise ImportError("Pillow is required to convert images to graphs")

    for row in range(utils.ConvertConstants.N_ROWS):
        for col in range(utils.ConvertConstants.N_COLS):
            if mask[row, col]:
                continue

            label = labels[row, col]
            x, y = centers[row, col]
            utils.color_box(image_array, x, y, box_radius, color_map[label])

    image = Image.fromarray(image_array)
    image_draw = ImageDraw.Draw(image)
    font_size = 30
    font = Font.pillow(Font.ROBOTO, font_size)
    for node in G:
        row, col = G.nodes[node]['position']
        center = centers[row, col]
        image_draw.text(center, node, fill=(0, 0, 0), font=font)

    return image


def save_debug_image(image: Image, debug_dirpath: Path, graph_id: str):
    filename = f'kami-{graph_id}-debug.png'
    filepath = debug_dirpath / filename
    print(f"Debug image saved to {filepath}")
    image.save(filepath)
