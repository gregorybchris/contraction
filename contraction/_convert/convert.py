from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np

from contraction._convert.graph_metadata import GraphMetadata
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

    image_filepath = images_dirpath / f'{graph_id}.png'
    print(f"Reading input image {image_filepath}")
    image = Image.open(image_filepath)
    image_array = np.asarray(image)

    graph_metadata = GraphMetadata.get(graph_id)

    box_radius = image.size[0] // 100
    normalizer_image_array = np.asarray(Image.open(images_dirpath / 'kami-1-1.png'))

    centers = utils.get_centers(image.size)
    mask = utils.get_mask(image_array, centers, box_radius)
    samples = utils.get_samples(image_array, centers, box_radius)
    samples = utils.normalize_samples(samples, centers, normalizer_image_array, box_radius)
    labels = utils.get_labels(samples, graph_metadata.n_colors, mask)
    color_map = utils.get_color_map(samples, labels, mask)

    G = utils.construct_graph(labels, color_map, graph_metadata.max_contractions, mask)
    utils.add_node_positions(G, mask)
    utils.simplify_graph(G)

    if debug_dirpath is not None:
        debug_image = get_debug_image(G, image_array, labels, centers, color_map, box_radius, mask)

        debug_image_filepath = debug_dirpath / f'{graph_id}-debug.png'
        debug_image.save(debug_image_filepath)
        print(f"Debug image saved to {debug_image_filepath}")

        debug_image_filepath = debug_dirpath / f'{graph_id}-color-debug.png'
        samples[mask] = 0
        debug_image = Image.fromarray(samples, mode='RGB')
        debug_image = debug_image.resize([d * 20 for d in debug_image.size], resample=Image.BOX)
        debug_image.save(debug_image_filepath)
        print(f"Debug color image saved to {debug_image_filepath}")

    utils.remove_node_positions(G)
    return G


def save_graph(G: nx.Graph, graph_id: str, graphs_dirpath: Path, zip_graph: bool = True) -> None:
    extension = 'gml.gz' if zip_graph else 'gml'
    output_filepath = graphs_dirpath / f'{graph_id}.{extension}'
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
