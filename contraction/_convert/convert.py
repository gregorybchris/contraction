import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

from contraction._solve import ops
from contraction._solve.color import Color
from contraction._convert.puzzle_metadata import get_max_contractions, get_n_colors

HAS_PILLOW = True
try:
    from PIL import Image
except ImportError:
    HAS_PILLOW = False

HAS_SKLEARN = True
try:
    from sklearn.cluster import KMeans
except ImportError:
    HAS_SKLEARN = False

N_COLS = 10
N_ROWS = 29
N_RGB = 3


def _rename_graph_nodes(G: nx.Graph):
    nodes = [node for node in G]
    sorted_nodes = sorted(nodes, key=lambda x: int(x))
    for i, node in enumerate(sorted_nodes):
        nx.relabel_nodes(G, {node: str(i)}, copy=False)


def _simplify_graph(G: nx.Graph) -> None:
    nodes = [node for node in G]
    for node in nodes:
        if node in G:
            size_1 = len(G)
            size_2 = 0
            while size_1 != size_2:
                contraction = node, G.nodes[node]['color']
                ops.contract(G, contraction, mutate=True)
                size_2 = size_1
                size_1 = len(G)


def _get_node_id(row: int, col: int):
    return str(row * N_COLS + col)


def _construct_graph(labels: np.ndarray, color_map: Dict[int, Color], max_contractions: int):
    G = nx.Graph(contractions=max_contractions)

    for row in range(N_ROWS):
        for col in range(N_COLS):
            label = labels[row, col]
            color = color_map[label]
            node = _get_node_id(row, col)
            G.add_node(node, color=color.value)
            even = (row + col) % 2 == 0

            if col > 0 and even:  # Left
                G.add_edge(node, _get_node_id(row, col - 1))

            if col < N_COLS - 1 and not even:  # Right
                G.add_edge(node, _get_node_id(row, col + 1))

            if row > 0:  # Up
                G.add_edge(node, _get_node_id(row - 1, col))

            if row < N_ROWS - 1:  # Down
                G.add_edge(node, _get_node_id(row + 1, col))

    return G


def _get_color_map(samples: np.ndarray, labels: np.ndarray) -> Dict[int, Color]:
    color_map = {}
    colors_used = set()
    for row in range(N_ROWS):
        for col in range(N_COLS):
            label = labels[row][col]
            if label not in color_map:
                sample = samples[row][col]

                closest_color = Color.BLACK
                closest_dist = sys.maxsize
                for color in Color:
                    if color not in colors_used:
                        dist = _color_dist(sample, color)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_color = color

                color_map[label] = closest_color
                colors_used.add(closest_color)

    return color_map


def _color_dist(color_a: np.ndarray, color_b: Color) -> float:
    color_b_rgb = Color.to_rgb(color_b)
    dr = color_a[0] - color_b_rgb[0]
    dg = color_a[1] - color_b_rgb[1]
    db = color_a[2] - color_b_rgb[2]
    return np.sqrt(dr**2 + dg**2 + db**2)


def _get_centers(image_width: int, image_height: int) -> np.ndarray:
    scale_width = 311
    scale_height = 640

    puzzle_offset_x = 15
    puzzle_offset_y = 24

    triangle_width = 31
    triangle_height = 18

    triangle_offset_x = 5

    centers = np.zeros((N_ROWS, N_COLS, 2), dtype=int)
    for row in range(N_ROWS):
        for col in range(N_COLS):
            # Compute approximate x, y coordinates for triangle
            x = puzzle_offset_x + col * triangle_width
            y = puzzle_offset_y + row * triangle_height

            # Handle triangle offsets
            x += -triangle_offset_x if (row + col) % 2 == 0 else triangle_offset_x

            # Handle top and bottom rows with only half-triangles
            if row == 0:
                y += triangle_height // 4
            if row == N_ROWS - 1:
                y -= triangle_height // 4

            centers[row, col, :] = (x, y)

    # Scale based on image size
    centers[:, :, 0] = centers[:, :, 0] * image_width // scale_width
    centers[:, :, 1] = centers[:, :, 1] * image_height // scale_height

    return centers


def _get_samples(image_array: np.ndarray, centers: np.ndarray, box_radius: int = 3) -> np.ndarray:
    samples = np.zeros((N_ROWS, N_COLS, N_RGB))
    for row in range(N_ROWS):
        for col in range(N_COLS):
            x, y = centers[row, col]
            sample = _take_sample(image_array, x, y, box_radius)
            samples[row, col, :] = sample
    return samples


def _get_labels(samples: np.ndarray, n_colors: int, n_init: int = 10, max_iter: int = 300) -> np.ndarray:
    samples = samples.reshape(N_ROWS * N_COLS, N_RGB)
    kmeans = KMeans(n_clusters=n_colors, n_init=n_init, max_iter=max_iter)
    kmeans.fit(samples)
    return kmeans.labels_.reshape(N_ROWS, N_COLS)


def _color_box(image_array: np.ndarray, x: int, y: int, radius: int, color: Color) -> None:
    for r in range(-radius, radius):
        for c in range(-radius, radius):
            image_array[y + r, x + c] = Color.to_rgb(color)


def _take_sample(image_array: np.ndarray, x: int, y: int, radius: int) -> Tuple[int, int, int]:
    total = np.zeros(N_RGB, dtype=int)
    n_pixels = 0
    for r in range(-radius, radius + 1):
        for c in range(-radius, radius + 1):
            n_pixels += 1
            total += image_array[y + r, x + c]
    return tuple(total // n_pixels)


def _get_debug_image(
    image_array: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    color_map: Dict[int, Color],
    box_radius: int,
) -> Image:
    for row in range(N_ROWS):
        for col in range(N_COLS):
            label = labels[row, col]
            x, y = centers[row, col]
            _color_box(image_array, x, y, box_radius, color_map[label])

    return Image.fromarray(image_array)


def _save_debug_image(image: Image, debug_dirpath: Path, graph_id: str):
    filename = f'kami-{graph_id}-debug.png'
    filepath = debug_dirpath / filename
    image.save(filepath)


def save_graph(G: nx.Graph, graph_id: str, graphs_dirpath: Path, zip_graph: bool = True) -> None:
    extension = 'gml.gz' if zip_graph else 'gml'
    output_filepath = graphs_dirpath / f'graph-{graph_id}.{extension}'
    nx.write_gml(G, output_filepath)


def convert_image(graph_id: str, images_dirpath: Path, debug_dirpath: Optional[Path]) -> None:
    if not HAS_PILLOW:
        raise ImportError("Pillow is required to convert images to graphs")
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required to convert images to graphs")

    image_filename = f'kami-{graph_id}.png'
    image_filepath = images_dirpath / image_filename
    image = Image.open(image_filepath)

    n_colors = get_n_colors(graph_id)
    max_contractions = get_max_contractions(graph_id)

    image_array = np.asarray(image)
    image_width, image_height = image.size

    centers = _get_centers(image_width, image_height)
    box_radius = image_width // 100
    samples = _get_samples(image_array, centers, box_radius=box_radius)
    labels = _get_labels(samples, n_colors)
    color_map = _get_color_map(samples, labels)

    G = _construct_graph(labels, color_map, max_contractions)
    _simplify_graph(G)
    _rename_graph_nodes(G)

    if debug_dirpath is not None:
        debug_image = _get_debug_image(image_array, labels, centers, color_map, box_radius)
        _save_debug_image(debug_image, debug_dirpath, graph_id)

    return G
