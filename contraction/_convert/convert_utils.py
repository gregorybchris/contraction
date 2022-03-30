import sys
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from contraction._solve import ops
from contraction._solve.color import Color

HAS_SKLEARN = True
try:
    from sklearn.cluster import KMeans
except ImportError:
    HAS_SKLEARN = False


class ConvertConstants:
    N_COLS = 10
    N_ROWS = 29
    N_RGB = 3


def simplify_graph(G: nx.Graph) -> None:
    nodes = [node for node in G]
    max_contractions = G.graph['contractions']
    for node in nodes:
        if node in G:
            size_1 = len(G)
            size_2 = 0
            while size_1 != size_2:
                contraction = node, G.nodes[node]['color']
                ops.contract(G, contraction, mutate=True)
                size_2 = size_1
                size_1 = len(G)
    G.graph['contractions'] = max_contractions

    _rename_graph_nodes(G)


def _rename_graph_nodes(G: nx.Graph):
    nodes = [node for node in G]
    sorted_nodes = sorted(nodes, key=lambda x: int(x))
    for i, node in enumerate(sorted_nodes):
        nx.relabel_nodes(G, {node: str(i)}, copy=False)


def _get_node_id(row: int, col: int):
    return str(row * ConvertConstants.N_COLS + col)


def construct_graph(labels: np.ndarray, color_map: Dict[int, Color], max_contractions: int, mask: np.ndarray):
    G = nx.Graph(contractions=max_contractions)

    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            if mask[row, col]:
                continue

            label = labels[row, col]

            color = color_map[label]
            node = _get_node_id(row, col)
            G.add_node(node, color=color.value)
            even = (row + col) % 2 == 0

            if col > 0 and even and not mask[row, col - 1]:  # Left
                G.add_edge(node, _get_node_id(row, col - 1))

            if col < ConvertConstants.N_COLS - 1 and not even and not mask[row, col + 1]:  # Right
                G.add_edge(node, _get_node_id(row, col + 1))

            if row > 0 and not mask[row - 1, col]:  # Up
                G.add_edge(node, _get_node_id(row - 1, col))

            if row < ConvertConstants.N_ROWS - 1 and not mask[row + 1, col]:  # Down
                G.add_edge(node, _get_node_id(row + 1, col))

    return G


def add_node_positions(G: nx.Graph, mask: np.ndarray) -> None:
    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            if mask[row, col]:
                continue
            node = _get_node_id(row, col)
            G.nodes[node]['position'] = (row, col)


def remove_node_positions(G: nx.Graph) -> None:
    for node in G:
        data = G.nodes[node]
        if 'position' in data:
            del data['position']


def get_color_map(samples: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Dict[int, Color]:
    color_map = {}
    colors_used = set()
    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            if mask[row][col]:
                continue

            label = labels[row][col]
            if label not in color_map:
                sample = samples[row][col]

                closest_color = Color.BLACK
                closest_dist = sys.maxsize
                for color in Color:
                    if color not in colors_used:
                        dist = _color_dist(sample, Color.to_rgb(color))
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_color = color

                color_map[label] = closest_color
                colors_used.add(closest_color)

    return color_map


def _color_dist(color_a: np.ndarray, color_b: np.ndarray) -> float:
    dr = color_a[0] - color_b[0]
    dg = color_a[1] - color_b[1]
    db = color_a[2] - color_b[2]
    return np.sqrt(dr**2 + dg**2 + db**2)


def get_centers(image_size: Tuple[int, int]) -> np.ndarray:
    image_width, image_height = image_size

    scale_width = 311
    scale_height = 640

    puzzle_offset_x = 15
    puzzle_offset_y = 24

    triangle_width = 31
    triangle_height = 18

    triangle_offset_x = 5

    centers = np.zeros((ConvertConstants.N_ROWS, ConvertConstants.N_COLS, 2), dtype=int)
    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            # Compute approximate x, y coordinates for triangle
            x = puzzle_offset_x + col * triangle_width
            y = puzzle_offset_y + row * triangle_height

            # Handle triangle offsets
            x += -triangle_offset_x if (row + col) % 2 == 0 else triangle_offset_x

            # Handle top and bottom rows with only half-triangles
            if row == 0:
                y += triangle_height // 4
            if row == ConvertConstants.N_ROWS - 1:
                y -= triangle_height // 4

            centers[row, col, :] = (x, y)

    # Scale based on image size
    centers[:, :, 0] = centers[:, :, 0] * image_width // scale_width
    centers[:, :, 1] = centers[:, :, 1] * image_height // scale_height

    return centers


def get_samples(image_array: np.ndarray, centers: np.ndarray, box_radius: int) -> np.ndarray:
    samples = np.zeros((ConvertConstants.N_ROWS, ConvertConstants.N_COLS, ConvertConstants.N_RGB), dtype=np.uint8)
    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            x, y = centers[row, col]
            samples[row, col, :] = _take_sample(image_array, x, y, box_radius)
    return samples


def _take_sample(image_array: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    total = np.zeros(ConvertConstants.N_RGB, dtype=int)
    n_pixels = 0
    for r in range(-radius, radius + 1):
        for c in range(-radius, radius + 1):
            n_pixels += 1
            total += image_array[y + r, x + c]
    return (total // n_pixels).astype(np.uint8)


def get_mask(image_array: np.ndarray, centers: np.ndarray, box_radius: int) -> np.ndarray:
    mask = np.ones((ConvertConstants.N_ROWS, ConvertConstants.N_COLS), dtype=bool)
    roughness_threshold = 50.0
    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            x, y = centers[row, col]
            roughness = _sample_roughness(image_array, x, y, box_radius)
            if roughness > roughness_threshold:
                mask[row, col] = False

    return mask


def _sample_roughness(image_array: np.ndarray, x: int, y: int, radius: int) -> float:
    total_diff = 0
    n_diffs = 0
    for r in range(-radius, radius):
        row_diff = image_array[y + r + 1, x - radius:x + radius + 1] - image_array[y + r, x - radius:x + radius + 1]
        total_diff += np.absolute(row_diff).sum()
    for c in range(-radius, radius):
        col_diff = image_array[y - radius:y + radius + 1, x + c + 1] - image_array[y - radius:y + radius + 1, x + c]
        total_diff += np.absolute(col_diff).sum()

    row_size = radius * 2 + 1
    n_diffs = (row_size - 1) * row_size * 2

    return total_diff / n_diffs / 3


def get_labels(
    samples: np.ndarray,
    n_colors: int,
    mask: np.ndarray,
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required to convert images to graphs")

    samples = samples.reshape(ConvertConstants.N_ROWS * ConvertConstants.N_COLS, ConvertConstants.N_RGB)
    mask = mask.reshape(ConvertConstants.N_ROWS * ConvertConstants.N_COLS)

    labels = np.zeros_like(mask, dtype=int)
    arg_unmask = np.argwhere(~mask).reshape(-1)
    samples = samples[~mask]

    kmeans = KMeans(n_clusters=n_colors, n_init=n_init, max_iter=max_iter)
    kmeans.fit(samples)

    labels[arg_unmask] = kmeans.labels_
    labels[mask] = -1

    return labels.reshape(ConvertConstants.N_ROWS, ConvertConstants.N_COLS)


def color_box(image_array: np.ndarray, x: int, y: int, radius: int, color: Color) -> None:
    for r in range(-radius, radius):
        for c in range(-radius, radius):
            image_array[y + r, x + c] = Color.to_rgb(color)


def normalize_samples(
    samples: np.ndarray,
    centers: np.ndarray,
    normalizer_image_array: np.ndarray,
    box_radius: int,
) -> np.ndarray:
    scaler = _get_brightness_scaler(normalizer_image_array, centers, box_radius)
    samples = np.clip(samples + scaler, 0, 255).astype(np.uint8)
    return samples


def _get_brightness_scaler(
    normalizer_image_array: np.ndarray,
    centers: np.ndarray,
    box_radius: int,
) -> np.ndarray:
    samples = get_samples(normalizer_image_array, centers, box_radius)
    rgb_weight = np.array([0.6, 0.3, 0.1])
    unit_weight = np.array([1, 1, 1])

    side_difference = 0

    left = samples[:, :samples.shape[1] // 2, :]
    left_mean = left.mean(axis=(0, 1))
    left_brightness = (left_mean * rgb_weight).sum()

    right = samples[:, samples.shape[1] // 2:, :]
    right_mean = right.mean(axis=(0, 1))
    right_brightness = (right_mean * rgb_weight).sum() - side_difference

    scaler = np.zeros((ConvertConstants.N_ROWS, ConvertConstants.N_COLS, ConvertConstants.N_RGB))
    for row in range(ConvertConstants.N_ROWS):
        for col in range(ConvertConstants.N_COLS):
            sample = samples[row, col]
            sample_brightness = (sample * rgb_weight).sum()
            side_brightness = left_brightness if col < ConvertConstants.N_COLS // 2 else right_brightness
            scaler[row, col] = (side_brightness - sample_brightness) * unit_weight

    return scaler
