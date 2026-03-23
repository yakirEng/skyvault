import numpy as np
from pipeline.config import PipelineConfig


def init_canvas(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes a blank canvas and road mask.
    Returns:
        canvas:    (canvas_px, canvas_px) uint8, all zeros
        road_mask: (canvas_px, canvas_px) bool, all False
    """
    size = cfg.resolution.canvas_px
    canvas = np.zeros((size, size), dtype=np.uint8)
    road_mask = np.zeros((size, size), dtype=bool)
    return canvas, road_mask


def place_roads(
        canvas: np.ndarray,
        road_mask: np.ndarray,
        cfg: PipelineConfig,
        rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Places horizontal and vertical road bands.
    """
    size = cfg.resolution.canvas_px
    padding = int(size * cfg.roads.padding_frac)

    n_horiz = rng.integers(cfg.roads.n_horizontal[0], cfg.roads.n_horizontal[1] + 1)
    n_vert = rng.integers(cfg.roads.n_vertical[0], cfg.roads.n_vertical[1] + 1)

    # Place horizontal roads
    for _ in range(n_horiz):
        y = rng.integers(padding, size - padding - cfg.roads.width_px)
        canvas[y:y + cfg.roads.width_px, :] = cfg.ROAD_INTERNAL
        road_mask[y:y + cfg.roads.width_px, :] = True

    # Place vertical roads
    for _ in range(n_vert):
        x = rng.integers(padding, size - padding - cfg.roads.width_px)
        canvas[:, x:x + cfg.roads.width_px] = cfg.ROAD_INTERNAL
        road_mask[:, x:x + cfg.roads.width_px] = True

    return canvas, road_mask