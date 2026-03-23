import cv2
import numpy as np
from pipeline.config import PipelineConfig


def _free_mask(canvas: np.ndarray) -> np.ndarray:
    """Returns boolean mask where canvas is purely background."""
    return canvas == 0


def place_tennis_courts(
        canvas: np.ndarray,
        road_mask: np.ndarray,
        cfg: PipelineConfig,
        rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """Places 0-2 tennis courts, allowing them to be partially cut off at the edges."""
    log = {}
    n_courts = rng.integers(0, 3)
    size = cfg.resolution.canvas_px

    for i in range(n_courts):
        name = f"tennis_{i + 1}"
        rot = rng.choice(cfg.objects.tennis_rotations)
        w, h = cfg.objects.tennis_court_w_px, cfg.objects.tennis_court_h_px
        if rot == 90:
            w, h = h, w

        placed = False
        # Back to 10 attempts
        for _ in range(10):
            # Allow the top-left corner to be outside the canvas
            # bounds ensure at least half the court remains visible
            x = rng.integers(-w // 2, size - w // 2)
            y = rng.integers(-h // 2, size - h // 2)

            # Clamp the object coordinates to the canvas boundaries
            y1, y2 = max(0, y), min(size, y + h)
            x1, x2 = max(0, x), min(size, x + w)

            # Safety check (though rng bounds make this highly unlikely)
            if y2 <= y1 or x2 <= x1:
                continue

            # Check if the visible slice is completely free of roads/other objects
            if np.all(_free_mask(canvas)[y1:y2, x1:x2]):
                canvas[y1:y2, x1:x2] = cfg.class_ids["tennis_court"]
                placed = True
                break

        log[name] = placed

    return canvas, log


def place_pools(
        canvas: np.ndarray,
        road_mask: np.ndarray,
        cfg: PipelineConfig,
        rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """Places 0-3 rectangular pools on free space."""
    log = {}
    n_pools = rng.integers(0, 4)
    size = cfg.resolution.canvas_px

    for i in range(n_pools):
        name = f"pool_{i + 1}"
        w = rng.integers(cfg.objects.pool_min_px, cfg.objects.pool_max_px)
        h = rng.integers(cfg.objects.pool_min_px, cfg.objects.pool_max_px)

        placed = False
        for _ in range(10):
            x = rng.integers(0, size - w)
            y = rng.integers(0, size - h)

            if np.all(_free_mask(canvas)[y:y + h, x:x + w]):
                canvas[y:y + h, x:x + w] = cfg.class_ids["pool"]
                placed = True
                break
        log[name] = placed

    return canvas, log


def place_eucalyptus(
        canvas: np.ndarray,
        road_mask: np.ndarray,
        cfg: PipelineConfig,
        rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """Places 0-5 eucalyptus canopy blobs overlapping anything beneath them."""
    log = {}
    n_trees = rng.integers(0, 6)
    size = cfg.resolution.canvas_px

    for i in range(n_trees):
        name = f"eucalyptus_{i + 1}"
        radius = rng.integers(cfg.objects.eucalyptus_min_px, cfg.objects.eucalyptus_max_px) // 2
        x = rng.integers(radius, size - radius)
        y = rng.integers(radius, size - radius)

        blob_mask = np.zeros_like(canvas, dtype=bool)
        cv2.circle(blob_mask.view(np.uint8), (x, y), radius, 1, -1)
        blob_mask = blob_mask.astype(bool)

        # FIX: Trees in BEV occlude EVERYTHING on the ground.
        # We completely remove the `valid_ground` check.
        if np.any(blob_mask):
            canvas[blob_mask] = cfg.class_ids["eucalyptus"]
            log[name] = True
        else:
            log[name] = False

    return canvas, log


def place_crosswalks(
        canvas: np.ndarray,
        road_mask: np.ndarray,
        cfg: PipelineConfig,
        rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """Paints striped crosswalks strictly on straight road segments using a tight local bounding box."""
    log = {"crosswalk_1": False}
    size = cfg.resolution.canvas_px

    # 1. Filter out intersections to find pure straight roads
    row_counts = np.sum(road_mask, axis=1)
    col_counts = np.sum(road_mask, axis=0)

    row_grid = np.tile(row_counts[:, None], (1, size))
    col_grid = np.tile(col_counts[None, :], (size, 1))

    pure_road_mask = road_mask & (
            ((row_grid > size * 0.5) & (col_grid < size * 0.5)) |
            ((col_grid > size * 0.5) & (row_grid < size * 0.5))
    )

    road_y, road_x = np.where(pure_road_mask)
    if len(road_y) == 0:
        return canvas, log

    # 2. Pick a safe center point
    idx = rng.integers(0, len(road_y))
    cy, cx = road_y[idx], road_x[idx]

    is_horizontal_road = row_counts[cy] > col_counts[cx]

    # 3. Generate correct stripe rotation
    stripe_pattern = np.zeros((size, size), dtype=bool)
    period = cfg.objects.crosswalk_stripe_px + cfg.objects.crosswalk_gap_px

    if is_horizontal_road:
        for i in range(0, size, period):
            stripe_pattern[i:i + cfg.objects.crosswalk_stripe_px, :] = True
    else:
        for i in range(0, size, period):
            stripe_pattern[:, i:i + cfg.objects.crosswalk_stripe_px] = True

    # 4. STRICT LOCAL BOUNDING BOX
    # A square ensures it covers the crossing but cannot span the whole canvas
    box_size = max(cfg.objects.crosswalk_len_px, cfg.roads.width_px) + 20
    half_box = box_size // 2

    y1, y2 = max(0, cy - half_box), min(size, cy + half_box)
    x1, x2 = max(0, cx - half_box), min(size, cx + half_box)

    crosswalk_region = np.zeros_like(canvas, dtype=bool)
    crosswalk_region[y1:y2, x1:x2] = True

    # 5. Paint
    paint_mask = road_mask & stripe_pattern & crosswalk_region & (canvas == cfg.ROAD_INTERNAL)

    if np.any(paint_mask):
        canvas[paint_mask] = cfg.class_ids["crosswalk"]
        log["crosswalk_1"] = True

    return canvas, log