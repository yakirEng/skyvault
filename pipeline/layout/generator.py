import os
import json
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.layout.canvas import init_canvas, place_roads
from pipeline.layout.placers import (
    place_tennis_courts, place_pools, place_eucalyptus, place_crosswalks
)


@dataclass
class LayoutResult:
    canvas: np.ndarray
    road_mask: np.ndarray
    pixel_counts: dict[int, int]
    classes_present: list[int]
    placement_log: dict[str, bool]
    seed: int


@dataclass
class SampleMetadata:
    sample_id: int
    seed: int
    gsd_cm: float
    classes_present: list[int]
    pixel_counts: dict[int, int]
    placement_log: dict[str, bool]
    split: str
    source: str
    background_tile_path: str
    mask_path: str
    image_path: str


def generate_layout(seed: int, cfg: PipelineConfig) -> LayoutResult:
    """Authoritative entry point for a single layout."""
    rng = np.random.default_rng(seed)
    canvas, road_mask = init_canvas(cfg)
    placement_log = {}

    # Placement Order: Roads -> Tennis -> Pools -> Eucalyptus -> Crosswalks
    canvas, road_mask = place_roads(canvas, road_mask, cfg, rng)
    canvas, t_log = place_tennis_courts(canvas, road_mask, cfg, rng)
    canvas, p_log = place_pools(canvas, road_mask, cfg, rng)
    canvas, e_log = place_eucalyptus(canvas, road_mask, cfg, rng)
    canvas, c_log = place_crosswalks(canvas, road_mask, cfg, rng)

    placement_log.update(t_log);
    placement_log.update(p_log)
    placement_log.update(e_log);
    placement_log.update(c_log)

    # Calculate final counts, merging internal road (255) into background (0)
    pixel_counts = {i: int(np.sum(canvas == i)) for i in range(1, 5)}
    pixel_counts[0] = int(cfg.resolution.canvas_px ** 2 - sum(pixel_counts.values()))
    classes_present = [cid for cid, count in pixel_counts.items() if count > 0]

    return LayoutResult(canvas, road_mask, pixel_counts, classes_present, placement_log, seed)


def export_mask(canvas: np.ndarray) -> np.ndarray:
    """FIX #1: Normalize road pixels (255) to background (0)."""
    return np.where(canvas == 255, 0, canvas).astype(np.uint8)


def generate_dataset(
        n: int,
        cfg: PipelineConfig,
        output_dir: str,
        start_seed: int = 0
) -> list[SampleMetadata]:
    """Loop generate_layout() and save normalized results to disk."""
    base_path = Path(output_dir)
    mask_dir = base_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    for i in range(n):
        seed = start_seed + i
        result = generate_layout(seed=seed, cfg=cfg)

        # Normalize for disk
        mask = export_mask(result.canvas)

        mask_filename = f"mask_{i:04d}.png"
        meta_filename = f"metadata_{i:04d}.json"
        mask_path = mask_dir / mask_filename

        # Save as single-channel grayscale (values 0-4)
        cv2.imwrite(str(mask_path), mask)

        meta = SampleMetadata(
            sample_id=i,
            seed=seed,
            gsd_cm=cfg.resolution.gsd_cm,
            classes_present=result.classes_present,
            pixel_counts=result.pixel_counts,
            placement_log=result.placement_log,
            split="",  # Assigned in stage 09
            source="synthetic",
            background_tile_path="",
            mask_path=str(mask_path),
            image_path=""
        )

        with open(base_path / meta_filename, "w") as f:
            json.dump(asdict(meta), f, indent=2)  #

        metadata_list.append(meta)

    print(f"Dataset generation complete. Saved {n} samples to {output_dir}")
    return metadata_list