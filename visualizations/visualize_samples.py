import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.config import load_config
from pipeline.layout.generator import generate_layout, export_mask
from pipeline.layout.palette import mask_to_ade20k_rgb, mask_to_canny_edges

# Color map for the internal layout visualization
COLOR_MAP = {
    0: [30, 30, 30],  # Background: Dark Gray
    1: [255, 255, 255],  # Crosswalk: White
    2: [255, 140, 0],  # Tennis Court: Orange
    3: [0, 191, 255],  # Pool: Deep Sky Blue
    4: [34, 139, 34],  # Eucalyptus: Forest Green
    255: [100, 100, 100]  # Road (Internal Sentinel): Mid Gray
}


def apply_color_map(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        rgb_image[canvas == class_id] = color
    return rgb_image


def main():
    print("Loading pipeline configuration...")
    cfg = load_config("pipeline_config.yaml")

    # Metadata for the title
    gsd_cm = cfg.resolution.gsd_cm
    canvas_px = cfg.resolution.canvas_px
    physical_m = canvas_px * (gsd_cm / 100.0)

    # Generate 2 samples for the comparison grid
    base_seed = int(time.time())
    seeds = [base_seed, base_seed + 1]

    fig, axes = plt.subplots(len(seeds), 3, figsize=(15, 10))

    fig.suptitle(
        f"ControlNet Conditioning Verification\n"
        f"Size: {canvas_px}px | Resolution: {gsd_cm}cm/px | Area: ~{physical_m:.1f}m",
        fontsize=16, fontweight='bold'
    )

    for i, seed in enumerate(seeds):
        # 1. Generate Raw Layout
        result = generate_layout(seed=seed, cfg=cfg)
        vis_layout = apply_color_map(result.canvas)

        # 2. Export & Map Palette (Fix #1: 255 -> 0)
        normalized = export_mask(result.canvas)
        ade_vis = mask_to_ade20k_rgb(normalized)
        canny_vis = mask_to_canny_edges(normalized)

        # Plotting
        ax_row = axes[i]

        # Row labels
        ax_row[0].set_ylabel(f"Seed {seed}", rotation=0, labelpad=40, fontsize=12, fontweight='bold')

        # Internal Layout
        ax_row[0].imshow(vis_layout)
        ax_row[0].set_title("Internal Layout (with Sentinels)")

        # ADE20K
        ax_row[1].imshow(ade_vis)
        ax_row[1].set_title("ADE20K (ControlNet-Seg)")

        # Canny
        ax_row[2].imshow(canny_vis, cmap='gray')
        ax_row[2].set_title("Canny Edges (ControlNet-Canny)")

        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    print("Verification grid generated. Displaying...")
    plt.show()


if __name__ == "__main__":
    main()