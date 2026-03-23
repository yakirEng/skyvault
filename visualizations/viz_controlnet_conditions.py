"""
Stage 05 — ControlNet Conditioning Verification

Runs locally (no GPU). Shows the three inputs fed to SDXL for N seeds:
  col 1: internal layout  (roads + class colors)
  col 2: seg condition    (ADE20K RGB → ControlNet-Seg)
  col 3: edge condition   (Canny edges → ControlNet-Canny)

Usage:
    cd <repo-root>
    python visualizations/viz_controlnet_conditions.py
    python visualizations/viz_controlnet_conditions.py --seeds 0 7 42 99
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import load_config
from pipeline.layout.generator import generate_layout
from pipeline.generation.controlnet_pipeline import build_conditions

# Internal layout color map (canvas values including sentinel 255)
_LAYOUT_COLORS = {
    0:   [30,  30,  30],   # background  — dark gray
    1:   [255, 255, 255],  # crosswalk   — white
    2:   [255, 140,   0],  # tennis      — orange
    3:   [0,  191, 255],   # pool        — sky blue
    4:   [34,  139,  34],  # eucalyptus  — green
    255: [100, 100, 100],  # road        — mid gray
}
_LEGEND = {
    "Background": [30,  30,  30],
    "Crosswalk":  [255, 255, 255],
    "Tennis":     [255, 140,   0],
    "Pool":       [0,  191, 255],
    "Eucalyptus": [34,  139,  34],
    "Road":       [100, 100, 100],
}


def _colorize_layout(canvas: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*canvas.shape, 3), dtype=np.uint8)
    for val, color in _LAYOUT_COLORS.items():
        rgb[canvas == val] = color
    return rgb


def _legend_patches():
    return [
        mpatches.Patch(color=np.array(c) / 255, label=name)
        for name, c in _LEGEND.items()
    ]


def visualize(seeds: list[int], cfg) -> None:
    n = len(seeds)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing

    col_titles = [
        "Internal Layout\n(roads + objects)",
        "Seg Condition\n(ADE20K RGB → ControlNet-Seg)",
        "Edge Condition\n(Canny → ControlNet-Canny)",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")

    for i, seed in enumerate(seeds):
        layout = generate_layout(seed=seed, cfg=cfg)
        seg, edge = build_conditions(layout)

        axes[i, 0].imshow(_colorize_layout(layout.canvas))
        axes[i, 1].imshow(seg)
        axes[i, 2].imshow(edge, cmap="gray")

        axes[i, 0].set_ylabel(
            f"seed={seed}\nclasses={layout.classes_present}",
            rotation=0, labelpad=80, fontsize=9, va="center",
        )
        for ax in axes[i]:
            ax.axis("off")

    fig.legend(
        handles=_legend_patches(),
        loc="lower center",
        ncol=len(_LEGEND),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    gsd = cfg.resolution.gsd_cm
    px  = cfg.resolution.canvas_px
    fig.suptitle(
        f"Stage 05 — ControlNet Conditioning Inputs\n"
        f"Canvas: {px}×{px}px  |  GSD: {gsd}cm/px  |  "
        f"Area: ~{px * gsd / 100:.1f}m²",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = Path(__file__).parent / "controlnet_conditions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3],
        help="Seeds to visualize (default: 0 1 2 3)",
    )
    args = parser.parse_args()

    cfg = load_config("pipeline_config.yaml")
    visualize(args.seeds, cfg)


if __name__ == "__main__":
    main()
