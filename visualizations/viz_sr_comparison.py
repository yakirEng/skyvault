import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.config import load_config
from pipeline.sr.super_resolution import load_and_crop_tile, upscale_to_5cm


def main():
    print("Initializing SR Fidelity Audit...")
    cfg = load_config("pipeline_config.yaml")

    # Paths - Update these to your local test files
    geotiff_path = "data/raw_tiles/test_tile.tif"
    model_path = "weights/RealESRGAN_x4plus.pth"

    if not os.path.exists(geotiff_path):
        print(f"Error: Could not find test tile at {geotiff_path}. Please provide a valid GeoTIFF.")
        return

    # 1. Load Source Crop (12.5 cm/px, 410x410)
    print("Loading 12.5 cm/px source crop...")
    src_tile = load_and_crop_tile(geotiff_path, crop_size_px=410)

    # 2. Perform Fix #3 Upscale (5 cm/px, 1024x1024)
    print("Running Real-ESRGAN x4 + INTER_AREA Resize...")
    upscaled_tile = upscale_to_5cm(src_tile, model_path)

    # 3. Create Visualization Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top Row: Full Images
    axes[0, 0].imshow(src_tile)
    axes[0, 0].set_title(f"Source: 12.5 cm/px (410x410)\nOriginal Gov.il Ortho", fontsize=12)

    axes[0, 1].imshow(upscaled_tile)
    axes[0, 1].set_title(f"Upscaled: 5.0 cm/px (1024x1024)\nFix #3 (Real-ESRGAN x4 -> Resize)", fontsize=12)

    # Bottom Row: Zoomed Patches for Texture Audit
    zoom_size_src = 100
    zoom_size_up = int(zoom_size_src * (12.5 / 5.0))  # 250px

    # Take a crop from the center
    cy_s, cx_s = 205, 205
    src_patch = src_tile[cy_s - 50:cy_s + 50, cx_s - 50:cx_s + 50]

    cy_u, cx_u = 512, 512
    up_patch = upscaled_tile[cy_u - 125:cy_u + 125, cx_u - 125:cx_u + 125]

    axes[1, 0].imshow(src_patch)
    axes[1, 0].set_title("Source Texture Zoom (100x100)")

    axes[1, 1].imshow(up_patch)
    axes[1, 1].set_title("Upscaled Texture Zoom (250x250)")

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle("Component 04: Super-Resolution Fidelity Audit", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the audit for the thesis documentation
    output_path = "visualizations/sr_fidelity_audit.png"
    plt.savefig(output_path)
    print(f"Audit saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()