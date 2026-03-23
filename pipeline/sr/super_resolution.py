import sys
import torchvision

# Monkey patch for basicsr compatibility with modern torchvision
# basicsr expects functional_tensor, which was moved/renamed in newer versions.
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules['torchvision.transforms.functional_tensor'] = functional
    except ImportError:
        pass

import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from pipeline.config import PipelineConfig

# Real-ESRGAN requires these for the x4plus model
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def load_and_crop_tile(
        geotiff_path: str,
        crop_size_px: int = 410,
        rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Open GeoTIFF, read RGB bands, and take a random crop.
    """
    if rng is None:
        rng = np.random.default_rng()

    with rasterio.open(geotiff_path) as src:
        # Check dimensions
        if src.width < crop_size_px or src.height < crop_size_px:
            raise ValueError(f"Tile {geotiff_path} is too small for crop size {crop_size_px}")

        # Pick a random origin
        row_off = rng.integers(0, src.height - crop_size_px)
        col_off = rng.integers(0, src.width - crop_size_px)

        window = Window(col_off, row_off, crop_size_px, crop_size_px)

        # Read bands 1, 2, 3 (R, G, B in standard gov.il orthos)
        # Rasterio returns (C, H, W)
        bands = src.read([1, 2, 3], window=window)

        # Transpose to (H, W, C) for OpenCV/Real-ESRGAN
        tile = np.moveaxis(bands, 0, -1)

    return tile


def upscale_to_5cm(tile: np.ndarray, model_path: str) -> np.ndarray:
    """
    FIX #3: Upscale 410px -> 1640px (x4) -> 1024px (net x2.5).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Weights not found at {model_path}. "
            "Download them from: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        )

    # 1. Setup Real-ESRGAN x4plus model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,  # Tile to avoid OOM on smaller GPUs
        tile_pad=10,
        pre_pad=0,
        half=True  # Use half-precision if on GPU
    )

    # Step 1: Real-ESRGAN x4 -> 1640x1640
    upscaled, _ = upsampler.enhance(tile, outscale=4)

    # Step 2: Resize to exactly 1024x1024
    # INTER_AREA is preferred for downsampling to avoid aliasing
    final = cv2.resize(upscaled, (1024, 1024), interpolation=cv2.INTER_AREA)

    return final


def process_geotiff_tiles(
        geotiff_paths: list[str],
        output_dir: str,
        n_crops_per_tile: int,
        cfg: PipelineConfig,
        model_path: str
) -> list[str]:
    """
    Batch processing loop for all GeoTIFFs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)  # Consistent seeds for reproducibility
    saved_paths = []
    counter = 0

    for gt_path in geotiff_paths:
        print(f"Processing source tile: {gt_path}")
        for j in range(n_crops_per_tile):
            # Load & Crop
            raw_tile = load_and_crop_tile(gt_path, crop_size_px=410, rng=rng)

            # Upscale
            processed_tile = upscale_to_5cm(raw_tile, model_path)

            # Save
            save_name = out_path / f"bg_{counter:04d}.png"
            # OpenCV uses BGR for saving
            cv2.imwrite(str(save_name), cv2.cvtColor(processed_tile, cv2.COLOR_RGB2BGR))

            saved_paths.append(str(save_name.absolute()))
            counter += 1
            print(f"  -> Saved crop {j + 1}/{n_crops_per_tile}: {save_name.name}")

    return saved_paths