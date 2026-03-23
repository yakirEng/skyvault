# Component 04 — Super-Resolution (Background Tiles)

## Purpose

Download gov.il orthophoto GeoTIFF tiles (12.5 cm/px, EPSG:2039), crop random 1024×1024-meter patches, and upscale to 5 cm/px using Real-ESRGAN. Produces unlabeled real background tiles used by the compositor.

**Fix #3 applied here:** Real-ESRGAN has no native ×2.5 mode. The correct approach is ×4 upscaling followed by `cv2.resize` to achieve net ×2.5.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/sr/super_resolution.py` | All SR logic — tile loading, upscaling, batch processing |

---

## Dependencies

- `01_config` — `PipelineConfig` (for `canvas_px`)
- External: `rasterio`, `Real-ESRGAN` (`realesrgan` pip package), `opencv-python`

---

## Inputs

### `load_and_crop_tile()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `geotiff_path` | `str` | Path to a `.tif` file, expected projection EPSG:2039, 12.5 cm/px |
| `crop_size_px` | `int` | Crop size in source pixels (default `410`) |
| `rng` | `np.random.Generator \| None` | RNG for random crop origin; if None uses a fixed default |

**Why `crop_size_px = 410`:** At 12.5 cm/px, 410 pixels = 51.25 m ≈ same ground coverage as 1024 px at 5 cm/px.

### `upscale_to_5cm()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile` | `np.ndarray` shape `(410,410,3)` dtype `uint8` | RGB crop in source resolution |
| `model_path` | `str` | Path to `RealESRGAN_x4plus.pth` weights file |

### `process_geotiff_tiles()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `geotiff_paths` | `list[str]` | List of GeoTIFF file paths |
| `output_dir` | `str` | Where to save output PNG files |
| `n_crops_per_tile` | `int` | Number of random crops per GeoTIFF |
| `cfg` | `PipelineConfig` | Used for `canvas_px` (output resolution) |
| `model_path` | `str` | Path to Real-ESRGAN weights |

---

## Outputs

### `load_and_crop_tile()`

```
Output: np.ndarray shape (410, 410, 3) dtype uint8 — RGB, channel order BGR→RGB converted
```

### `upscale_to_5cm()` — FIX #3

```
Output: np.ndarray shape (1024, 1024, 3) dtype uint8
```

**Why two steps:**

| Step | Operation | Size |
|------|-----------|------|
| Input | Source crop | 410 × 410 |
| Step 1 | Real-ESRGAN ×4 | 1640 × 1640 |
| Step 2 | `cv2.resize(..., (1024, 1024), interpolation=cv2.INTER_AREA)` | 1024 × 1024 |
| Net scale | 1024/410 ≈ 2.497× ≈ **×2.5** | ✓ |

Real-ESRGAN provides ×2 and ×4 modes only. Net ×2.5 is achieved by ×4 then INTER_AREA downscale. INTER_AREA is correct here (downsampling) and avoids aliasing artifacts.

### `process_geotiff_tiles()`

```
Saves: {output_dir}/bg_{i:04d}.png  — (1024,1024,3) uint8 RGB PNG
Returns: list[str] — absolute paths of all saved PNG files
```

---

## Functions

```python
def load_and_crop_tile(
    geotiff_path: str,
    crop_size_px: int = 410,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Open GeoTIFF with rasterio, read RGB bands, take a random crop.

    Steps:
    1. Open file with rasterio.open()
    2. If CRS is not EPSG:2039, reproject on-the-fly using rasterio.warp
    3. Sample a random (row_off, col_off) such that crop fits within tile bounds
    4. Read crop_size_px × crop_size_px window
    5. Stack bands as (H,W,3) uint8 RGB (rasterio returns (C,H,W) — transpose)
    6. Return array

    Raises:
        FileNotFoundError: if geotiff_path does not exist
        ValueError: if tile is smaller than crop_size_px in either dimension
    """


def upscale_to_5cm(
    tile: np.ndarray,
    model_path: str
) -> np.ndarray:
    """
    Upscale a 410×410 tile from 12.5 cm/px to 5 cm/px.

    FIX #3: Real-ESRGAN has no native ×2.5 mode.
    Step 1: apply Real-ESRGAN ×4  → output (1640,1640,3) uint8
    Step 2: cv2.resize to (1024,1024) with INTER_AREA  → net ×2.5

    Args:
        tile:       (410,410,3) uint8 RGB
        model_path: path to RealESRGAN_x4plus.pth

    Returns: (1024,1024,3) uint8 RGB
    """


def process_geotiff_tiles(
    geotiff_paths: list[str],
    output_dir: str,
    n_crops_per_tile: int,
    cfg: PipelineConfig,
    model_path: str
) -> list[str]:
    """
    Batch-process a list of GeoTIFF files into background PNG tiles.

    For each (geotiff, crop_index):
        tile       = load_and_crop_tile(geotiff_path, rng=rng)
        upscaled   = upscale_to_5cm(tile, model_path)
        Save as PNG to {output_dir}/bg_{counter:04d}.png

    Returns: list of absolute paths to saved PNG files.
    Prints progress to stdout.
    """
```

---

## Installation

```bash
pip install realesrgan rasterio
# Download weights:
# wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

---

## Implementation Notes

- Background tiles carry **no labels** — they are just RGB images for the compositor
- Tiles should cover diverse geography: use `rng` with varied seeds to avoid repetitive crops from the same tile
- For geographic diversity: target at least 20 distinct GeoTIFF tiles from different regions (north, center, south)
- gov.il orthophotos use EPSG:2039 (Israeli Transverse Mercator); rasterio handles reprojection
- Output PNG is lossless (no JPEG compression) to preserve texture fidelity for compositing
