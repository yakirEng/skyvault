# Component 03 — Dataset Generator + Palette Mapper

## Purpose

Two responsibilities handled together:
1. **Dataset Generator** — loops `generate_layout()` over N seeds, exports normalized masks (values 0-4 only) and JSON metadata per sample.
2. **Palette Mapper** — converts grayscale masks to ADE20K RGB conditioning images and Canny edge maps for ControlNet input.

**Fix #1 is applied here:** road pixels (`canvas == 255`) are normalized to 0 (background) before any mask is saved to disk.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/layout/generator.py` | Extend with `generate_dataset()` and `export_mask()` |
| `pipeline/layout/palette.py` | New — `mask_to_ade20k_rgb()`, `mask_to_canny_edges()` |

---

## Dependencies

- `01_config` — `PipelineConfig`
- `02_layout_engine` — `generate_layout()`, `LayoutResult`

---

## Inputs

### `generate_dataset()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Number of layout samples to generate |
| `cfg` | `PipelineConfig` | Pipeline configuration |
| `output_dir` | `str` | Root output directory |
| `start_seed` | `int` | First seed (default 0); sample i uses seed `start_seed + i` |

### `export_mask()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `canvas` | `np.ndarray` shape `(H,W)` dtype `uint8` | Raw canvas from `LayoutResult.canvas` (may contain 255) |

### `mask_to_ade20k_rgb()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `np.ndarray` shape `(H,W)` dtype `uint8` | Normalized mask, values 0-4 ONLY |

### `mask_to_canny_edges()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `np.ndarray` shape `(H,W)` dtype `uint8` | Normalized mask, values 0-4 ONLY |

---

## Outputs

### `generate_dataset()` — files saved to disk

```
{output_dir}/
├── masks/
│   └── mask_{i:04d}.png     # grayscale PNG, values 0-4 only (uint8)
└── metadata_{i:04d}.json    # SampleMetadata fields as JSON
```

Returns: `list[SampleMetadata]` — one entry per sample; `split` field is left empty string at this stage (assigned in `09_export`).

### `export_mask()` — FIX #1

```
Input:  canvas (H,W) uint8 — values 0-4 and 255
Output: mask   (H,W) uint8 — values 0-4 ONLY
```

Implementation: `np.where(canvas == 255, 0, canvas).astype(np.uint8)`

### `mask_to_ade20k_rgb()`

```
Input:  mask (H,W) uint8, values 0-4
Output: rgb  (H,W,3) uint8 — ADE20K palette colors
```

### `mask_to_canny_edges()`

```
Input:  mask  (H,W) uint8, values 0-4
Output: edges (H,W) uint8 — Canny edge map
```

---

## Functions

```python
# layout/generator.py

def export_mask(canvas: np.ndarray) -> np.ndarray:
    """
    Normalize raw canvas for disk export.

    FIX #1: Road pixels (value 255) become background (0).
    Crosswalk pixels (value 1) that were already painted over road pixels
    are unaffected — they are < 255 and pass through unchanged.

    Returns: (H,W) uint8, values guaranteed in range [0, 4].
    """
    return np.where(canvas == 255, 0, canvas).astype(np.uint8)


def generate_dataset(
    n: int,
    cfg: PipelineConfig,
    output_dir: str,
    start_seed: int = 0
) -> list[SampleMetadata]:
    """
    Generate n layout samples.

    For each i in range(n):
        result = generate_layout(seed=start_seed + i, cfg=cfg)
        mask   = export_mask(result.canvas)
        Save mask as PNG to {output_dir}/masks/mask_{i:04d}.png
        Save SampleMetadata as JSON to {output_dir}/metadata_{i:04d}.json

    The SampleMetadata fields background_tile_path and image_path are left
    as empty strings — they are filled in by later stages.

    Returns: list[SampleMetadata]
    """


# layout/palette.py

ADE20K_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (120, 120, 120),   # background → gray
    1: (140, 140, 215),   # crosswalk  → road/pavement blue-gray
    2: (180, 120, 120),   # tennis     → sports court reddish
    3: ( 61, 230, 250),   # pool       → water cyan
    4: (  4, 200,   3),   # eucalyptus → tree green
}


def mask_to_ade20k_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert grayscale segmentation mask to ADE20K RGB palette image.

    Input:  mask (H,W) uint8, values 0-4 — must be normalized (no 255)
    Output: rgb  (H,W,3) uint8

    Used as seg_condition for ControlNet-Seg.
    Raises ValueError if mask contains values outside 0-4.
    """


def mask_to_canny_edges(mask: np.ndarray) -> np.ndarray:
    """
    Generate Canny edge map from segmentation mask class boundaries.

    Input:  mask  (H,W) uint8, values 0-4
    Output: edges (H,W) uint8 — binary edge map (0 or 255)

    Implementation:
        scaled = (mask * 50).astype(np.uint8)   # spread class IDs into visible range
        edges  = cv2.Canny(scaled, threshold1=50, threshold2=150)

    Used as edge_condition for ControlNet-Canny.
    """
```

---

## `SampleMetadata` JSON Schema

```json
{
  "sample_id": 0,
  "seed": 0,
  "gsd_cm": 5.0,
  "classes_present": [0, 1, 2, 3],
  "pixel_counts": {"0": 750000, "1": 3000, "2": 88000, "3": 15000},
  "placement_log": {"tennis_1": true, "pool_1": true, "pool_2": false},
  "split": "",
  "source": "synthetic",
  "background_tile_path": "",
  "mask_path": "output/layouts/masks/mask_0000.png",
  "image_path": ""
}
```

---

## Implementation Notes

- `export_mask()` must be called before any mask is passed to `mask_to_ade20k_rgb()` or `mask_to_canny_edges()`
- `mask_to_ade20k_rgb()` raises `ValueError` if `mask.max() > 4` — acts as a safety check that `export_mask()` was applied
- Masks are saved as single-channel grayscale PNG (not RGB) — values 0-4 are directly the class IDs
- Do not apply any color palette when saving the mask PNG — the raw class ID values must be preserved for training
