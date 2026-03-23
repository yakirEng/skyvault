# Component 02 — Layout Engine

## Purpose

Procedurally generate ground-truth segmentation masks for the full pipeline. Uses a **mask-first** approach: the segmentation mask is created deterministically from a seed before any image is generated. Each canvas is a 1024×1024 uint8 array where pixel values = class IDs.

This component is **already implemented**. Design is documented here for interface reference.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/layout/canvas.py` | Canvas initialization and road placement |
| `pipeline/layout/placers.py` | One placer function per object class |
| `pipeline/layout/generator.py` | `generate_layout()` entry point; enforces placement order |

---

## Dependencies

- `01_config` — `PipelineConfig`

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `int` | RNG seed for full reproducibility |
| `cfg` | `PipelineConfig` | All pixel dimensions |

---

## Output

`LayoutResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `canvas` | `np.ndarray` shape `(1024,1024)` dtype `uint8` | Pixel values: 0=background, 1=crosswalk, 2=tennis, 3=pool, 4=eucalyptus, **255=road (internal only)** |
| `road_mask` | `np.ndarray` shape `(1024,1024)` dtype `bool` | True where roads exist |
| `pixel_counts` | `dict[int, int]` | `{class_id: count}` for classes 0-4 **only** — 255 excluded, road pixels counted as 0 unless overwritten |
| `classes_present` | `list[int]` | Subset of `[0, 1, 2, 3, 4]` |
| `placement_log` | `dict[str, bool]` | `{'tennis_1': True, 'pool_1': False, ...}` |
| `seed` | `int` | Echo of input seed |

**Critical:** `canvas` value `255` is an internal marker. It must never be written to disk as a mask. Always call `export_mask(canvas)` (see `03_dataset_generator`) before saving.

---

## Functions

```python
# canvas.py

def init_canvas(cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        canvas:    (1024,1024) uint8, all zeros
        road_mask: (1024,1024) bool, all False
    """

def place_roads(
    canvas: np.ndarray,
    road_mask: np.ndarray,
    cfg: PipelineConfig,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Places 1-3 horizontal and 1-3 vertical road bands.
    Writes cfg.ROAD_INTERNAL (255) into canvas for road pixels.
    Sets road_mask=True for road pixels.
    Returns updated (canvas, road_mask).
    """

# placers.py

def _free_mask(canvas: np.ndarray) -> np.ndarray:
    """Returns (1024,1024) bool — True where canvas == 0 (background only)."""

def place_tennis_courts(
    canvas: np.ndarray,
    road_mask: np.ndarray,
    cfg: PipelineConfig,
    rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """
    Places 0-2 tennis courts on free space (canvas==0).
    Writes class ID 2 into canvas.
    Returns (canvas, placement_log entries for tennis).
    """

def place_pools(
    canvas: np.ndarray,
    road_mask: np.ndarray,
    cfg: PipelineConfig,
    rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """
    Places 0-3 rectangular pools on free space.
    Writes class ID 3. Never overlaps roads or tennis courts.
    """

def place_eucalyptus(
    canvas: np.ndarray,
    road_mask: np.ndarray,
    cfg: PipelineConfig,
    rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """
    Places 0-5 eucalyptus canopy blobs (Perlin noise approximation) on free space.
    Writes class ID 4. paint_mask = (blob==1) & _free_mask(canvas).
    Never covers any previously placed object.
    """

def place_crosswalks(
    canvas: np.ndarray,
    road_mask: np.ndarray,
    cfg: PipelineConfig,
    rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, bool]]:
    """
    Paints striped crosswalks ON road pixels only.
    Implementation: np.where(road_mask & stripe_pattern, 1, canvas)
    Does not compete for free space — only modifies road pixels.
    Writes class ID 1.
    """

# generator.py

def generate_layout(seed: int, cfg: PipelineConfig) -> LayoutResult:
    """
    Authoritative entry point. Calls placers in strict order:
        1. place_roads       (canvas value 255)
        2. place_tennis_courts
        3. place_pools
        4. place_eucalyptus
        5. place_crosswalks  (overwrites road pixels → class 1)
    Returns LayoutResult.
    Never call individual placers directly for dataset generation.
    """
```

---

## Placement Order (must not change)

```
1. Roads          → canvas = 255, road_mask = True
2. Tennis courts  → largest object, needs maximum free space
3. Pools          → fits around tennis courts
4. Eucalyptus     → fills roadside gaps around existing objects
5. Crosswalks     → painted ON road pixels last; doesn't compete for free space
```

**Why this order matters:** each placer calls `_free_mask()` which returns pixels where `canvas == 0`. Eucalyptus last ensures it never covers any object. Crosswalks last ensures they only touch road pixels without affecting free-space detection for other objects.

---

## Spatial Constraints (verified: 100 samples, 0 violations)

| Constraint | Enforcement mechanism |
|---|---|
| Crosswalk only on road | `np.where(road_mask & stripe_pattern, 1, canvas)` |
| Pool never on road | eroded `_free_mask()` used as placement gate |
| Tennis never on road | `_free_mask()` checked at paint time |
| Eucalyptus never on pool/tennis | `paint_mask = (blob==1) & _free_mask(canvas, road_mask)` |
| Pool never on tennis | first-placed object sets `canvas > 0`, blocks subsequent placers |

---

## Internal Canvas Value Note

Road pixels hold value `255` during layout generation. This is a sentinel — roads are not a labeled class. After all placers run:
- Road pixels WITHOUT crosswalk: `canvas == 255` → background (0) when exported
- Road pixels WITH crosswalk: `canvas == 1` (crosswalk painted over road)

`pixel_counts` in `LayoutResult` must already reflect this: report road pixels as background (0) unless overwritten by class 1-4.
