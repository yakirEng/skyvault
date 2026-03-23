# System Overview — Synthetic Aerial Segmentation Dataset Pipeline

## Purpose

Hybrid pipeline that generates a labeled synthetic dataset for training a semantic segmentation model on high-resolution aerial imagery (5 cm/px) over Israel. Real gov.il orthophoto tiles provide geographically accurate backgrounds; procedural layout + ControlNet SDXL provide synthetic labeled objects with perfect ground-truth masks.

**4 target classes:** crosswalks (1), tennis courts (2), private swimming pools (3), eucalyptus canopy (4). Background = 0.
**Canvas:** 1024×1024 px at 5 cm/px = ~51×51 meters per sample.

---

## Component Map

| # | Component | Design File | Python File(s) | Status |
|---|-----------|-------------|----------------|--------|
| 01 | Config | `01_config.md` | `pipeline/config.py`, `pipeline/pipeline_config.yaml` | TODO |
| 02 | Layout Engine | `02_layout_engine.md` | `pipeline/layout/canvas.py`, `placers.py`, `generator.py` | Implemented |
| 03 | Dataset Generator + Palette | `03_dataset_generator.md` | `pipeline/layout/generator.py` (extend), `pipeline/layout/palette.py` | TODO |
| 04 | Super-Resolution | `04_super_resolution.md` | `pipeline/sr/super_resolution.py` | TODO |
| 05 | ControlNet Generation | `05_controlnet_generation.md` | `pipeline/generation/controlnet_pipeline.py` | TODO |
| 06 | Compositor | `06_compositor.md` | `pipeline/compositor/compositor.py` | TODO |
| 07 | Quality Gate | `07_quality_gate.md` | `pipeline/quality/filter.py` | TODO |
| 08 | Augmentation | `08_augmentation.md` | `pipeline/augmentation/augmentor.py` | TODO |
| 09 | Export & Split | `09_export.md` | `pipeline/export/splitter.py` | TODO |

---

## Data Flow

```
pipeline_config.yaml
    │
    ▼ load_config()                                          [01_config]
PipelineConfig
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                       ▼
GeoTIFF tiles (12.5 cm/px, EPSG:2039)           generate_layout(seed, cfg)   [02_layout_engine]
    │                                                       │
    ▼ process_geotiff_tiles()           [04_super_resolution] │ LayoutResult
list[PNG] backgrounds/ (1024×1024×3 uint8, 5cm/px)          │
                                                             ▼
                                               export_mask(canvas)            [03_dataset_generator]
                                                             │ FIX #1: 255→0
                                                             ▼
                                               normalized_mask (1024×1024) uint8, values 0-4
                                               masks/mask_{i:04d}.png
                                               metadata_{i:04d}.json
                                                             │
                                                             ▼
                                               build_conditions()             [05_controlnet_generation]
                                                             │
                                                             ▼
                                               seg_condition (1024×1024×3) + edge_condition (1024×1024)
                                                             │
                                                             ▼
                                               generate_sdxl_scene()          [05_controlnet_generation]
                                                             │
                                                             ▼
                                               scene (1024×1024×3) uint8
                                                             │
                                                             ▼
                                               extract_object_patches()        [05_controlnet_generation]
                                                             │ FIX #2: explicit extraction
                                                             ▼
                                               list[ObjectPatch]
                                                             │
    ┌────────────────────────────────────────────────────────┘
    │ background tile + patches + normalized_mask
    ▼
composite()                                                  [06_compositor]
    │ FIX #5: no scale jitter, same pixel coords as mask
    ▼
CompositeResult (image 1024×1024×3, mask 1024×1024 values 0-4)
    │
    ▼
filter_sample()                                              [07_quality_gate]
    │ FIX #6: Laplacian blur + class coverage (no CLIP)
    ▼ passed samples only
    │
    ▼
augment_sample() ×4                                          [08_augmentation]
    │
    ▼
~4000 (image, mask) pairs
    │
    ▼
stratified_split() + export_split()                          [09_export]
    │ FIX #4: test set = manually annotated real tiles
    ▼
dataset/train/ val/ test/
```

---

## Shared Data Structures

All types below are used across multiple components. Defined in their primary module but imported freely elsewhere.

```python
# pipeline/config.py
@dataclass
class ResolutionConfig:
    gsd_cm: float       # ground sample distance in cm (5.0)
    canvas_px: int      # 1024

@dataclass
class ObjectConfig:
    tennis_court_w_px: int      # 475  (23.77m / 0.05m)
    tennis_court_h_px: int      # 219  (10.97m / 0.05m)
    tennis_rotations: list[int] # [0, 90]
    crosswalk_stripe_px: int    # 10   (0.50m / 0.05m)
    crosswalk_gap_px: int       # 10
    crosswalk_len_px: int       # 60   (3.00m / 0.05m)
    pool_min_px: int            # 80   (4.0m / 0.05m)
    pool_max_px: int            # 240  (12.0m / 0.05m)
    eucalyptus_min_px: int      # 100  (5.0m / 0.05m)
    eucalyptus_max_px: int      # 300  (15.0m / 0.05m)

@dataclass
class RoadConfig:
    width_px: int               # 70   (3.5m / 0.05m)
    n_horizontal: tuple[int,int]# (1, 3) — random range
    n_vertical: tuple[int,int]  # (1, 3)
    padding_frac: float         # 0.125

@dataclass
class PipelineConfig:
    resolution: ResolutionConfig
    objects: ObjectConfig
    roads: RoadConfig
    class_ids: dict[str, int]   # {background:0, crosswalk:1, tennis_court:2, pool:3, eucalyptus:4}
    ROAD_INTERNAL: int = 255    # canvas value for roads — NEVER exported in masks

# pipeline/layout/generator.py
@dataclass
class LayoutResult:
    canvas: np.ndarray          # (1024,1024) uint8 — 0=bg, 1-4=classes, 255=road (internal)
    road_mask: np.ndarray       # (1024,1024) bool
    pixel_counts: dict[int,int] # {class_id: count} for 0-4 only (255 excluded)
    classes_present: list[int]  # subset of [0,1,2,3,4]
    placement_log: dict[str,bool]  # {'tennis_1': True, 'pool_1': False, ...}
    seed: int

# pipeline/generation/controlnet_pipeline.py
@dataclass
class ObjectPatch:
    image: np.ndarray           # (h,w,3) uint8 — RGB crop from SDXL output
    alpha: np.ndarray           # (h,w) bool — pixel-level object mask
    class_id: int               # 1-4
    bbox: tuple[int,int,int,int]# (x1, y1, x2, y2) in layout pixel coordinates

@dataclass
class SDXLResult:
    scene: np.ndarray           # (1024,1024,3) uint8 — full SDXL output
    patches: list[ObjectPatch]  # one per connected object instance
    seg_condition: np.ndarray   # (1024,1024,3) uint8 — ADE20K RGB (input to SDXL)
    edge_condition: np.ndarray  # (1024,1024) uint8 — Canny edges (input to SDXL)

# pipeline/compositor/compositor.py
@dataclass
class CompositeResult:
    image: np.ndarray           # (1024,1024,3) uint8 — final RGB image
    mask: np.ndarray            # (1024,1024) uint8 — values 0-4 ONLY (no 255)

# pipeline/layout/generator.py (also used in export)
@dataclass
class SampleMetadata:
    sample_id: int
    seed: int
    gsd_cm: float
    classes_present: list[int]
    pixel_counts: dict[int,int]
    placement_log: dict[str,bool]
    split: str                  # 'train' | 'val' | 'test' — assigned in Stage 09
    source: str                 # 'synthetic' | 'real'
    background_tile_path: str
    mask_path: str
    image_path: str

# pipeline/quality/filter.py
@dataclass
class QualityResult:
    passed: bool
    blur_score: float           # Laplacian variance — reject if < 100.0
    class_coverage: dict[int,int]  # {class_id: pixel_count}
    rejection_reason: str | None
```

---

## Bug Fixes Applied vs. Original Design

| # | Problem in original design | Fix in this design |
|---|---|----|
| 1 | Road pixels (value 255) were never converted to background (0) before mask export — invalid class values in saved masks | `export_mask()` in `03_dataset_generator`: `np.where(canvas==255, 0, canvas)` |
| 2 | SDXL generates a full scene but no process was defined for extracting per-object patches | `extract_object_patches()` in `05_controlnet_generation` uses `cv2.connectedComponents` per class |
| 3 | "Real-ESRGAN ×2.5" is not a native mode (only ×2 and ×4 exist) | `upscale_to_5cm()` in `04_super_resolution`: ×4 Real-ESRGAN → `cv2.resize` to (1024,1024) for net ×2.5 |
| 4 | Real annotated test set requirement was stated but no process was defined | `09_export` documents the mandatory LabelStudio/CVAT annotation step as a prerequisite before final evaluation |
| 5 | Per-object ±10% scale jitter in compositor creates mask-image misalignment | Scale jitter removed; objects pasted at exact layout pixel positions; background variety comes from tile selection |
| 6 | CLIP score quality filter is unreliable for aerial imagery (CLIP not trained on orthophotos) | Replaced with Laplacian variance blur detection + minimum class pixel coverage checks |

---

## Project Directory Structure

```
pipeline/
├── pipeline_config.yaml         # single source of truth for all parameters
├── config.py                    # typed loader → PipelineConfig
├── visualize_samples.py         # debug visualization grid
├── layout/
│   ├── __init__.py
│   ├── canvas.py                # init_canvas, place_roads
│   ├── constants.py             # legacy (superseded by config.py)
│   ├── generator.py             # generate_layout, generate_dataset, export_mask
│   ├── placers.py               # place_tennis_courts, place_pools, place_eucalyptus, place_crosswalks
│   └── palette.py               # mask_to_ade20k_rgb, mask_to_canny_edges   ← new
├── sr/
│   └── super_resolution.py      # load_and_crop_tile, upscale_to_5cm         ← new
├── generation/
│   └── controlnet_pipeline.py   # build_conditions, generate_sdxl_scene,
│                                #   extract_object_patches, run_generation    ← new
├── compositor/
│   └── compositor.py            # blend_patch, composite                      ← new
├── quality/
│   └── filter.py                # check_blur, check_class_coverage,
│                                #   filter_sample, filter_dataset             ← new
├── augmentation/
│   └── augmentor.py             # augment_sample, augment_dataset             ← new
├── export/
│   └── splitter.py              # stratified_split, export_split,
│                                #   write_manifests                           ← new
├── test_canvas.py
├── test_placers.py
└── design/                      # this directory
    ├── 00_system_overview.md
    ├── 01_config.md
    ├── 02_layout_engine.md
    ├── 03_dataset_generator.md
    ├── 04_super_resolution.md
    ├── 05_controlnet_generation.md
    ├── 06_compositor.md
    ├── 07_quality_gate.md
    ├── 08_augmentation.md
    └── 09_export.md
```

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | orchestration |
| numpy + OpenCV | latest | layout engine, compositing, blur detection |
| rasterio / GDAL | 3.8 | GeoTIFF reading, reprojection (EPSG:2039) |
| Real-ESRGAN | latest | super-resolution (×4 then resize) |
| diffusers | 0.27+ | SDXL + Multi-ControlNet |
| albumentations | 1.4 | augmentation |
| PyTorch | 2.2+ | model inference (SR + SDXL) |
| LabelStudio / CVAT | latest | manual annotation of real test set |

**Removed from original:** `clip-interrogator` (replaced by OpenCV Laplacian blur check).

---

## End-to-End Run Order

```bash
# 1. Install CPU deps
pip install numpy opencv-python pyyaml rasterio albumentations

# 2. Verify layout engine (existing)
cd pipeline
python test_canvas.py
python test_placers.py

# 3. Generate layout masks + metadata (no GPU needed)
python -c "
from config import load_config
from layout.generator import generate_dataset
cfg = load_config()
generate_dataset(n=1000, cfg=cfg, output_dir='output/layouts')
"

# 4. Process gov.il GeoTIFF background tiles (CPU, ~30min)
python -c "
from config import load_config
from sr.super_resolution import process_geotiff_tiles
cfg = load_config()
process_geotiff_tiles(geotiff_paths=[...], output_dir='output/backgrounds',
                      n_crops_per_tile=5, cfg=cfg,
                      model_path='weights/RealESRGAN_x4plus.pth')
"

# 5. ControlNet SDXL generation (GPU required — run on Colab/RunPod)
python -c "
from generation.controlnet_pipeline import run_generation, load_pipeline
pipeline = load_pipeline()
# ... loop over layout results
"

# 6. Composite + quality filter + augment + export (CPU)
python -c "
from compositor.compositor import composite
from quality.filter import filter_dataset
from augmentation.augmentor import augment_dataset
from export.splitter import stratified_split, export_split, write_manifests
# ... orchestration script
"
```

---

## Class Balance Strategy

Expected pixel distribution and recommended training loss weights:

| Class | ID | Avg % pixels | Loss weight |
|-------|----|-------------|-------------|
| Background | 0 | ~75% | 0.3 |
| Tennis court | 2 | ~8% | 2.0 |
| Eucalyptus | 4 | ~5% | 3.0 |
| Pool | 3 | ~2.5% | 4.0 |
| Crosswalk | 1 | ~0.3% | 5.0 |

Use **Focal Loss** (γ=2) + class weights. Crosswalk patches should be oversampled ×3 during training.

---

## Known Limitations

1. **Domain gap** — SDXL aerial is not true 5cm/px orthophoto. Mitigated by using real backgrounds.
2. **Super-res artifacts** — ×4 then resize sharpens texture but doesn't add real detail. Acceptable for segmentation training.
3. **Eucalyptus realism** — Perlin noise blobs approximate canopy shape; color prompting needed for silver-green tone.
4. **Geographic bias** — gov.il tiles denser in central Israel; oversample north/south explicitly.
5. **Synthetic val is not a proxy for real-world performance** — only the real annotated test set is an honest metric.
