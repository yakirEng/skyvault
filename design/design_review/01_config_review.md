# Design Review — `design/01_config.md`

**Reviewer:** Senior ML Engineer
**Date:** 2026-03-06
**Files reviewed:** `design/01_config.md`, `design/00_system_overview.md`
**Scope:** Pixel value correctness, YAML completeness, missing fields needed by downstream components, ambiguities.

---

## Pixel Value Verification

All derived pixel values were manually verified against the formula `px = round(value_m / (gsd_cm / 100))` at `gsd_cm = 5`:

| Field | Calculation | Result | Stated | Status |
|-------|-------------|--------|--------|--------|
| `tennis_court_w_px` | round(23.77 / 0.05) | 475 | 475 | ✓ |
| `tennis_court_h_px` | round(10.97 / 0.05) | 219 | 219 | ✓ |
| `crosswalk_stripe_px` | round(0.50 / 0.05) | 10 | 10 | ✓ |
| `crosswalk_gap_px` | round(0.50 / 0.05) | 10 | 10 | ✓ |
| `crosswalk_len_px` | round(3.00 / 0.05) | 60 | 60 | ✓ |
| `pool_min_px` | round(4.0 / 0.05) | 80 | 80 | ✓ |
| `pool_max_px` | round(12.0 / 0.05) | 240 | 240 | ✓ |
| `eucalyptus_min_px` | round(5.0 / 0.05) | 100 | 100 | ✓ |
| `eucalyptus_max_px` | round(15.0 / 0.05) | 300 | 300 | ✓ |
| `road_width_px` | round(3.5 / 0.05) | 70 | 70 | ✓ |

All pixel values are arithmetically correct at the nominal GSD of 5 cm/px.

---

## Findings

### 1. `total_length_m` for crosswalk is critically ambiguous

**Severity: CRITICAL**

The YAML defines:
```yaml
crosswalk:
  stripe_width_m: 0.50
  stripe_gap_m:   0.50
  total_length_m: 3.00
```

`total_length_m` is undefined: does it mean (a) the length of each individual stripe (how far it extends across the road), or (b) the total span of the crosswalk along the road direction (the extent cars must stop for)?

If (a): stripe length = 60px, but the number of stripes is completely unspecified — the placer cannot be implemented.
If (b): total span = 60px along the road, and the number of stripes is implicitly `floor(60 / (10+10)) = 3` — but this derivation is never stated, making the config opaque to implementors.

A crosswalk sits ON the road (70px wide). Interpretation (b) is almost certainly intended — each stripe spans the full road width (70px), and the total crosswalk box is 60px × 70px with 3 stripes. But interpretation (a) is equally valid syntactically, and a 60px stripe does not span the 70px-wide road cleanly.

**Suggested fix:** Rename `total_length_m` to `total_span_m` (along the road direction). Add a note stating the stripe length is always equal to `road.width_m`. Explicitly document the implied stripe count formula: `n_stripes = floor(total_span_m / (stripe_width_m + stripe_gap_m))`. Consider adding `n_stripes` as an explicit YAML field to avoid silent derivation.

---

### 2. Source GSD missing from config — SR crop size silently depends on hardcoded assumption

**Severity: CRITICAL**

`design/04_super_resolution.md` hardcodes `crop_size_px = 410` derived from the gov.il tile resolution of 12.5 cm/px:
```
crop_size_px = round(canvas_px * gsd_cm / source_gsd_cm) = round(1024 * 5 / 12.5) = 409.6 ≈ 410
```

Neither `source_gsd_cm: 12.5` nor `crop_size_px: 410` appears anywhere in `pipeline_config.yaml` or the `PipelineConfig` dataclass. If `gsd_cm` or `canvas_px` is changed in the YAML, the SR component's crop size becomes silently wrong — the whole point of the config being "single source of truth" is violated.

**Suggested fix:** Add to YAML:
```yaml
super_resolution:
  source_gsd_cm: 12.5   # gov.il orthophoto native resolution
```
The `crop_size_px` should then be a derived field in `PipelineConfig`, computed at load time as `round(canvas_px * gsd_cm / source_gsd_cm)`, not a magic number in the SR module.

---

### 3. `ROAD_INTERNAL = 255` is a mutable dataclass field — should be a module constant

**Severity: CRITICAL**

`ROAD_INTERNAL` is defined as a field on `PipelineConfig` with a default value of 255:
```python
@dataclass
class PipelineConfig:
    ...
    ROAD_INTERNAL: int = 255
```

Being a dataclass field (even with a default) means:
- It can be accidentally overridden when constructing `PipelineConfig` programmatically or in tests.
- If set to 0, 1, 2, 3, or 4 it silently collides with a valid class ID, corrupting the canvas without any error.
- Python dataclasses with `UPPER_CASE` field names violate convention — it reads as a constant but behaves as an instance attribute.

`load_config()` does not validate that `ROAD_INTERNAL` doesn't equal any class ID.

**Suggested fix:** Remove from the dataclass. Define it as a module-level constant in `config.py`:
```python
ROAD_INTERNAL_VALUE: int = 255  # sentinel used internally in canvas; never a class ID
```
If downstream code needs it from `cfg`, add a property `@property def road_internal(self) -> int: return 255` that is read-only.

---

### 4. Maximum object counts per canvas not in YAML — violates single-source-of-truth principle

**Severity: WARNING**

`design/02_layout_engine.md` documents that placers generate "0-2 tennis courts, 0-3 pools, 0-5 eucalyptus". These counts are not in `pipeline_config.yaml` or `PipelineConfig`. Tuning them (a common ML dataset iteration task) requires code changes in `placers.py` rather than a YAML edit, which directly violates the stated design principle: "change one value and the entire pipeline rescales."

**Suggested fix:** Add to YAML:
```yaml
objects:
  tennis_court:
    ...
    max_count: 2
  pool:
    ...
    max_count: 3
  eucalyptus:
    ...
    max_count: 5
  crosswalk:
    ...
    max_count: 4    # max crosswalks per canvas
```
Add `max_count: int` to the relevant per-object dataclass or to `ObjectConfig`.

---

### 5. Pool shape is completely unspecified

**Severity: WARNING**

The config defines only `min_m` and `max_m` for pools, but says nothing about shape. Pools are realistically oval or rectangular — the placer must know which, and if rectangular, whether both dimensions are independently sampled in [min_m, max_m] or if an aspect ratio constraint applies.

Without shape specification, two implementors of `place_pools()` could produce incompatible outputs (circles vs. rectangles), making the design non-deterministic across implementations.

**Suggested fix:** Add shape parameters to YAML:
```yaml
pool:
  min_m: 4.0
  max_m: 12.0
  shape: rectangular    # 'rectangular' | 'elliptical'
  max_aspect_ratio: 2.5  # max(w,h)/min(w,h) — prevents degenerate thin pools
```

---

### 6. Augmentation count (`n_per_sample`) hardcoded — should be in config

**Severity: WARNING**

`design/08_augmentation.md` hardcodes `n: int = 4` as the default number of augmented variants per sample. This is a key dataset size parameter: `total_samples = n_base * n_per_sample`. Changing it to 8 (to get 8000 samples) requires a code edit. It belongs in the config as `augmentation.n_per_sample: 4`.

**Suggested fix:** Add to YAML:
```yaml
augmentation:
  n_per_sample: 4
```
Add `n_per_sample: int` to a new `AugmentationConfig` dataclass nested under `PipelineConfig`.

---

### 7. Quality gate thresholds hardcoded — should be in config

**Severity: WARNING**

`design/07_quality_gate.md` defines:
- `BLUR_THRESHOLD: float = 100.0`
- `MIN_PIXELS_PER_CLASS: dict = {1: 500, 2: 5000, 3: 1000, 4: 2000}`

These are tuneable ML hyperparameters. After an initial generation run, you may inspect the distribution of blur scores and decide to tighten or loosen the threshold. Having them hardcoded requires code changes and makes tuning experiments difficult to track.

**Suggested fix:** Add to YAML:
```yaml
quality:
  blur_threshold: 100.0
  min_pixels:
    crosswalk:    500
    tennis_court: 5000
    pool:         1000
    eucalyptus:   2000
```

---

### 8. Dataset split ratios hardcoded — should be in config

**Severity: WARNING**

`design/09_export.md` defaults `split_ratios = (0.70, 0.15, 0.15)` as a function argument. This is a dataset design decision that should be tracked in version control via config, not buried in a function signature.

**Suggested fix:** Add to YAML:
```yaml
export:
  split_ratios:
    train: 0.70
    val:   0.15
    test:  0.15
  split_seed: 0
```

---

### 9. `canvas_size` (YAML) vs. `canvas_px` (dataclass) naming mismatch is implicit

**Severity: WARNING**

The YAML uses `resolution.canvas_size` but the dataclass field is `canvas_px`. The design says "`canvas_px` is read directly from YAML" — but the YAML key is `canvas_size`, not `canvas_px`. The loader must perform a key rename with no guidance. This will cause a `KeyError` if the implementor uses the dataclass field name as the YAML key.

**Suggested fix:** Either rename the YAML key to `canvas_px` to match the dataclass field, or add an explicit note in the implementation section: "`canvas_px` is loaded from YAML key `resolution.canvas_size`."

---

### 10. No range and consistency validation specified for `load_config()`

**Severity: WARNING**

`load_config()` is documented to raise `ValueError` "if any derived pixel value is 0 or negative." But these additional consistency checks are never mentioned:
- `pool_min_m < pool_max_m`
- `eucalyptus_min_m < eucalyptus_max_m`
- `n_horizontal[0] <= n_horizontal[1]` and `n_vertical[0] <= n_vertical[1]`
- `padding_frac` is in `(0.0, 0.5)` — more than 0.5 makes the padded area larger than the canvas
- `ROAD_INTERNAL` (255) does not equal any class ID value (0–4)
- Crosswalk dimensions: `crosswalk_stripe_px + crosswalk_gap_px <= road_width_px` (a crosswalk period must fit in the road)
- Tennis court fits in canvas after padding: `tennis_court_w_px < canvas_px * (1 - 2 * padding_frac)`

A misconfigured YAML will silently produce degenerate or impossible layouts with no clear error message.

**Suggested fix:** List all validation assertions explicitly in the design doc's Functions section so the implementor knows exactly what to check.

---

### 11. Python's banker's rounding (`round()`) is not conventional rounding

**Severity: MINOR**

The formula `px = round(value_m / (gsd_cm / 100))` uses Python 3's built-in `round()`, which rounds half-to-even (banker's rounding), not half-up. At current values this doesn't matter (no values produce a .5 fractional result). However, at `gsd_cm = 10`, `pool_min_m / 0.10 = 40.0` — still fine. But at `gsd_cm = 7`: `10.97 / 0.07 = 156.71...` — fine. The risk is low but real for future GSD changes.

Example: `round(2.5)` in Python 3 → `2` (not `3`). If a future GSD change causes `value_m / (gsd_cm/100)` to land on exactly X.5, the result would be unexpected.

**Suggested fix:** Use `int(value_m / (gsd_cm / 100) + 0.5)` or `math.floor(value_m / (gsd_cm / 100) + 0.5)` for conventional half-up rounding and document the choice.

---

### 12. `n_horizontal`/`n_vertical` inclusive range not documented

**Severity: MINOR**

The YAML comment says "inclusive range" for `n_horizontal: [1, 3]`. It's unclear whether the range is inclusive-inclusive `[1, 3]` (can place 3 roads) or inclusive-exclusive `[1, 3)` (can place 1 or 2). The placer implementation must match the config intent. `numpy.random.Generator.integers` uses exclusive upper bound by default; `random.randint` uses inclusive. A one-off error here means the maximum road count may never be reached or may exceed the stated limit.

**Suggested fix:** Change YAML comment to: `# inclusive on both ends — i.e., 1, 2, or 3 roads`. Add a note in Implementation Notes specifying which random API to use: `rng.integers(low=n_min, high=n_max+1)`.

---

### 13. Tennis court feasibility constraint not documented

**Severity: MINOR**

With `canvas_px = 1024` and `padding_frac = 0.125`, the usable canvas after road padding is approximately `1024 * (1 - 2 * 0.125) = 768px`. A tennis court at 0° rotation is 475px wide. Two tennis courts side-by-side horizontally would require `2 * 475 = 950px > 768px` — they cannot be placed horizontally adjacent. The `max_count: 2` constraint (once added) can be satisfied, but only if the placer handles rotation and spacing correctly. The config design does not document this feasibility constraint or guarantee that valid placements exist.

**Suggested fix:** Add a design note: "With default values, at most one tennis court can be placed horizontally (0°); a second must be placed at 90° or in a non-adjacent canvas region. The placer must verify fit after rotation."

---

### 14. ControlNet conditioning weights not in config

**Severity: MINOR**

`design/05_controlnet_generation.md` hardcodes `conditioning_scale = [0.8, 0.5]` for seg and edge ControlNets. This is a generation quality tuning parameter. Different prompts and model versions may require different values. Should be in config.

**Suggested fix:** Add to YAML:
```yaml
generation:
  conditioning_scale: [0.8, 0.5]   # [seg_weight, edge_weight]
  sdxl_model: "stabilityai/stable-diffusion-xl-base-1.0"
  controlnet_seg: "diffusers/controlnet-seg-sdxl-1.0"
  controlnet_canny: "diffusers/controlnet-canny-sdxl-1.0"
```

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | CRITICAL | `total_length_m` for crosswalk is ambiguous — stripe count undefined |
| 2 | CRITICAL | Source GSD (12.5 cm/px) missing from config; SR crop size silently breaks on GSD/canvas change |
| 3 | CRITICAL | `ROAD_INTERNAL` is a mutable dataclass field; can be accidentally overridden to collide with a class ID |
| 4 | WARNING | Object max counts per canvas not in YAML — violates single-source-of-truth |
| 5 | WARNING | Pool shape (rectangle/ellipse) and aspect ratio not specified |
| 6 | WARNING | Augmentation `n_per_sample` hardcoded as 4, not in config |
| 7 | WARNING | Quality gate thresholds hardcoded, not in config |
| 8 | WARNING | Dataset split ratios hardcoded, not in config |
| 9 | WARNING | YAML key `canvas_size` does not match dataclass field `canvas_px` — implicit rename |
| 10 | WARNING | `load_config()` validation list is incomplete — missing 7 feasibility checks |
| 11 | MINOR | Python banker's rounding may surprise maintainers at future GSD values |
| 12 | MINOR | `n_horizontal`/`n_vertical` inclusive vs exclusive range ambiguity |
| 13 | MINOR | Tennis court + canvas feasibility constraint undocumented |
| 14 | MINOR | ControlNet conditioning weights not in config |

**3 CRITICAL, 7 WARNING, 4 MINOR findings.**
