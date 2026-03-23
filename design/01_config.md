# Component 01 — Config

## Purpose

Single source of truth for all pipeline parameters. All pixel dimensions are derived at load time from `gsd_cm` so that changing one value rescales the entire pipeline. No other component hard-codes pixel sizes.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/pipeline_config.yaml` | Raw YAML parameters (meters, counts, class IDs) |
| `pipeline/config.py` | Typed dataclass loader — converts meters → pixels, returns `PipelineConfig` |

---

## Dependencies

None. All other components depend on this one.

---

## Input

```
pipeline_config.yaml path (str, default "pipeline_config.yaml")
```

**`pipeline_config.yaml` schema:**
```yaml
resolution:
  gsd_cm: 5           # ground sample distance in cm
  canvas_size: 1024   # output canvas width and height in pixels

classes:
  background:   0
  crosswalk:    1
  tennis_court: 2
  pool:         3
  eucalyptus:   4

objects:
  tennis_court:
    width_m:  23.77   # ITF standard
    height_m: 10.97
    rotations: [0, 90]
  crosswalk:
    stripe_width_m: 0.50
    stripe_gap_m:   0.50
    total_length_m: 3.00
  pool:
    min_m:  4.0
    max_m: 12.0
  eucalyptus:
    min_m:  5.0
    max_m: 15.0

roads:
  width_m:      3.5
  n_horizontal: [1, 3]   # inclusive range — number of horizontal roads per canvas
  n_vertical:   [1, 3]
  padding_frac: 0.125    # fraction of canvas kept clear of roads at edges
```

---

## Output

`PipelineConfig` dataclass instance — all `_px` fields pre-computed.

```python
@dataclass
class ResolutionConfig:
    gsd_cm: float       # 5.0
    canvas_px: int      # 1024

@dataclass
class ObjectConfig:
    tennis_court_w_px: int       # round(23.77 / 0.05) = 475
    tennis_court_h_px: int       # round(10.97 / 0.05) = 219
    tennis_rotations: list[int]  # [0, 90]
    crosswalk_stripe_px: int     # round(0.50 / 0.05) = 10
    crosswalk_gap_px: int        # round(0.50 / 0.05) = 10
    crosswalk_len_px: int        # round(3.00 / 0.05) = 60
    pool_min_px: int             # round(4.0  / 0.05) = 80
    pool_max_px: int             # round(12.0 / 0.05) = 240
    eucalyptus_min_px: int       # round(5.0  / 0.05) = 100
    eucalyptus_max_px: int       # round(15.0 / 0.05) = 300

@dataclass
class RoadConfig:
    width_px: int                # round(3.5 / 0.05) = 70
    n_horizontal: tuple[int,int] # (1, 3)
    n_vertical: tuple[int,int]   # (1, 3)
    padding_frac: float          # 0.125

@dataclass
class PipelineConfig:
    resolution: ResolutionConfig
    objects: ObjectConfig
    roads: RoadConfig
    class_ids: dict[str, int]    # {'background':0, 'crosswalk':1, 'tennis_court':2, 'pool':3, 'eucalyptus':4}
    ROAD_INTERNAL: int = 255     # canvas value for roads during layout — never exported
```

---

## Functions

```python
def load_config(path: str = "pipeline_config.yaml") -> PipelineConfig:
    """
    Load and parse pipeline_config.yaml.
    Derives all _px fields using: px = round(value_m / (gsd_cm / 100))
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if any derived pixel value is 0 or negative.
    """
```

---

## Implementation Notes

- Pixel conversion formula: `px = round(meters / (gsd_cm / 100.0))`
- `ROAD_INTERNAL = 255` is a constant, never read from YAML — it must not equal any class ID (0-4)
- Raw meter values must NOT be used outside `config.py` — always use the `_px` fields downstream
- `canvas_px` is read directly from YAML (not derived), since it defines the output resolution independently
