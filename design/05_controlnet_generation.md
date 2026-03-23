# Component 05 — ControlNet Generation

## Purpose

Run Multi-ControlNet SDXL conditioned on the layout mask to generate a synthetic aerial scene image. Then extract per-instance object patches from the SDXL output for use by the compositor.

**Fix #2 applied here:** The original design stated "paste SDXL object patches" but never defined how to extract them from the full SDXL output. `extract_object_patches()` fills this gap using connected-component analysis on the layout mask.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/generation/controlnet_pipeline.py` | All generation logic — conditioning, SDXL inference, patch extraction |

---

## Dependencies

- `01_config` — `PipelineConfig`
- `02_layout_engine` — `LayoutResult`
- `03_dataset_generator` — `export_mask()`, `mask_to_ade20k_rgb()`, `mask_to_canny_edges()`
- External: `diffusers>=0.27`, `torch>=2.2`, `Pillow`

---

## Inputs

### `build_conditions()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout_result` | `LayoutResult` | From `generate_layout()` |

### `generate_sdxl_scene()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `seg_condition` | `np.ndarray` `(1024,1024,3)` `uint8` | ADE20K RGB palette image |
| `edge_condition` | `np.ndarray` `(1024,1024)` `uint8` | Canny edge map |
| `pipeline` | `StableDiffusionXLControlNetPipeline` | Loaded diffusers pipeline |
| `seed` | `int` | For `torch.Generator` reproducibility |
| `conditioning_scale` | `list[float]` | Default `[0.8, 0.5]` — seg first, edge second |

### `extract_object_patches()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene` | `np.ndarray` `(1024,1024,3)` `uint8` | Full SDXL output image |
| `normalized_mask` | `np.ndarray` `(1024,1024)` `uint8` | Values 0-4 (from `export_mask()`) |

### `run_generation()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout_result` | `LayoutResult` | |
| `pipeline` | `StableDiffusionXLControlNetPipeline` | |
| `cfg` | `PipelineConfig` | |

---

## Outputs

### `build_conditions()`

```
Returns: tuple[np.ndarray, np.ndarray]
    seg_condition:  (1024,1024,3) uint8 — ADE20K RGB
    edge_condition: (1024,1024)   uint8 — Canny edges
```

### `generate_sdxl_scene()`

```
Returns: np.ndarray shape (1024,1024,3) dtype uint8 — full SDXL scene
```

### `extract_object_patches()` — FIX #2

```
Returns: list[ObjectPatch]

ObjectPatch:
    image:    (h, w, 3) uint8  — RGB crop from SDXL output at the object's bounding box
    alpha:    (h, w)    bool   — True where layout mask == class_id within the bounding box
    class_id: int              — 1-4
    bbox:     (x1, y1, x2, y2) — in layout pixel coordinates (same coordinate space as mask)
```

One `ObjectPatch` per connected component per class. Background (0) is skipped.

### `run_generation()`

```
Returns: SDXLResult
    scene:          (1024,1024,3) uint8 — full SDXL output
    patches:        list[ObjectPatch]
    seg_condition:  (1024,1024,3) uint8
    edge_condition: (1024,1024)   uint8
```

---

## Functions

```python
POSITIVE_PROMPT = (
    "ultra-high resolution, 5cm per pixel drone photography, "
    "top-down bird's eye view, Levantine architecture, "
    "Israeli residential neighborhood, harsh Mediterranean sunlight, "
    "sharp shadows, flat roofs with solar water heaters, "
    "arid dry soil, dusty asphalt"
)

NEGATIVE_PROMPT = (
    "slanted roofs, snow, european architecture, "
    "isometric view, blurry, low resolution, satellite view"
)

# Minimum pixel area for an extracted patch to be included
MIN_COMPONENT_AREA: dict[int, int] = {
    1: 200,    # crosswalk — small
    2: 5000,   # tennis court — large
    3: 500,    # pool — medium
    4: 1000,   # eucalyptus — medium
}


def build_conditions(layout_result: LayoutResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare ControlNet conditioning images from a LayoutResult.

    Steps:
    1. normalized_mask = export_mask(layout_result.canvas)  # road pixels → 0
    2. seg_condition   = mask_to_ade20k_rgb(normalized_mask)
    3. edge_condition  = mask_to_canny_edges(normalized_mask)

    Returns: (seg_condition, edge_condition)
    """


def generate_sdxl_scene(
    seg_condition: np.ndarray,
    edge_condition: np.ndarray,
    pipeline,                      # StableDiffusionXLControlNetPipeline
    seed: int,
    conditioning_scale: list[float] = [0.8, 0.5]
) -> np.ndarray:
    """
    Run SDXL inference with two ControlNet conditions.

    Converts seg_condition and edge_condition to PIL Images before passing to pipeline.
    Uses torch.Generator(device).manual_seed(seed) for reproducibility.
    Returns the generated image as (1024,1024,3) uint8 numpy array.
    """


def extract_object_patches(
    scene: np.ndarray,
    normalized_mask: np.ndarray,
) -> list[ObjectPatch]:
    """
    FIX #2: Extract per-instance object patches from SDXL output.

    For each class_id in [1, 2, 3, 4]:
        binary = (normalized_mask == class_id).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(binary)
        for label_id in range(1, n_labels):
            component = (labels == label_id)
            area = component.sum()
            if area < MIN_COMPONENT_AREA[class_id]:
                continue
            rows = np.where(component.any(axis=1))[0]
            cols = np.where(component.any(axis=0))[0]
            y1, y2 = rows[0], rows[-1] + 1
            x1, x2 = cols[0], cols[-1] + 1
            image = scene[y1:y2, x1:x2].copy()
            alpha = component[y1:y2, x1:x2]
            patches.append(ObjectPatch(image, alpha, class_id, (x1, y1, x2, y2)))

    Returns: list[ObjectPatch], ordered by class_id then top-to-bottom.
    """


def load_pipeline(device: str = "cuda") -> object:  # StableDiffusionXLControlNetPipeline
    """
    Load Multi-ControlNet SDXL pipeline.

    controlnets = [
        ControlNetModel.from_pretrained("diffusers/controlnet-seg-sdxl-1.0",  torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16),
    ]
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnets,
        torch_dtype=torch.float16
    ).to(device)
    pipeline.enable_model_cpu_offload()

    Returns: loaded pipeline object.
    """


def run_generation(
    layout_result: LayoutResult,
    pipeline,
    cfg: PipelineConfig
) -> SDXLResult:
    """
    Orchestrate: build_conditions → generate_sdxl_scene → extract_object_patches.
    Returns SDXLResult.
    """
```

---

## Installation

```bash
pip install diffusers>=0.27 transformers accelerate torch>=2.2
```

**GPU required.** Recommended: Google Colab A100 or RunPod (~$1-2/hr). VRAM requirement: 16GB minimum for float16 SDXL.

---

## Implementation Notes

- `edge_condition` must be passed as a 3-channel image to diffusers (repeat grayscale to RGB): `np.stack([edges]*3, axis=2)`
- The seg ControlNet conditions the coarse layout (what goes where); the edge ControlNet sharpens object boundaries — hence higher weight (0.8) for seg
- `extract_object_patches()` extracts patches at the SAME pixel coordinates as the layout mask. No coordinate transformation occurs — this is what guarantees alignment in the compositor (Fix #5 in `06_compositor`)
- SDXL generates the full scene including background areas. Only object pixels are used (via `alpha` masks in each `ObjectPatch`) — the SDXL background is discarded in favor of the real tile
