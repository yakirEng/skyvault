# Component 07 — Quality Gate

## Purpose

Filter out low-quality composited samples before augmentation. Two checks are applied: blur detection (Laplacian variance) and minimum class pixel coverage.

**Fix #6 applied here:** The original design used CLIP score (`clip-interrogator`) as a quality metric. CLIP was trained on internet images and has limited representation of aerial orthophoto imagery, making its scores unreliable in this domain. Replaced with deterministic, domain-agnostic checks that require no external model and run on CPU.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/quality/filter.py` | `check_blur()`, `check_class_coverage()`, `filter_sample()`, `filter_dataset()` |

---

## Dependencies

- `06_compositor` — `CompositeResult`
- `03_dataset_generator` — `SampleMetadata`
- `01_config` — `PipelineConfig`
- External: `opencv-python`, `numpy`

---

## Inputs

### `check_blur()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` `(H,W,3)` `uint8` | RGB image to evaluate |
| `threshold` | `float` | Reject if score below this (default `100.0`) |

### `check_class_coverage()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `np.ndarray` `(H,W)` `uint8` | Segmentation mask, values 0-4 |
| `classes_expected` | `list[int]` | Classes that must be present (from `SampleMetadata.classes_present`) |
| `min_pixels_per_class` | `dict[int, int]` | Minimum pixel count per class |

### `filter_sample()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `composite` | `CompositeResult` | Image + mask from compositor |
| `metadata` | `SampleMetadata` | For `classes_present` |
| `cfg` | `PipelineConfig` | (reserved for future thresholds) |

### `filter_dataset()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `composites` | `list[CompositeResult]` | All samples |
| `metadatas` | `list[SampleMetadata]` | Matching metadata list |
| `cfg` | `PipelineConfig` | |

---

## Outputs

### `check_blur()`

```
Returns: float — Laplacian variance score
    High = sharp image (keep)
    Low  = blurry image (reject if < threshold)
```

### `check_class_coverage()`

```
Returns: dict[int, int] — {class_id: pixel_count} for each class in classes_expected
    Any count below min_pixels_per_class[class_id] → QualityResult.passed = False
```

### `filter_sample()`

```
Returns: QualityResult
    passed:           bool
    blur_score:       float
    class_coverage:   dict[int, int]
    rejection_reason: str | None  — human-readable reason for first failure, else None
```

### `filter_dataset()`

```
Returns: tuple[list[CompositeResult], list[SampleMetadata]]
    Filtered to only passing samples.
    Logs: total count, n_passed, n_rejected, rejection reason breakdown.
```

---

## Functions

```python
# Minimum pixel count per class for a sample to pass
MIN_PIXELS_PER_CLASS: dict[int, int] = {
    1: 500,    # crosswalk  — small stripes, but must be visible
    2: 5000,   # tennis     — large court, should cover significant area
    3: 1000,   # pool       — medium
    4: 2000,   # eucalyptus — medium canopy
}

BLUR_THRESHOLD: float = 100.0


def check_blur(image: np.ndarray, threshold: float = BLUR_THRESHOLD) -> float:
    """
    Compute Laplacian variance as a sharpness metric.

    Implementation:
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

    Returns: float — reject sample if score < threshold.
    Typical values: blurry composite ~20-50, sharp orthophoto ~150-500.
    """


def check_class_coverage(
    mask: np.ndarray,
    classes_expected: list[int],
    min_pixels_per_class: dict[int, int] = MIN_PIXELS_PER_CLASS,
) -> dict[int, int]:
    """
    Count pixels per expected class and verify minimums are met.

    Returns: {class_id: pixel_count} for each class_id in classes_expected.
    Does NOT return a pass/fail — filter_sample() applies the threshold logic.
    """


def filter_sample(
    composite: CompositeResult,
    metadata: SampleMetadata,
    cfg: PipelineConfig,
) -> QualityResult:
    """
    Run all quality checks on a single composited sample.

    Checks (in order — short-circuits on first failure):
    1. Blur: if check_blur(composite.image) < BLUR_THRESHOLD → fail
    2. Coverage: for each class in metadata.classes_present (excluding 0):
           if pixel_count < MIN_PIXELS_PER_CLASS[class_id] → fail

    Returns: QualityResult with passed=True only if all checks pass.
    rejection_reason examples:
        "blur_score=42.3 below threshold=100.0"
        "class 2 (tennis) has 200 pixels, minimum is 5000"
    """


def filter_dataset(
    composites: list[CompositeResult],
    metadatas: list[SampleMetadata],
    cfg: PipelineConfig,
) -> tuple[list[CompositeResult], list[SampleMetadata]]:
    """
    Apply filter_sample() to all samples and return only passing ones.

    Logs a rejection summary:
        Total: N  |  Passed: P  |  Rejected: R
        Rejection reasons:
          blur_too_low:      X
          class_coverage:    Y

    Returns: (passing_composites, passing_metadatas) — lists are parallel (same indices).
    """
```

---

## Implementation Notes

- **No external models required** — both checks use only numpy and OpenCV
- **Blur threshold rationale:** real orthophoto tiles typically score 150-500 on Laplacian variance. Poorly blended composites score below 100. Threshold of 100 rejects the worst composites while keeping realistic-looking samples. Tune by visually inspecting rejected samples.
- **Class coverage rationale:** `classes_present` is set by the layout engine based on placement success. If a class was placed in the layout but is invisible in the composite (e.g., completely covered during blending), the sample is misleading for training.
- **What CLIP would have caught but this doesn't:** CLIP could theoretically detect semantically wrong outputs (e.g., SDXL generating a swimming pool that looks like asphalt). The coverage check is a weaker substitute — it only verifies pixel count, not visual correctness. This trade-off is acceptable given the domain mismatch of CLIP with aerial imagery.
