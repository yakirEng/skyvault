# Component 08 — Augmentation

## Purpose

Expand ~1000 base samples to ~4000 by applying ×4 deterministic augmentations per sample. Augmentations are applied identically to image and mask to maintain pixel-level alignment. Mask interpolation uses nearest-neighbor only — bilinear or cubic would corrupt class ID integer values.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/augmentation/augmentor.py` | `augment_sample()`, `augment_dataset()` |

---

## Dependencies

- `01_config` — `PipelineConfig`
- External: `albumentations>=1.4`, `opencv-python`, `numpy`

---

## Inputs

### `augment_sample()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `np.ndarray` `(1024,1024,3)` `uint8` | Composited RGB image |
| `mask` | `np.ndarray` `(1024,1024)` `uint8` | Segmentation mask, values 0-4 |
| `seed` | `int` | Base seed for deterministic augmentation |
| `n` | `int` | Number of augmented variants to produce (default `4`) |

### `augment_dataset()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | `str` | Directory with `images/sample_{i:04d}.png` and `masks/sample_{i:04d}.png` |
| `output_dir` | `str` | Where to write augmented outputs |
| `cfg` | `PipelineConfig` | For any config-dependent logic |

---

## Outputs

### `augment_sample()`

```
Returns: list[tuple[np.ndarray, np.ndarray]] — length n
    Each tuple: (augmented_image, augmented_mask)
        augmented_image: (1024,1024,3) uint8
        augmented_mask:  (1024,1024)   uint8, values 0-4
```

### `augment_dataset()`

```
Saves to {output_dir}/:
    images/augmented_{i:04d}_{j}.png  — j in [0, n-1]
    masks/augmented_{i:04d}_{j}.png

Returns: list[SampleMetadata] — updated metadata for all augmented samples
    (split field still empty string — assigned in Stage 09)
    source: 'synthetic'
    sample_id incremented sequentially across all augmented files
```

---

## Functions

```python
import albumentations as A

AUGMENTATION_PIPELINE = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),                   # random 0/90/180/270° rotation
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.8
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=20,
            val_shift_limit=0,                     # no value shift — preserves luminance
            p=0.8
        ),
        A.GaussNoise(var_limit=(25, 75), p=0.5),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),
    ],
    additional_targets={"mask": "mask"}            # apply SAME spatial transforms to mask
)


def augment_sample(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int,
    n: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate n augmented (image, mask) pairs.

    For each i in range(n):
        Set albumentations random seed to seed * 100 + i for determinism.
        result = AUGMENTATION_PIPELINE(image=image, mask=mask)
        Collect (result["image"], result["mask"])

    Mask interpolation: albumentations uses nearest-neighbor for integer masks
    when target type is "mask" — class IDs are preserved exactly.

    Returns: list of n (image, mask) tuples.
    """


def augment_dataset(
    input_dir: str,
    output_dir: str,
    cfg: PipelineConfig,
    n_per_sample: int = 4,
) -> list[SampleMetadata]:
    """
    Apply augment_sample() to every base sample in input_dir.

    Reads: {input_dir}/images/sample_{i:04d}.png
           {input_dir}/masks/sample_{i:04d}.png
           {input_dir}/metadata_{i:04d}.json

    Writes: {output_dir}/images/augmented_{i:04d}_{j}.png
            {output_dir}/masks/augmented_{i:04d}_{j}.png

    Returns: list[SampleMetadata] for all augmented samples.
    """
```

---

## Augmentation Pipeline Rationale

| Transform | Reason |
|-----------|--------|
| `HorizontalFlip` | Aerial imagery has no canonical left-right orientation |
| `VerticalFlip` | Same — nadir view has no up/down |
| `RandomRotate90` | Adds 0/90/180/270° variants; only 90° multiples to avoid interpolation of mask pixels |
| `RandomBrightnessContrast` | Simulates time-of-day lighting variation |
| `HueSaturationValue` | Simulates seasonal color variation (dry vs. green season); `val_shift=0` prevents unnatural brightness changes |
| `GaussNoise` | Simulates sensor noise and JPEG artifacts in aerial imagery |
| `ImageCompression` | Simulates JPEG-compressed orthophoto tiles |

## Why `RandomRotate90` only (not arbitrary rotation)

Arbitrary rotation (e.g., 45°) would require resampling the mask, which at non-90° angles produces fractional pixel values that corrupt class ID integers even with nearest-neighbor interpolation at boundaries. 90° multiples involve only transposition and reflection — no interpolation, no corruption.

---

## Implementation Notes

- `additional_targets={"mask": "mask"}` is critical — without it, color transforms would be applied to the mask (corrupting class IDs), and spatial transforms would not be applied to the mask (breaking alignment)
- Base samples (before augmentation) are NOT included in the augmented output directory — they are kept separately to avoid double-counting in the dataset split
- `sample_id` in `SampleMetadata` for augmented samples: use `base_sample_id * n_per_sample + j` to maintain a unique, deterministic ID scheme
