# Component 09 — Export & Dataset Split

## Purpose

Split the augmented dataset into train/val/test partitions using stratified sampling by class presence. Write the final directory structure, `class_labels.json`, and `metadata.json`.

**Fix #4 applied here:** The original design stated "test set must be real images only" but provided no process for obtaining them. This component defines the real test set assembly as an explicit, mandatory prerequisite step before final evaluation.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/export/splitter.py` | `stratified_split()`, `export_split()`, `write_manifests()` |

---

## Dependencies

- `08_augmentation` — `list[SampleMetadata]`
- External: `shutil`, `json`, `pathlib`

---

## Inputs

### `stratified_split()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadatas` | `list[SampleMetadata]` | All augmented samples |
| `ratios` | `tuple[float,float,float]` | `(train, val, test)` — must sum to 1.0 (default `(0.70, 0.15, 0.15)`) |
| `seed` | `int` | Shuffle seed for reproducibility (default `0`) |

### `export_split()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadatas` | `list[SampleMetadata]` | Samples for one split |
| `source_dir` | `str` | Root directory with all images and masks |
| `output_dir` | `str` | Root output directory (e.g., `dataset/`) |
| `split` | `str` | `'train'` \| `'val'` \| `'test'` |

### `write_manifests()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `train` | `list[SampleMetadata]` | |
| `val` | `list[SampleMetadata]` | |
| `test` | `list[SampleMetadata]` | |
| `output_dir` | `str` | Root dataset directory |

---

## Outputs

### `stratified_split()`

```
Returns: tuple[list[SampleMetadata], list[SampleMetadata], list[SampleMetadata]]
    (train_metadatas, val_metadatas, test_metadatas)
    Each SampleMetadata has split field set: 'train' | 'val' | 'test'
```

### `export_split()`

```
Copies files to:
    {output_dir}/{split}/images/{filename}.png
    {output_dir}/{split}/masks/{filename}.png
Returns: None
```

### Dataset directory structure

```
dataset/
├── train/
│   ├── images/    *.png  (1024×1024×3 uint8 RGB)
│   └── masks/     *.png  (1024×1024 uint8 grayscale, values 0-4)
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/    ← REAL IMAGES ONLY after Fix #4 (see below)
│   └── masks/     ← MANUALLY ANNOTATED ONLY
├── class_labels.json
└── metadata.json
```

### `write_manifests()`

```
Writes:
    {output_dir}/class_labels.json
    {output_dir}/metadata.json
Returns: None
```

**`class_labels.json` schema:**
```json
{
  "0": "background",
  "1": "crosswalk",
  "2": "tennis_court",
  "3": "pool",
  "4": "eucalyptus"
}
```

**`metadata.json` schema:**
```json
{
  "n_train": 2800,
  "n_val":    600,
  "n_test":    50,
  "gsd_cm": 5.0,
  "canvas_px": 1024,
  "samples": [
    {
      "sample_id": 0,
      "seed": 0,
      "gsd_cm": 5.0,
      "classes_present": [0, 1, 2],
      "pixel_counts": {"0": 800000, "1": 3000, "2": 88000},
      "placement_log": {"tennis_1": true},
      "split": "train",
      "source": "synthetic",
      "background_tile_path": "backgrounds/bg_0012.png",
      "mask_path": "dataset/train/masks/augmented_0000_0.png",
      "image_path": "dataset/train/images/augmented_0000_0.png"
    }
  ]
}
```

---

## Functions

```python
def stratified_split(
    metadatas: list[SampleMetadata],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> tuple[list[SampleMetadata], list[SampleMetadata], list[SampleMetadata]]:
    """
    Stratified split by classes_present bitmask.

    Stratification key: frozenset(metadata.classes_present)
    Groups samples by which combination of classes they contain,
    then splits each group proportionally.

    Ensures every class combination present in the full dataset is
    represented in each split (train, val, test).

    Sets metadata.split = 'train' | 'val' | 'test' in place.

    Returns: (train, val, test) lists.
    """


def export_split(
    metadatas: list[SampleMetadata],
    source_dir: str,
    output_dir: str,
    split: str,
) -> None:
    """
    Copy image and mask files into the dataset directory structure.

    For each metadata:
        src_image = Path(source_dir) / metadata.image_path
        src_mask  = Path(source_dir) / metadata.mask_path
        dst_image = Path(output_dir) / split / "images" / src_image.name
        dst_mask  = Path(output_dir) / split / "masks"  / src_mask.name
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask,  dst_mask)
        metadata.image_path = str(dst_image)
        metadata.mask_path  = str(dst_mask)
    """


def write_manifests(
    train: list[SampleMetadata],
    val: list[SampleMetadata],
    test: list[SampleMetadata],
    output_dir: str,
) -> None:
    """
    Write class_labels.json and metadata.json to output_dir.
    """
```

---

## Fix #4 — Real Test Set Assembly (Mandatory)

The automated splitter creates a synthetic-only test partition as a development proxy. **This is not suitable for reporting final evaluation metrics.** Real-world performance can only be measured on real annotated images.

### Required steps before final evaluation:

1. **Select tiles:** Choose 50-100 gov.il orthophoto tiles from geographic areas NOT used as backgrounds in training (different cities or regions).

2. **Annotate:** Use LabelStudio or CVAT with a polygon annotation tool. Label each of the 4 target classes (crosswalk, tennis court, pool, eucalyptus) at pixel level. Export as grayscale PNG masks with values 0-4.

3. **Replace test split:** Copy the annotated real images and masks into `dataset/test/images/` and `dataset/test/masks/`, replacing the synthetic proxy. Update `metadata.json` with `"source": "real"` for these samples.

4. **Geographic constraint:** The real test tiles must not overlap geographically with the background tiles used in compositing. This is the only control for geographic overfitting.

5. **Synthetic test proxy:** Keep the synthetic test partition as `dataset/test_synthetic/` for development debugging, but do NOT report metrics on it as a proxy for real-world performance.

---

## Implementation Notes

- `stratified_split` must be deterministic given the same `seed` — use `random.seed(seed)` before shuffling
- Class balance in train/val splits benefits from stratification — without it, rare class combinations (e.g., samples with all 4 classes) may end up only in one split
- The `test` split from `stratified_split` is synthetic — its only purpose is internal debugging during development; replace with real annotated tiles before reporting results
- `metadata.json` includes a `source` field for each sample (`'synthetic'` or `'real'`) to make the data provenance machine-readable
