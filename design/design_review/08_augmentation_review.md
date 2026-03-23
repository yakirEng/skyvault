# Design Review — 08_augmentation.md

Reviewer: Senior ML Engineer
Date: 2026-03-06

---

## Findings

---

### 1. [CRITICAL] `additional_targets={"mask": "mask"}` re-registers a built-in default key — raises `ValueError` on `A.Compose` initialization in albumentations >= 1.3

**Description:** In albumentations, `"mask"` is a reserved default target key. `A.Compose` raises `ValueError: Key mask is not allowed in additional_targets` (or equivalent) when a built-in key is passed via `additional_targets`. The pipeline-level constant `AUGMENTATION_PIPELINE` is constructed at module import time, so this error fires before any augmentation is attempted — the entire augmentor module fails to import.

The sync between image and mask spatial transforms does NOT require `additional_targets`. Passing `mask=mask` in the call to `AUGMENTATION_PIPELINE(image=image, mask=mask)` is sufficient: `"mask"` is already a first-class default target in `A.Compose`, spatial transforms are applied to it identically, and color/noise transforms skip it automatically.

**Suggested fix:** Remove `additional_targets={"mask": "mask"}` from `A.Compose`. Call the pipeline as `AUGMENTATION_PIPELINE(image=image, mask=mask)` without any `additional_targets`. Update the implementation note that describes this line as "critical" — it is incorrect and actively harmful.

---

### 2. [CRITICAL] Implementation note's claim about `additional_targets` purpose is factually wrong — perpetuates a misunderstanding that could cause future mask corruption

**Description:** The implementation note states: "`additional_targets={"mask": "mask"}` is critical — without it, color transforms would be applied to the mask (corrupting class IDs)". This is false. Albumentations excludes pixel-level (color/noise) transforms from `mask`-typed targets by design, regardless of whether `additional_targets` is used. The `"mask"` key in the standard call `transform(image=..., mask=...)` is already treated as a mask and skipped by `RandomBrightnessContrast`, `HueSaturationValue`, `GaussNoise`, and `ImageCompression`. A future implementer reading this note may attempt workarounds based on incorrect assumptions.

**Suggested fix:** Replace the note with the accurate explanation: "albumentations automatically skips pixel-level transforms for any target registered as type `mask`. The default `mask=` keyword already provides this guarantee. No `additional_targets` entry is needed for the single segmentation mask."

---

### 3. [WARNING] Seeding mechanism is unspecified — "Set albumentations random seed to `seed * 100 + i`" does not say how

**Description:** The `augment_sample()` docstring states "Set albumentations random seed to `seed * 100 + i` for determinism" but does not specify the mechanism. Options available in albumentations 1.4 include: (a) calling `numpy.random.seed()` before each pipeline call (global state — not thread-safe), (b) using `A.ReplayCompose` to record and replay a transform, or (c) passing a `random_state` via internal APIs. Without pinning a mechanism, two implementers will produce different "deterministic" outputs, breaking reproducibility across machines.

**Suggested fix:** Specify exactly: set `numpy.random.seed(seed * 100 + i)` immediately before each `AUGMENTATION_PIPELINE(...)` call, and document that `augment_dataset()` must not be parallelized (or that each worker must receive its own seeded pipeline instance) for this to be thread-safe.

---

### 4. [WARNING] `augment_dataset()` does not specify writing metadata JSON files to disk — Stage 09 will not find them

**Description:** The output spec for `augment_dataset()` lists only image and mask PNGs written to `{output_dir}/images/` and `{output_dir}/masks/`. The function returns `list[SampleMetadata]` in memory but no corresponding `metadata_{i:04d}.json` files are written. The system overview's end-to-end run script shows Stage 09 (`export_split`) reading metadata from disk alongside images and masks. If augmented samples have no on-disk metadata, the export/split stage cannot correctly attribute `seed`, `classes_present`, `placement_log`, or `background_tile_path` to augmented samples.

**Suggested fix:** Add to the output spec: write `{output_dir}/metadata/augmented_{i:04d}_{j}.json` for each augmented sample, serialized from the `SampleMetadata` produced in memory. Alternatively, define a manifest file (e.g., a single `augmented_manifest.json`) and document which component is responsible for reading it in Stage 09.

---

### 5. [WARNING] `GaussNoise(var_limit=(25, 75))` variance semantics changed in albumentations 1.4+ — noise strength may differ from intent

**Description:** In albumentations < 1.3, `GaussNoise(var_limit=(lo, hi))` sampled variance from `[lo, hi]` where values were in squared pixel units (0–255²), producing very mild noise. Starting in albumentations 1.3–1.4, the parameter was reinterpreted: `var_limit` is now sampled as a fraction of the per-channel standard deviation or as a direct pixel-unit variance depending on a `per_channel` flag. With the pinned constraint `albumentations>=1.4`, the effective noise level could be substantially different from what the author intended — either imperceptibly weak or unrealistically strong for aerial imagery.

**Suggested fix:** Verify the actual pixel-level noise output for `var_limit=(25, 75)` against albumentations 1.4 by computing `std = sqrt(var)` on a sample output. If the result is too strong (std > 10 on a 0–255 scale is typically visually harsh for aerial orthophotos), reduce to `var_limit=(5, 25)`. Pin the interpretation in a code comment.

---

### 6. [WARNING] Nearest-neighbor enforcement for mask is implicit, not verified — no assertion protects against future pipeline additions

**Description:** The design correctly restricts rotations to 90° multiples to avoid interpolation artifacts, and relies on albumentations' built-in NN behavior for mask targets. However, there is no guard against a future developer adding a spatial transform that involves arbitrary resampling (e.g., `A.ShiftScaleRotate`, `A.ElasticTransform`, `A.Perspective`) to `AUGMENTATION_PIPELINE`. Such a transform would silently corrupt mask class IDs at object boundaries — pixels near class borders would receive interpolated values (e.g., 1.5, 2.7) that get rounded to unexpected class IDs.

**Suggested fix:** Add a post-augmentation assertion in `augment_sample()`: `assert set(np.unique(aug_mask)).issubset({0, 1, 2, 3, 4}), f"Mask corruption detected: unexpected values {np.unique(aug_mask)}"`. This catches interpolation bugs immediately. Additionally, add a comment in `AUGMENTATION_PIPELINE` prohibiting the addition of any transform with `interpolation` parameter.

---

### 7. [MINOR] `augment_sample(n=...)` vs `augment_dataset(n_per_sample=...)` parameter name inconsistency

**Description:** `augment_sample()` uses the parameter name `n` while `augment_dataset()` uses `n_per_sample` for the same concept. When `augment_dataset()` internally calls `augment_sample()`, it must pass `n=n_per_sample` positionally or by keyword. An implementer who calls `augment_dataset(n=4)` (using the `augment_sample` naming) will hit a `TypeError` since `augment_dataset` declares `n_per_sample`.

**Suggested fix:** Standardize to one name. `n_per_sample` is more descriptive and should be the canonical name in both function signatures.

---

### 8. [MINOR] `AUGMENTATION_PIPELINE` as a module-level singleton is thread-unsafe under parallelism

**Description:** `AUGMENTATION_PIPELINE` is constructed once at module import time with shared internal random state. If `augment_dataset()` is ever parallelized using `multiprocessing.Pool` or `concurrent.futures.ThreadPoolExecutor`, multiple workers will share and mutate this object's random state concurrently, producing non-deterministic and non-reproducible outputs even when per-sample seeds are set.

**Suggested fix:** Either (a) document explicitly that `augment_dataset()` must run single-threaded, or (b) construct a fresh `A.Compose(...)` instance inside each worker call rather than sharing the module-level constant.

---

### 9. [MINOR] No per-class oversampling for crosswalks — responsibility is unassigned across components

**Description:** `00_system_overview.md` states "Crosswalk patches should be oversampled ×3 during training." Component 08 does not implement any class-conditional oversampling (e.g., augmenting crosswalk-containing samples with a higher `n` value, or duplicating them before Stage 09 splitting). No other component claims this responsibility either. The ×3 oversampling requirement will silently fall through to the training script without any pipeline enforcement.

**Suggested fix:** Either (a) add an `oversample_rare_classes` parameter to `augment_dataset()` that generates `n * 3` variants for samples where `classes_present` includes class 1 (crosswalk), or (b) add an explicit note in this design doc and in `09_export.md` assigning oversampling to the training data loader, not the pipeline. Ambiguous ownership across components is a maintenance risk.

---

### 10. [MINOR] Output canvas size after augmentation is not explicitly guaranteed to remain 1024×1024

**Description:** The output spec states `augmented_image: (1024,1024,3) uint8` and `augmented_mask: (1024,1024) uint8`. The transforms in the current pipeline (flips, 90° rotations, color adjustments) all preserve dimensions. However, the design does not include a `PadIfNeeded` or `CenterCrop` safeguard, nor any assertion. If `ImageCompression` or `GaussNoise` ever behaves unexpectedly (e.g., a version that pads internally), or if the pipeline is extended, a silent shape mismatch would propagate downstream to the compositor or export stage.

**Suggested fix:** Add a post-transform assertion in `augment_sample()`: `assert aug_image.shape == (1024, 1024, 3) and aug_mask.shape == (1024, 1024)`. This costs negligible runtime and catches dimension bugs immediately.
