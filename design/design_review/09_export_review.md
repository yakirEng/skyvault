# Design Review — 09_export.md

Reviewer: senior ML engineer
Date: 2026-03-06
Files reviewed: `design/00_system_overview.md`, `design/09_export.md`

---

## Findings

### 1. [CRITICAL] Augmented variants of the same source sample are split independently — data leakage

**Description:** `stratified_split` receives `list[SampleMetadata]` from `08_augmentation`, which produces ×4 augmented variants per original composite. These 4 variants share the same layout, seed, background tile, and object positions — they differ only by geometric/colour transform. If they are split independently by their `classes_present` key, some variants end up in train while others end up in val or test. A val or test image that is a flipped/colour-jittered version of a training image is a direct data leak and will artificially inflate reported val/test metrics.

**Suggested fix:** Stratified split must operate on the **pre-augmentation** sample list (one entry per original composite). Each original is assigned a split, and all its augmented children inherit that assignment. Concretely: group the augmented list by `seed` (or a `parent_id` field added in stage 08), assign the group's split, then expand. Document this grouping requirement explicitly in the `stratified_split` docstring.

---

### 2. [CRITICAL] `test_synthetic/` directory is referenced but never created by any function

**Description:** Fix #4 Step 5 states "Keep the synthetic test partition as `dataset/test_synthetic/`". However:
- The dataset directory structure diagram only shows `dataset/test/`.
- No call to `export_split(..., split='test_synthetic')` is defined.
- `metadata.json` has no provision for a `test_synthetic` split value.
- `write_manifests` signature accepts only `train`, `val`, `test`.

The synthetic proxy test set will be silently discarded or overwritten rather than preserved for development debugging.

**Suggested fix:** Add `'test_synthetic'` as a valid `split` string throughout. Add a fourth return value to `stratified_split` or a separate parameter. Update the directory structure diagram, `metadata.json` schema, and `write_manifests` signature. Alternatively, document that a separate `export_synthetic_test()` helper writes the synthetic test partition before real tiles replace `dataset/test/`.

---

### 3. [WARNING] Rare class-combination groups may be too small to split proportionally

**Description:** With 4 object classes there are up to 16 possible non-background class combinations. Rare combinations (e.g., samples containing all four classes simultaneously) may produce groups with 1 or 2 members. Proportional splitting of a 2-member group at 70/15/15 gives 1.4/0.3/0.3 — the rounding is undefined, and at minimum one split receives 0 members. A group of 1 cannot appear in all three splits at all. The design says nothing about this case.

**Suggested fix:** Add a fallback rule: if a group has fewer than `ceil(1 / min_ratio)` members (i.e., fewer than `ceil(1/0.15) = 7` for default ratios), assign all members to train rather than trying to split. Log a warning listing which combinations fell back to train-only. Document the threshold as a parameter.

---

### 4. [WARNING] LabelStudio/CVAT annotation export format is not specified precisely enough to produce correct masks

**Description:** Fix #4 Step 2 says to "Export as grayscale PNG masks with values 0-4." Neither LabelStudio nor CVAT exports in this format by default:
- LabelStudio's brush annotation export produces RLE JSON or RGB-coloured PNGs, not integer-class grayscale.
- CVAT's segmentation mask export uses a colour-coded PNG palette, not raw class-integer values.

An annotator following this guide without additional tool-specific instructions will almost certainly produce the wrong output format.

**Suggested fix:** Add a "Annotation Export Configuration" sub-section specifying per-tool settings. For LabelStudio: use the "Brush" label type, export via the "PNG mask" converter, and apply a custom converter that maps label colours to integer values 0-4. For CVAT: export in "Segmentation mask" format and document the `labelmap.txt` mapping that produces single-channel uint8 PNGs with the correct class values. Include a one-line validation command (e.g., `python -c "import numpy as np; from PIL import Image; m=np.array(Image.open('mask.png')); assert m.dtype==np.uint8 and m.max()<=4"`) that annotators can run to verify their export.

---

### 5. [WARNING] Geographic exclusion of real test tiles is unverifiable because background tile metadata is not saved

**Description:** Fix #4 Step 4 states "The real test tiles must not overlap geographically with the background tiles used in compositing." However, component 04 (`super_resolution`) was designed to save only `bg_{i:04d}.png` files with no sidecar geographic metadata (source GeoTIFF path, crop bounding box in EPSG:2039). Without that information, there is no programmatic way to verify geographic separation. This makes the geographic exclusion requirement aspirational rather than enforceable.

**Suggested fix:** This finding depends on a gap identified in `04_super_resolution_review.md` (finding #8). The fix there (saving per-tile sidecar JSON with crop bounding box) is a prerequisite. Once bounding boxes are recorded, add a `check_geographic_overlap(test_tile_bbox, background_bboxes, min_separation_m=5000)` utility here and call it as part of the real test set assembly checklist.

---

### 6. [WARNING] `image_path` and `mask_path` in `metadata.json` will be absolute paths, breaking portability

**Description:** `export_split` computes `dst_image = Path(output_dir) / split / "images" / src_image.name` and then sets `metadata.image_path = str(dst_image)`. If `output_dir` is an absolute path (which it will be in any realistic invocation), `metadata.json` is written with machine-specific absolute paths. When the dataset is moved to a training machine (Colab, RunPod, a different local path), all paths in `metadata.json` are invalid and the DataLoader cannot find files.

**Suggested fix:** Store paths in `metadata.json` relative to the dataset root (e.g., `"train/images/augmented_0000_0.png"`). In `write_manifests`, compute `Path(image_path).relative_to(Path(output_dir))` before serialisation. Document that paths in `metadata.json` are always relative to the directory containing `metadata.json`.

---

### 7. [WARNING] No `dataset.yaml` or per-split file list — output is not consumable by any standard training framework

**Description:** The output structure (`class_labels.json` + `metadata.json`) is custom and not natively supported by any major segmentation training framework:
- MMSegmentation requires a `dataset_info` config + per-split `.txt` file listing image paths.
- HuggingFace `datasets` requires a `dataset_dict.json` or Arrow files.
- PyTorch Lightning / custom DataLoaders typically expect `train.txt` / `val.txt` file lists.

A researcher receiving this dataset cannot plug it into any framework without writing custom glue code. `class_labels.json` maps IDs to names but is not in any standard schema.

**Suggested fix:** Have `write_manifests` additionally produce:
1. `train.txt`, `val.txt`, `test.txt` — one `images/<filename>.png` path per line (relative), matching the common paired-path convention.
2. A `dataset.yaml` in the format expected by the target training framework (document which framework is targeted).

---

### 8. [WARNING] Filename collision: augmented variants may silently overwrite each other during copy

**Description:** `export_split` copies files using `src_image.name` as the destination filename (`dst_image = Path(output_dir) / split / "images" / src_image.name`). If the augmentation stage (08) names variants with a suffix like `augmented_0000_0.png`, `augmented_0000_1.png`, etc., this is safe. But the naming convention for augmented files is not specified in this document (it is defined in 08 design, which is not reviewed here). If two augmented variants share the same filename (e.g., both derived from `composite_0000.png` with augmentation indices not included in the name), `shutil.copy2` silently overwrites the first with the second — a data-loss bug with no error or warning.

**Suggested fix:** Add a pre-copy uniqueness check: assert that all `src_image.name` values in a given split are unique before any copies are made. Raise `ValueError` listing any duplicate filenames. Document the required augmented file naming convention (`{parent_id}_{aug_index}.png`) as an explicit contract between stage 08 and stage 09.

---

### 9. [MINOR] `pixel_counts` integer keys become string keys after JSON round-trip, causing KeyError in DataLoader

**Description:** `SampleMetadata.pixel_counts` is typed as `dict[int, int]`. Python's `json.dumps` serialises integer dictionary keys as JSON strings (`"0"`, `"1"`, etc.), which is correct JSON. But `json.loads` returns `{"0": ..., "1": ...}` with string keys. If a training DataLoader parses `metadata.json` and then accesses `pixel_counts[class_id]` where `class_id` is an `int`, it gets a `KeyError`. The example in the design already shows string keys (`"0"`, `"1"`, `"2"`), which is the correct JSON representation — but the Python type annotation says `dict[int, int]`, creating a mismatch that will confuse implementers.

**Suggested fix:** Either change the `SampleMetadata` type annotation to `dict[str, int]` (matching JSON reality), or add a note in `write_manifests` that keys must be serialised as strings and parsed as strings. Add a corresponding note in the `metadata.json` schema documentation.

---

### 10. [MINOR] `random.seed(seed)` in Implementation Notes is inconsistent with the pipeline's numpy RNG pattern

**Description:** The Implementation Notes say "use `random.seed(seed)` before shuffling". The rest of the pipeline exclusively uses `np.random.default_rng(seed)` (a modern, seedable numpy Generator). Mixing Python's `random` module with numpy's RNG creates two independent random streams that are separately seeded. If an implementer shuffles a numpy array using `random.shuffle` (which works but is slower and less idiomatic for arrays) and elsewhere uses numpy operations, the overall pipeline reproducibility is harder to reason about.

**Suggested fix:** Change to `rng = np.random.default_rng(seed); rng.shuffle(group)` consistently with the rest of the pipeline. Update the Implementation Notes accordingly.

---

### 11. [MINOR] No minimum per-class sample count specified for the real test set

**Description:** "50-100 gov.il orthophoto tiles" is stated without justification in terms of class coverage. Given that crosswalks occupy ~0.3% of pixels per the class balance table, a random 51×51 m tile has only a ~15% chance of containing a crosswalk at all (rough estimate). With 50 tiles, the expected number of crosswalk-positive test images may be fewer than 10, which is insufficient for statistically meaningful evaluation of the rarest class.

**Suggested fix:** Specify minimum per-class positive sample counts (e.g., "at least 20 images containing at least one instance of each class"). Require annotators to deliberately sample tiles known to contain target objects (using aerial imagery viewers to pre-screen) rather than selecting randomly.

---

### 12. [MINOR] Background class always present in stratification key dilutes its discriminative value

**Description:** `classes_present` is documented as a "subset of [0,1,2,3,4]" where 0 is background. Background pixels exist in virtually every sample (background typically covers ~75% of each canvas). Including class 0 in the `frozenset` stratification key means almost every sample has `0` in its key, adding no discriminative information. The stratification effectively groups by object class combinations `{1,2,3,4} ∩ classes_present`, but the docstring does not make this clear.

**Suggested fix:** Document explicitly that class 0 (background) is expected in nearly all keys and does not contribute to stratification diversity. Optionally, derive the stratification key as `frozenset(c for c in metadata.classes_present if c != 0)` and note this in the docstring, so the grouping logic is transparent and intentional.

---

### 13. [MINOR] End-to-End Run Order does not show the correct `export_split` call pattern

**Description:** The run order in `00_system_overview.md` lists `stratified_split, export_split, write_manifests` in a single pseudocode block with `# ... orchestration script`. `export_split` must be called **three times** (once per split), and `write_manifests` must be called **after all three** copy operations complete. This is not shown. A developer following the run order will not know to call `export_split` three times.

**Suggested fix:** Expand the orchestration pseudocode in `09_export.md` to show the correct call sequence explicitly:
```python
train, val, test = stratified_split(metadatas, seed=42)
export_split(train, source_dir, 'dataset/', 'train')
export_split(val,   source_dir, 'dataset/', 'val')
export_split(test,  source_dir, 'dataset/', 'test')
write_manifests(train, val, test, 'dataset/')
```
