# Design Review — Component 03: Dataset Generator + Palette Mapper

Reviewed against: `design/03_dataset_generator.md` and `design/00_system_overview.md`
Reviewer role: Senior ML Engineer

---

## Findings

### 1. [CRITICAL] `pixel_counts` key type mismatch between dataclass and JSON schema

**Description:**
`SampleMetadata.pixel_counts` is typed `dict[int, int]` in `00_system_overview.md`, but the JSON example in `03_dataset_generator.md` correctly shows string keys: `{"0": 750000, "1": 3000, ...}`. This is not just an example artifact — Python's `json.dumps()` always serializes integer dict keys as strings (JSON spec mandates string keys). On deserialization with `json.loads()`, the keys will be strings, yielding `dict[str, int]`, not `dict[int, int]`. Any downstream code that accesses `metadata.pixel_counts[4]` will raise `KeyError`; it must use `metadata.pixel_counts["4"]`. Nothing in the design documents this discrepancy or specifies a deserialization hook to restore integer keys.

**Suggested fix:**
Either (a) change the `SampleMetadata` dataclass annotation to `dict[str, int]` and update all downstream consumers to use string keys, or (b) add a custom JSON loader that converts keys back to `int` via `{int(k): v for k, v in d.items()}` and document this as a requirement. Option (a) is simpler and more honest about the on-disk representation.

---

### 2. [CRITICAL] `export_mask()` does not validate for unexpected canvas values

**Description:**
`export_mask()` applies `np.where(canvas == 255, 0, canvas)`, which correctly remaps road pixels to background. However, it assumes the canvas contains only values in `{0, 1, 2, 3, 4, 255}`. If `generate_layout()` (Stage 02) introduces any other value due to a bug (e.g., an uninitialized region, a future class addition, or an off-by-one error in a placer), those values pass through silently as invalid class IDs. A mask saved with class ID 100 would corrupt training data with no error or warning raised. The `mask_to_ade20k_rgb()` guard (`mask.max() > 4`) catches this only if palette mapping is called — it does not protect the saved PNG file itself.

**Suggested fix:**
Add an assertion or explicit check in `export_mask()` before returning:
```python
assert set(np.unique(normalized)).issubset({0, 1, 2, 3, 4}), \
    f"Unexpected values in mask after normalization: {np.unique(normalized)}"
```
This makes the function a hard gate rather than a silent pass-through.

---

### 3. [WARNING] `mask_to_canny_edges()` produces non-uniform edge detection across class pairs

**Description:**
The implementation scales class IDs by 50 before applying Canny: `scaled = (mask * 50).astype(np.uint8)`. The Canny gradient magnitude at a boundary between two classes equals approximately `50 × |id_a − id_b|`. With `threshold1=50`, a boundary between adjacent class IDs (e.g., crosswalk/1 and tennis/2, or background/0 and crosswalk/1) produces a gradient of exactly 50 — right at the hysteresis lower threshold and liable to be inconsistently detected or dropped entirely depending on neighboring pixels. Meanwhile, a boundary between background (0) and eucalyptus (4) produces a gradient of 200 and is robustly detected. The edge map quality is thus tied to the arbitrary assignment of class ID integers, not to semantic importance. The ControlNet edge conditioning will be systematically weaker for the rarest and most important boundaries (crosswalk borders are particularly at risk since class ID 1 abuts ID 0 with minimal gradient).

**Suggested fix:**
Apply Canny to the mask's binary boundary image directly:
```python
# Detect any class boundary without ID-magnitude bias
boundary = np.zeros_like(mask, dtype=np.uint8)
for axis in [0, 1]:
    boundary |= (np.diff(mask.astype(np.int16), axis=axis) != 0).view(np.uint8)
edges = (boundary * 255).astype(np.uint8)
```
Or, if Canny smoothing is desired, use a larger ID spacing (e.g., multiply by 63 so all adjacent classes exceed both thresholds) or set `threshold1=1, threshold2=10` to make detection ID-difference-agnostic.

---

### 4. [WARNING] `mask_path` (and other paths) in metadata JSON are relative — portability unspecified

**Description:**
The example JSON shows `"mask_path": "output/layouts/masks/mask_0000.png"`. This is a relative path, but relative to what — the project root, the `pipeline/` directory, or the CWD at run time? The design does not specify. If the metadata JSON is consumed on a different machine, in a different working directory (e.g., a Colab notebook), or after moving the output directory, all paths will be broken. The same ambiguity applies to `background_tile_path` and `image_path` when they are filled in by later stages.

**Suggested fix:**
Specify explicitly in the design whether paths should be stored as absolute paths or as paths relative to `output_dir`. The safest choice for portability is to store them relative to `output_dir` (i.e., `masks/mask_0000.png`) and document that callers must join with `output_dir` to get the full path. Alternatively, store absolute paths and document that the dataset must be re-exported if moved.

---

### 5. [WARNING] Output directory creation not specified in `generate_dataset()`

**Description:**
`generate_dataset()` writes files to `{output_dir}/masks/` but the design contains no instruction to create this directory (or `output_dir` itself) if it does not exist. Calling `cv2.imwrite` or `open()` on a nonexistent directory path raises an OS error. This is a silent omission that will cause the function to fail on first run against a fresh `output_dir`.

**Suggested fix:**
The design should explicitly state: "Create `{output_dir}/masks/` with `os.makedirs(exist_ok=True)` before the loop begins." This also applies to `{output_dir}/` for the JSON files.

---

### 6. [WARNING] No error-handling or checkpoint/resume strategy for partial runs

**Description:**
`generate_dataset()` is designed to produce up to ~1000 samples in a single loop. The docstring describes the happy path only. If the process is interrupted (OOM, keyboard interrupt, disk full) partway through, there is no specification of what state is left on disk, whether partially written files are valid, or how to resume from a checkpoint. Partial masks written by `cv2.imwrite` before a crash can be valid PNGs but their corresponding JSON metadata file may not exist (or vice versa), leaving the output directory in an inconsistent state that is silently consumed by later stages.

**Suggested fix:**
Specify one of: (a) write metadata JSON atomically after the mask PNG (so a missing JSON = incomplete sample, easily detectable), (b) add a `resume=True` parameter that skips already-complete `(mask_NNNN.png, metadata_NNNN.json)` pairs, or (c) document that users must delete the output directory and restart on failure. Any of the three is acceptable; the current silence is not.

---

### 7. [MINOR] `split` field uses `""` as sentinel — implicit invalid state

**Description:**
`SampleMetadata.split` is typed as `str` with documented valid values `'train' | 'val' | 'test'` (per `00_system_overview.md`). The JSON example and the `generate_dataset()` docstring use `""` as a sentinel meaning "not yet assigned." This is implicit: the type annotation does not reflect it, and no code is specified to validate that `split` is non-empty before the sample is consumed downstream. A sample accidentally consumed before Stage 09 assigns the split would silently have `split=""`, polluting any split-stratified analysis.

**Suggested fix:**
Type the field as `Optional[str] = None` (or `str | None = None`) to make the unassigned state explicit and type-checkable. Document that `None` means "not yet assigned." Stages that consume `split` should assert it is not `None`.

---

### 8. [MINOR] PNG save library not specified — potential for subtle mode errors

**Description:**
The design states masks are saved as "single-channel grayscale PNG, values 0-4." It does not specify whether `cv2.imwrite`, `PIL.Image.save`, or another library should be used. For a 2D `(H,W) uint8` numpy array, `cv2.imwrite` writes a valid 8-bit grayscale PNG. `PIL.Image.fromarray` without explicit `mode='L'` may infer mode incorrectly for arrays with small value ranges (e.g., promoting to `'P'` palette mode in some versions). Using the wrong mode or library could produce a 3-channel or palette-mode PNG that consumers interpret incorrectly. While the risk is low for `cv2`, the omission leaves implementation ambiguity.

**Suggested fix:**
Add one line to Implementation Notes: "Save masks using `cv2.imwrite(path, mask)` where `mask` is a `(H,W) uint8` numpy array. Do not use PIL without explicitly specifying `mode='L'`."

---

### 9. [MINOR] Metadata JSON files stored flat alongside `masks/` directory — asymmetric layout

**Description:**
The output tree places mask files under `{output_dir}/masks/mask_{i:04d}.png` but metadata files at `{output_dir}/metadata_{i:04d}.json` (flat, top-level). With n=1000, this puts 1000 JSON files directly in `output_dir` alongside the single `masks/` subdirectory. This is awkward to manage (e.g., `ls output/layouts/` returns 1001 entries), and the `mask_path` field in the JSON uses a relative path that goes _into_ a subdirectory while the JSON itself does not.

**Suggested fix:**
Place metadata files under a parallel subdirectory: `{output_dir}/metadata/metadata_{i:04d}.json`. Update `mask_path` example accordingly to use consistent relative paths.

---

### 10. [MINOR] Canny thresholds in `mask_to_canny_edges()` are hardcoded and not configurable

**Description:**
`threshold1=50` and `threshold2=150` are hardcoded in the pseudocode. These values are coupled to the `* 50` scaling factor (finding 10 above). If the scaling is changed (per finding 3's suggested fix), the thresholds must also change. There is no entry in `PipelineConfig` for these values, preventing tuning without modifying source code.

**Suggested fix:**
Add `canny_threshold1` and `canny_threshold2` (or a single `canny_thresholds: tuple[int, int]`) to an appropriate config section (e.g., a new `ConditioningConfig` dataclass), or at minimum document the coupling between the `* 50` factor and the thresholds explicitly so both are changed together.
