# Integration Review — End-to-End Interface Audit

Reviewed: all files in `design/` (00–09)
Reviewer role: Senior ML Engineer
Scope: inter-component interface mismatches only (not intra-component bugs)

---

## Findings

### 1. [CRITICAL] Components 03 → 05/06: `build_conditions()` consumes `normalized_mask` internally but never returns it

**Components affected:** 03 (Dataset Generator) → 05 (ControlNet Generation) → 06 (Compositor)

`build_conditions()` (05) computes `normalized_mask = export_mask(layout_result.canvas)` as its first step, uses it to produce `seg_condition` and `edge_condition`, then returns only `tuple[seg_condition, edge_condition]` — discarding `normalized_mask`. However, `extract_object_patches()` (05) requires `normalized_mask` as an explicit parameter, and `composite()` (06) also requires `normalized_mask` as an explicit parameter. The orchestrating `run_generation()` docstring says only "build_conditions → generate_sdxl_scene → extract_object_patches" without specifying where `normalized_mask` for `extract_object_patches()` comes from. The orchestration script would have to call `export_mask(layout_result.canvas)` a second time to produce it, but this is entirely undocumented. Any implementation that passes `None` or re-reads the mask from disk risks subtle divergence from what `build_conditions()` used internally.

**Suggested fix:** Change `build_conditions()` return type to `tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(seg_condition, edge_condition, normalized_mask)`, and update the `run_generation()` docstring to thread `normalized_mask` through to both `extract_object_patches()` and the compositor.

---

### 2. [CRITICAL] Components 03 → 08: Metadata JSON files written to Stage 03 output directory, never moved to compositor output directory that Stage 08 reads from

**Components affected:** 03 (Dataset Generator) → 08 (Augmentation)

Stage 03 `generate_dataset()` writes `{output_dir}/metadata_{i:04d}.json` where `output_dir` is the layout output root (e.g., `output/layouts/`). Stage 08 `augment_dataset()` reads `{input_dir}/metadata_{i:04d}.json` where `input_dir` is the compositor output directory containing `images/sample_{i:04d}.png` — a different location entirely. No stage in the pipeline copies or moves the metadata JSONs from the Stage 03 output directory to the compositor output directory. Stage 08 will find no metadata files and either crash or silently produce metadata-less output.

**Suggested fix:** Either (a) Stage 03 should write metadata JSONs to the same directory where the compositor will eventually write images (requires knowing that path at Stage 03 time), or (b) Stage 08 should accept a separate `metadata_dir: str` parameter distinct from `input_dir`, or (c) the orchestration script (with its own design document) must explicitly copy metadata JSONs to the compositor output dir before Stage 08 runs — document this as a required step.

---

### 3. [CRITICAL] Components 03 → 09: `SampleMetadata.mask_path` points to the layout mask throughout the pipeline — never updated to the final composited or augmented mask path

**Components affected:** 03 (Dataset Generator) → 06 (Compositor) → 08 (Augmentation) → 09 (Export)

Stage 03 sets `mask_path = "output/layouts/masks/mask_{i:04d}.png"` (the pre-compositing layout mask). No subsequent component design (06, 07, 08) specifies updating `mask_path` to reflect the composited mask (`images/sample_{i:04d}.png` output from Stage 06) or the augmented mask (`augmented_{i:04d}_{j}.png` from Stage 08). Stage 09 `export_split()` constructs `src_mask = Path(source_dir) / metadata.mask_path` — this path resolves to the original layout mask, not the final augmented mask that should be exported. The `metadata.json` example in 09 correctly shows `"mask_path": "dataset/train/masks/augmented_0000_0.png"` but no component is responsible for setting it to that value before Stage 09 runs.

**Suggested fix:** Explicitly specify in the design of each stage (06, 08) that `SampleMetadata.mask_path` is updated to the current stage's output mask path before the metadata object is passed forward. Alternatively, define a metadata-update step in the orchestration design for each stage that writes new files.

---

### 4. [CRITICAL] Components 03 → 09: `SampleMetadata.image_path` is never populated by any specified component

**Components affected:** 03 (Dataset Generator) → 06 (Compositor) → 08 (Augmentation) → 09 (Export)

Stage 03 sets `image_path = ""` with the note "filled in by later stages." No later stage design document specifies which component fills it in or when. Stage 06 says "Saved to disk by the orchestration script" but does not mention updating `image_path` in metadata. Stage 08 returns updated `SampleMetadata` but its docstring mentions only `sample_id`, `split`, and `source` fields — `image_path` is not mentioned. Stage 09 `export_split()` uses `src_image = Path(source_dir) / metadata.image_path` — if `image_path` is still `""`, this constructs `Path(source_dir) / ""` which resolves to `source_dir` itself (a directory, not a file), causing a copy failure or copying the wrong thing silently.

**Suggested fix:** Stage 06 orchestration (or Stage 08) must be specified to set `metadata.image_path` to the composited/augmented image path. Document this explicitly in the Stage 06 and Stage 08 design files.

---

### 5. [WARNING] Components 07 → 08: No mechanism specified for writing quality-filtered composites to disk between Stage 07 and Stage 08

**Components affected:** 07 (Quality Gate) → 08 (Augmentation)

Stage 07 `filter_dataset()` returns `(list[CompositeResult], list[SampleMetadata])` — in-memory Python objects. Stage 08 `augment_dataset()` takes `input_dir: str` and reads files from disk using the pattern `{input_dir}/images/sample_{i:04d}.png`. No component or design document specifies the step of persisting Stage 07's filtered output to disk in a form Stage 08 can consume. The entire 07→08 data handoff is unspecified. Implementations will need to invent their own intermediate save step.

**Suggested fix:** Either (a) add a `save_filtered()` function to Stage 07 that writes passing composites to a clean output directory (with contiguous re-numbered filenames), making Stage 08's `input_dir` the target; or (b) add a `list[SampleMetadata]` parameter to `augment_dataset()` so it can filter which files to process without requiring files be saved separately.

---

### 6. [WARNING] Components 07 → 08: Rejected sample indices leave gaps in filename sequence; Stage 08 has no skip mechanism

**Components affected:** 07 (Quality Gate) → 08 (Augmentation)

Stage 06 saves composites as `sample_{i:04d}.png` for `i = 0..N-1`. Stage 07 rejects some subset. If composites are saved before quality gate, the rejected files remain on disk (e.g., `sample_0004.png` may be rejected while `sample_0003.png` and `sample_0005.png` pass). Stage 08 reads by the fixed pattern `sample_{i:04d}.png` and has no parameter to pass a list of valid indices or skip rejected ones. It will either process all N files (including rejected) or fail on gaps in the sequence. If composites are saved only after quality gate (to avoid this), the numbering is non-contiguous and Stage 08's pattern-based iteration breaks differently.

**Suggested fix:** Either (a) Stage 07 should produce a "passing indices" list that Stage 08 accepts as a parameter, or (b) Stage 07's save step (per finding 5) renumbers files contiguously so Stage 08 can iterate `0..n_passed-1` without gaps.

---

### 7. [WARNING] Components 04 → 03/06: `SampleMetadata.background_tile_path` is never assigned by any specified component

**Components affected:** 04 (Super-Resolution) → 03 (Dataset Generator) → 06 (Compositor) → 09 (Export)

Stage 03 initializes `background_tile_path = ""`. Stage 04 `process_geotiff_tiles()` returns `list[str]` of saved background PNG paths. Stage 06 `composite()` takes `background: np.ndarray` — the numpy array, not a path — so path tracking is left to the caller. No stage design specifies: which component picks a background path from Stage 04's list, assigns it to a specific sample's `SampleMetadata.background_tile_path`, and passes both the array and path through. Stage 09's `metadata.json` example shows `"background_tile_path": "backgrounds/bg_0012.png"` — a non-empty value — but there is no code path in any stage to produce it.

**Suggested fix:** Specify explicitly in the Stage 06 (or orchestration) design that when a background tile is selected for a sample, its path is recorded in `metadata.background_tile_path` before `composite()` is called.

---

### 8. [WARNING] Components 05 → 06: `SDXLResult.edge_condition` is stored single-channel but the diffusers pipeline requires 3-channel; shape mismatch if the struct field is reused as pipeline input

**Components affected:** 05 (ControlNet Generation) → 06 (Compositor) / orchestration

`SDXLResult` stores `edge_condition: np.ndarray (1024,1024) uint8` — a single-channel array. The Implementation Notes in 05 state: "edge_condition must be passed as a 3-channel image to diffusers: `np.stack([edges]*3, axis=2)`." This conversion is documented only as a note, not in the `generate_sdxl_scene()` function signature or in `SDXLResult`'s field definition. Any caller that retrieves `sdxl_result.edge_condition` from a stored `SDXLResult` and passes it to `generate_sdxl_scene()` (e.g., for debugging, replay, or re-inference) will pass a `(H,W)` array where a `(H,W,3)` array is required, causing a runtime error inside the diffusers pipeline. The struct definition and function signature are inconsistent about what shape this field should have.

**Suggested fix:** Either store `edge_condition` in `SDXLResult` as `(1024,1024,3)` (the 3-channel form ready for the pipeline), or annotate the field explicitly as `(1024,1024)` and document in `generate_sdxl_scene()`'s parameter table that callers must expand it before passing.

---

### 9. [WARNING] Components 06 → 07: `SampleMetadata.classes_present` is stale at quality gate time — not updated to reflect what actually composited successfully

**Components affected:** 06 (Compositor) → 07 (Quality Gate)

`SampleMetadata.classes_present` is set by the layout engine (Stage 02/03) based on which classes were placed in the canvas. Stage 06 `blend_patch()` may skip patches that are within 3px of the image edge (explicitly documented: "Skip (log warning) if bbox is within 3px of image edge"). If an object is skipped, its class pixels never appear in the composited `image` or `output_mask`. Stage 07 `filter_sample()` checks `metadata.classes_present` to determine which classes to verify coverage for — using the layout-time list, not the post-compositing reality. A sample where an entire tennis court was skipped due to the edge rule will be rejected by the coverage check for class 2, even though the skip was intentional and logged. Conversely, `classes_present` may also contain class 0 (background) which has no entry in `MIN_PIXELS_PER_CLASS`, causing a `KeyError` if the coverage check iterates over all entries in `classes_present` without filtering out 0.

**Suggested fix:** Stage 06 `composite()` should return (or update in metadata) a revised `classes_present` list reflecting only classes that were actually blended. Stage 07 should document that it filters class 0 from `classes_present` before checking `MIN_PIXELS_PER_CLASS`.

---

### 10. [WARNING] Components 08 → 09: `augment_dataset()` does not specify which `SampleMetadata` fields it updates — Stage 09 depends on `seed`, `mask_path`, `image_path`, and `background_tile_path` being correct

**Components affected:** 08 (Augmentation) → 09 (Export)

Stage 08 `augment_dataset()` docstring states only: "sample_id incremented sequentially across all augmented files" and "split field still empty string." It says nothing about what `seed`, `mask_path`, `image_path`, or `background_tile_path` contain in the returned metadata. Stage 09 `export_split()` uses `metadata.image_path` and `metadata.mask_path` to locate source files for copying. `write_manifests()` writes all `SampleMetadata` fields to `metadata.json`. If Stage 08 does not update `image_path` and `mask_path` to the augmented file paths, Stage 09 copies from stale or empty paths (see also finding 3). Additionally, `seed` for augmented samples is ambiguous: should it be the base layout seed (for tracing back to the original layout), the augmentation seed (`seed * 100 + i`), or both? The design is silent.

**Suggested fix:** Stage 08's design should explicitly list every `SampleMetadata` field and its value in the returned augmented metadata: `sample_id` (new), `seed` (inherited from base), `image_path` (set to augmented image path), `mask_path` (set to augmented mask path), `background_tile_path` (inherited), `classes_present` (inherited), `pixel_counts` (inherited — note: stale after augmentation changes spatial layout), `split` (`""`), `source` (`"synthetic"`).

---

### 11. [MINOR] All components: `pixel_counts: dict[int, int]` is universally typed as int-keyed in Python but all JSON examples show string keys — no deserialization hook specified anywhere

**Components affected:** 03 (Dataset Generator) ↔ 08 (Augmentation) ↔ 09 (Export)

`LayoutResult.pixel_counts` (02), `SampleMetadata.pixel_counts` (00/03), `QualityResult.class_coverage` (00/07) are all typed `dict[int, int]`. JSON mandates string keys; `json.dumps({0: 750000})` produces `{"0": 750000}` and `json.loads(...)` produces `{"0": 750000}` (string-keyed). All three JSON examples (03, 09) correctly show string keys. But no design file specifies a deserialization hook. Any code loading a metadata JSON and then accessing `metadata.pixel_counts[0]` or `metadata.pixel_counts[4]` will raise `KeyError`. This is consistent across all files in the same way — all files have the same mismatch — but it will fail at runtime in every component that reads metadata from disk.

**Suggested fix:** Either change all Python type annotations to `dict[str, int]` (honest about the JSON representation) or specify a shared `load_metadata(path) -> SampleMetadata` utility in Stage 03 that applies `{int(k): v for k, v in d["pixel_counts"].items()}` on deserialization, and mandate its use across all stages.

---

### 12. [MINOR] Components 01 → all: YAML key `canvas_size` does not match Python field `canvas_px`

**Components affected:** 01 (Config) → all consumers

`pipeline_config.yaml` uses `resolution.canvas_size: 1024` but `ResolutionConfig` exposes the field as `canvas_px: int`. `load_config()` maps between them silently. No other design file documents this naming divergence. Developers implementing consumers (SR, compositor, augmentor) and searching the YAML for a key matching the Python field name `canvas_px` will not find it, and may mistakenly assume it is a derived value (like the `_px` fields from meters) rather than a directly read YAML value. The comment in 01 "canvas_px is read directly from YAML (not derived)" partially mitigates this but is only in 01.

**Suggested fix:** Rename the YAML key from `canvas_size` to `canvas_px` to match the Python field name, or add an explicit note in every component design that accesses `cfg.resolution.canvas_px` that this value comes from `YAML:resolution.canvas_size`.

---

### 13. [MINOR] Components 08 → 09: `export_split()` path join is broken if `metadata.image_path` is an absolute path

**Components affected:** 08 (Augmentation) → 09 (Export)

Stage 09 `export_split()` constructs `src_image = Path(source_dir) / metadata.image_path`. On POSIX systems, `Path("/a/b") / "/c/d"` returns `Path("/c/d")` (absolute path overrides). On Windows, joining two absolute paths raises `ValueError`. Stage 08's design does not specify whether `image_path` and `mask_path` in returned metadata should be absolute or relative. If Stage 08 stores absolute paths (natural when using `os.path.abspath`), Stage 09's path join produces either the wrong result (POSIX) or a crash (Windows). This is a cross-platform portability defect at the 08→09 boundary.

**Suggested fix:** Specify in Stage 08's design that `image_path` and `mask_path` in returned metadata must be relative to `output_dir`. Specify in Stage 09 that it joins relative paths only and should assert `not metadata.image_path.startswith("/")` (or use `os.path.relpath` when setting the field in Stage 08).
