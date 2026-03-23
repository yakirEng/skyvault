# Design Review — 00_system_overview.md

Reviewer: senior ML engineer
Date: 2026-03-06
File reviewed: `design/00_system_overview.md`

---

## Findings

### 1. [CRITICAL] Text prompt for SDXL is absent from the entire data flow and all shared dataclasses

**Description:** SDXL is a text-conditioned diffusion model — it requires a text prompt at inference time. The data flow shows `build_conditions()` producing `seg_condition + edge_condition`, then `generate_sdxl_scene()` receiving only those two tensors and producing a `scene`. No text prompt is defined, stored, or passed anywhere in the system. There is no `prompt: str` field in `LayoutResult`, `SDXLResult`, or `SampleMetadata`. There is no mention of how the prompt is derived (e.g., from `classes_present`), what its format is, or whether negative prompts are used. Without a prompt, SDXL cannot be called.

**Suggested fix:** Define a `PromptBuilder` utility (in component 05 or a new component 03b) that maps `classes_present` → `(positive_prompt, negative_prompt)`. Add `prompt: str` and `negative_prompt: str` to `SDXLResult`. Add a "Prompt generation" node to the data flow diagram between `export_mask` and `build_conditions`.

---

### 2. [CRITICAL] `SampleMetadata.split` and `SampleMetadata.image_path` / `mask_path` have no default values but are only populated at stage 09

**Description:** `SampleMetadata` is constructed at stage 03 (`export_mask`) but several of its fields are documented as assigned later: `split` is "assigned in Stage 09", and `image_path` / `mask_path` change value when `export_split` copies files in stage 09. The Python dataclass `split: str` (no default) means construction at stage 03 will raise a `TypeError` unless a sentinel value is supplied. There is no documented sentinel (e.g., `split: str = ''` or `split: str | None = None`). The same applies to `background_tile_path`, which is set when a background tile is selected — a step that is entirely missing from the data flow (see finding #5).

**Suggested fix:** Either (a) change the type annotation to `split: str = ''` and `background_tile_path: str = ''` with a note that they are populated progressively, or (b) split the dataclass into `LayoutMetadata` (fields known at stage 03) and `SampleMetadata` (all fields, constructed at stage 06 when background is assigned). Document the lifecycle of each field explicitly.

---

### 3. [CRITICAL] Real-ESRGAN is GPU-dependent but the run order labels step 4 as "CPU, ~30min"

**Description:** Real-ESRGAN's neural network inference is GPU-based. Running `RealESRGANer` on CPU is possible but requires 60–300× longer per inference than on GPU. At even 200 crops (40 tiles × 5 crops), CPU inference would take many hours, not 30 minutes. The "CPU, ~30min" label will cause implementers to assume no GPU is needed for step 4, then be surprised by multi-hour runtimes or attempt it on CPU-only machines.

**Suggested fix:** Change the step 4 comment to "GPU required — or CPU-only with `tile=0` disabled; expect ~2–8 hours on CPU". Add Real-ESRGAN to the list of GPU-required steps alongside SDXL (step 5). Note that Real-ESRGAN and SDXL must not be loaded simultaneously if sharing a single GPU with limited VRAM.

---

### 4. [WARNING] Fix #3 (Real-ESRGAN ×4→resize) is not annotated in the data flow diagram

**Description:** Fixes #1, #2, #4, #5, and #6 are all annotated inline in the data flow diagram with "FIX #N:" labels at the relevant node. Fix #3 — the ×4 Real-ESRGAN then `cv2.resize` approach — occurs inside `process_geotiff_tiles()` [04] on the left branch of the diagram, but has no "FIX #3" annotation. The fix table at the bottom references it, but a developer reading the diagram alone would not know a fix was applied there.

**Suggested fix:** Add "FIX #3: ×4 Real-ESRGAN → cv2.resize(1024) for net ×2.5" as an annotation on the `process_geotiff_tiles()` node in the data flow diagram, consistent with all other fix annotations.

---

### 5. [WARNING] Background tile selection mechanism is entirely absent from the data flow

**Description:** The left branch produces `list[PNG] backgrounds/`. The diagram then shows a merge arrow near `composite()` labelled "background tile + patches + normalized_mask". But the mechanism by which a specific background tile is selected and paired with a specific layout is never defined anywhere in the overview:
- Is selection random? Round-robin? Seeded?
- What happens when there are more layouts than background tiles (or vice versa)?
- Who performs the selection — the compositor, an orchestration script, or a new function?
- Is the selected tile path recorded in `SampleMetadata.background_tile_path` before or after compositing?

**Suggested fix:** Add a `select_background(backgrounds: list[str], layout_seed: int) -> str` function (or document that the orchestration script does `rng.choice(backgrounds)`) between the left branch and `composite()` in the data flow. Specify seeding strategy. Document the case where `len(backgrounds) < n_layouts`.

---

### 6. [WARNING] `SampleMetadata` is produced at stage 03 but finalized at stages 06 and 09; the data flow implies it is complete at 03

**Description:** The data flow shows `metadata_{i:04d}.json` being written at stage 03 alongside `mask_{i:04d}.png`. But `SampleMetadata` contains fields that cannot be known at stage 03:
- `background_tile_path` — set when background is selected (stage 06)
- `image_path` / `mask_path` — updated by `export_split` (stage 09)
- `split` — assigned by `stratified_split` (stage 09)
- `source` — defaults to `'synthetic'` but must be set

Writing the metadata JSON at stage 03 and then mutating it at stages 06 and 09 means the file on disk is stale after stage 03. The diagram gives no indication of when the JSON is updated or re-written, which will cause confusion if stages are re-run independently.

**Suggested fix:** Change the data flow to show stage 03 writing only `mask_{i:04d}.png` and producing an in-memory partial metadata object (not yet serialised to JSON). Show `metadata.json` being written only once, at the end of stage 09, when all fields are populated.

---

### 7. [WARNING] `realesrgan` and `torch` are missing from the step 1 install command; steps 4 and 5 will fail

**Description:** The End-to-End Run Order step 1 installs:
```
numpy opencv-python pyyaml rasterio albumentations
```
Step 4 (`process_geotiff_tiles`) requires `realesrgan` and `torch`. Step 5 requires `diffusers`, `transformers`, `accelerate`, `torch`, and `xformers` (for SDXL). None of these appear in step 1. The `realesrgan` package has its own install instructions in `04_super_resolution.md`, but the overview's run order does not reference them, giving the false impression that step 1 is sufficient to run steps 4 and 5.

**Suggested fix:** Either split the install into two steps (CPU deps in step 1; GPU deps in step 1b with `pip install torch realesrgan diffusers transformers accelerate xformers`), or add a `requirements.txt` / `requirements-gpu.txt` and reference it in the run order. At minimum, add a note after step 1 that GPU steps 4 and 5 require additional packages.

---

### 8. [WARNING] `SDXLResult` dataclass is defined in shared structures but is inconsistent with the data flow

**Description:** `SDXLResult` is defined with four fields: `scene`, `patches`, `seg_condition`, `edge_condition`. It is described as the output of stage 05. But the data flow diagram shows these fields flowing as separate nodes through three distinct functions (`build_conditions`, `generate_sdxl_scene`, `extract_object_patches`), with `seg_condition + edge_condition` passed into `generate_sdxl_scene` and `patches` extracted afterwards. No node in the data flow shows an `SDXLResult` object being assembled or passed downstream. The data flow and the dataclass tell contradictory stories about what stage 05 returns.

**Suggested fix:** Either (a) revise the data flow to show `SDXLResult` assembled after `extract_object_patches` and passed as a unit to `composite()`, or (b) remove `SDXLResult` from the shared dataclasses section if the implementation uses the individual fields directly and the wrapper is never constructed. Pick one consistent model.

---

### 9. [WARNING] `generate_dataset()` is called in the run order but never appears in the data flow; single-sample vs. batch orchestration is undefined

**Description:** The data flow diagram depicts a single-sample pipeline: one `LayoutResult` flows through all stages. The run order step 3 calls `generate_dataset(n=1000, ...)`. There is no definition of `generate_dataset` in the shared dataclasses section, no explanation of how it relates to single-sample `generate_layout`, and no description of the batch orchestration loop that drives all 1000 samples through stages 05–09. Key questions are unresolved:
- Is stage 04 (background tiles) run once up-front or interleaved with stage 05?
- Are stages 05–08 run per-sample in a loop, or batched?
- Is there any checkpointing/resume logic for the multi-hour GPU step?

**Suggested fix:** Add a "Batch Orchestration" section to the overview that shows the outer loop structure: generate all backgrounds first, generate all layouts first, then loop samples through stages 05–09. Define `generate_dataset` in the component map or as an orchestration script separate from the individual component functions.

---

### 10. [WARNING] `QualityResult` dataclass fields beyond `passed` serve no documented purpose in the data flow

**Description:** `QualityResult` defines `blur_score: float`, `class_coverage: dict[int,int]`, and `rejection_reason: str | None`. In the data flow, stage 07 outputs only "passed samples only" — the `QualityResult` object is never shown flowing anywhere. It is unclear whether `blur_score` and `rejection_reason` are logged, stored, or discarded. If they are stored (e.g., for dataset curation analysis), the storage mechanism is undefined.

**Suggested fix:** Add a "Rejection log" output node from `filter_sample()` that writes a `rejection_log.jsonl` file (one `QualityResult` per line, for rejected samples). Show this in the data flow. If the fields beyond `passed` are truly unused, remove them from the dataclass to avoid misleading implementers.

---

### 11. [WARNING] GPU VRAM conflict between Real-ESRGAN (step 4) and SDXL (step 5) is not addressed

**Description:** Real-ESRGAN x4plus requires ~4–6 GB VRAM. SDXL with two ControlNet adapters requires ~16–24 GB VRAM. If both are run on the same GPU (e.g., a 24 GB A10G on RunPod), loading SDXL without first unloading Real-ESRGAN would OOM. The tech stack lists both without noting this constraint. The run order labels step 4 as CPU and step 5 as GPU (implying they never share GPU memory), but finding #3 above shows step 4 also requires GPU. The actual memory lifecycle is unspecified.

**Suggested fix:** Add a "Hardware requirements" sub-section listing VRAM requirements per step and explicitly stating that Real-ESRGAN weights must be released before loading SDXL. Alternatively, document that step 4 should be run on a separate CPU-only machine or a smaller GPU instance before the SDXL step begins.

---

### 12. [WARNING] `PipelineConfig` is shown flowing only into stages 02 and 04 in the data flow, but is implicitly consumed by most other stages

**Description:** The data flow diagram shows `PipelineConfig` branching to `generate_layout()` (stage 02) and `process_geotiff_tiles()` (stage 04). But several other stages also need configuration values:
- Stage 07 (`filter_sample`) needs blur threshold and coverage thresholds (are these hardcoded or from config?)
- Stage 08 (`augment_sample`) needs augmentation parameters
- Stage 09 (`stratified_split`) needs train/val/test ratios

If these parameters are hardcoded in each component, the single-source-of-truth promise of `pipeline_config.yaml` is broken. If they come from `PipelineConfig`, they should appear in the dataclass and in the data flow.

**Suggested fix:** Extend `PipelineConfig` to include sub-configs for quality gate, augmentation, and export (or document that these stages use hardcoded constants). Show `PipelineConfig` flowing to all stages that consume it in the data flow diagram.

---

### 13. [MINOR] `augment_sample() ×4` is ambiguous — original composite may be lost

**Description:** The data flow shows `augment_sample() ×4` followed by "~4000 (image, mask) pairs", implying that 1000 composites × 4 augmentations = 4000. But it is not documented whether:
- The 4 augmented versions *replace* the original (4000 total), or
- The 4 augmented versions *supplement* the original (5000 total: 1000 originals + 4000 augmented)

If originals are replaced, the non-augmented perspective of each scene is lost. If originals are preserved, the `~4000` figure is understated.

**Suggested fix:** Clarify as "augment_sample() → 4 augmented variants per original (originals preserved: ~5000 total)" or "→ 4 augmented variants replacing original: 4000 total". Update the "~4000" label accordingly. This also connects to the data leakage finding in `09_export_review.md` finding #1.

---

### 14. [MINOR] `placement_log` semantics are ambiguous — exhaustive vs. successes-only

**Description:** `LayoutResult.placement_log: dict[str, bool]` with example `{'tennis_1': True, 'pool_1': False}`. `SampleMetadata.placement_log: dict[str, bool]` with example JSON `{"tennis_1": true}` (only one entry). It is unclear whether keys represent all *attempted* placements (with `False` for failures) or only successful placements (with `True` always). The two examples contradict each other: the dataclass comment shows a `False` entry; the JSON example shows only a `True` entry.

**Suggested fix:** Standardise the semantics: "All attempted placements are recorded; value is `True` if placed successfully, `False` if placement failed (e.g., insufficient space)." Update the JSON example in `metadata.json` to show a `False` entry, confirming the exhaustive interpretation.

---

### 15. [MINOR] The directory structure lists `constants.py` as "legacy (superseded by config.py)" but it remains in the tree

**Description:** The project directory structure includes `pipeline/layout/constants.py` with the annotation "legacy (superseded by config.py)". Keeping a superseded file in the canonical project structure creates ambiguity: should new components import from `config.py` or `constants.py`? Does `constants.py` still contain values that `config.py` does not? If it is truly superseded, including it in the design doc normalises its presence.

**Suggested fix:** Either remove `constants.py` from the directory structure (and note it will be deleted) or document exactly which values remain in `constants.py` and why they haven't been migrated to `config.py`.

---

### 16. [MINOR] Class Balance Strategy table ordering is by pixel coverage, not by class ID — not documented

**Description:** The class balance table lists classes in the order 0, 2, 4, 3, 1 (background, tennis, eucalyptus, pool, crosswalk) — sorted by descending pixel coverage. A reader expecting class-ID order (0, 1, 2, 3, 4) will spend time looking for crosswalk (class 1) and find it last. The ordering is not explained in a table caption or footnote.

**Suggested fix:** Add a note "(sorted by expected pixel coverage, descending)" as a table caption. Alternatively, reorder by class ID for consistency with `class_labels.json` and the mask value encoding.

---

### 17. [MINOR] `export_split` is called three times (train/val/test) in normal usage but the run order shows it once

**Description:** The End-to-End Run Order step 6 lists `stratified_split, export_split, write_manifests` in one pseudocode block followed by `# ... orchestration script`. `export_split` must be called once per split (three times total). This was also flagged in `09_export_review.md` finding #13. In the context of the overview, the omission gives a false impression of the call complexity.

**Suggested fix:** Expand the `# ... orchestration script` comment in step 6 to show the three-call pattern for `export_split` explicitly, matching the recommended pattern in `09_export.md`.
