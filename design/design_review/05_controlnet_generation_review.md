# Design Review — 05_controlnet_generation.md

Reviewer: Senior ML Engineer
Date: 2026-03-06

---

## Findings

---

### 1. [CRITICAL] `diffusers/controlnet-seg-sdxl-1.0` is not a verified HuggingFace model ID

**Description:** `load_pipeline()` references `"diffusers/controlnet-seg-sdxl-1.0"` as the segmentation ControlNet. The `diffusers` HuggingFace org does host `diffusers/controlnet-canny-sdxl-1.0`, but there is no confirmed `diffusers/controlnet-seg-sdxl-1.0` repo. Attempting to load a non-existent model ID will raise an `OSError` / `RepositoryNotFoundError` at runtime and halt the pipeline entirely.

**Suggested fix:** Verify the exact repo slug on HuggingFace before freezing the design. Known alternatives with documented SDXL segmentation support include `SargeZT/controlnet-sd-xl-1.0-seg` and `xinsir/controlnet-seg-sdxl-1.0`. Replace the placeholder with the confirmed ID and pin a specific `revision` or `commit_hash` for reproducibility.

---

### 2. [CRITICAL] `.to(device)` conflicts with `enable_model_cpu_offload()` in `load_pipeline()`

**Description:** The pseudocode chains `.from_pretrained(...).to(device)` and then immediately calls `pipeline.enable_model_cpu_offload()`. In diffusers, `enable_model_cpu_offload()` uses `accelerate` to manage its own device placement. Calling `.to(device)` first moves all weights to the GPU; the subsequent offload call then conflicts with that state, producing undefined behaviour and potentially exceeding the 16 GB VRAM budget.

**Suggested fix:** Remove `.to(device)` from the constructor chain. Use either `pipeline.enable_model_cpu_offload(gpu_id=0)` alone (offloaded mode), or `pipeline.to(device)` alone (fully resident on GPU). Do not combine both.

---

### 3. [CRITICAL] `normalized_mask` is not returned by `build_conditions()`, so `run_generation()` cannot pass it to `extract_object_patches()`

**Description:** `build_conditions()` is specified to return `(seg_condition, edge_condition)`. It internally computes `normalized_mask = export_mask(layout_result.canvas)` but discards it from the return value. `run_generation()` must pass `normalized_mask` to `extract_object_patches()` — but as currently specified, there is no way to obtain it without a redundant second call to `export_mask()`. Redundant computation on 1024×1024 arrays is wasteful and error-prone (e.g., if `export_mask` is ever made non-deterministic).

**Suggested fix:** Either (a) change `build_conditions()` to return a 3-tuple `(seg_condition, edge_condition, normalized_mask)`, or (b) have `run_generation()` call `export_mask()` directly before delegating to `build_conditions()` and `extract_object_patches()`, and update the data-flow docs accordingly.

---

### 4. [CRITICAL] `patches` list is never initialized in `extract_object_patches()` pseudocode

**Description:** The pseudocode body calls `patches.append(ObjectPatch(...))` but no line initializes `patches = []`. This is an unresolved reference that would raise `NameError` at runtime.

**Suggested fix:** Add `patches: list[ObjectPatch] = []` at the top of the function body before the outer `for class_id` loop. Also add the matching `return patches` at the end (currently implied only in the docstring).

---

### 5. [WARNING] `generate_sdxl_scene()` is missing critical inference hyperparameters

**Description:** The function signature specifies `seed` and `conditioning_scale` but omits `num_inference_steps`, `guidance_scale`, `height`, and `width`. These four parameters directly affect output quality, VRAM usage, and inference time. Without them the implementation will use diffusers' defaults (e.g., 50 steps, guidance 7.5), which may not match quality expectations and will not be reproducible across diffusers versions.

**Suggested fix:** Add `num_inference_steps: int = 30`, `guidance_scale: float = 7.5`, `height: int = 1024`, `width: int = 1024` to the signature. Specify recommended values in the design doc and store them in `PipelineConfig` or as module-level constants.

---

### 6. [WARNING] `device` for `torch.Generator` is undefined inside `generate_sdxl_scene()`

**Description:** The docstring states "Uses `torch.Generator(device).manual_seed(seed)`" but `device` is not a parameter of `generate_sdxl_scene()` — it is only a parameter of `load_pipeline()`. Using `torch.Generator("cpu")` is safe for seed reproducibility but explicitly using `"cuda"` when the pipeline runs on CPU-offload mode will raise a device mismatch error.

**Suggested fix:** Add a `device: str = "cuda"` parameter to `generate_sdxl_scene()`, or document that the generator is always created on `"cpu"` (which is safe for diffusers pipelines regardless of model device).

---

### 7. [WARNING] `edge_condition` channel expansion (1-channel → 3-channel) is specified only in implementation notes, not in the function contract

**Description:** `generate_sdxl_scene()` accepts `edge_condition` as `(1024, 1024)` uint8 (single-channel), but diffusers' `StableDiffusionXLControlNetPipeline` requires all control images to be PIL `RGB` images or 3-channel tensors. The expansion `np.stack([edges]*3, axis=2)` is buried in the "Implementation Notes" section and is absent from the function's docstring. An implementer reading only the function spec will produce an error.

**Suggested fix:** Either (a) change the `edge_condition` input type in the function signature to `(1024, 1024, 3) uint8` (expand before calling), or (b) document the expansion explicitly inside the `generate_sdxl_scene()` docstring as a required pre-processing step, not just in the module-level notes.

---

### 8. [WARNING] ADE20K class mapping for the four custom classes is not defined in this component

**Description:** `build_conditions()` calls `mask_to_ade20k_rgb(normalized_mask)` to produce the segmentation ControlNet input. The mapping from the pipeline's class IDs (crosswalk=1, tennis court=2, pool=3, eucalyptus=4) to ADE20K palette RGB values is entirely delegated to `palette.py` with no cross-reference here. If the wrong ADE20K class colours are chosen (e.g., "road" colour for crosswalk instead of "crosswalk" colour), the segmentation ControlNet will misinterpret the conditioning and generate incorrect textures — silently, with no error.

**Suggested fix:** Add a table to this design doc (or a direct link to the relevant section in `03_dataset_generator.md` / `palette.py`) listing the ADE20K RGB value chosen for each class and the rationale. This is critical for debugging realism issues later.

---

### 9. [WARNING] `cv2.connectedComponents` connectivity is unspecified; default 8-connectivity may merge crosswalk stripes

**Description:** `extract_object_patches()` calls `cv2.connectedComponents(binary)` without specifying the `connectivity` parameter. OpenCV's default is 8-connectivity. Crosswalk stripes (class 1) are thin, closely spaced rectangles that may touch diagonally under 8-connectivity, causing multiple stripes to be merged into one oversized component — or conversely, a single stripe to be split at a narrow diagonal gap under 4-connectivity.

**Suggested fix:** Explicitly specify `connectivity=8` (or `connectivity=4`) in the call and document the rationale. For crosswalks in particular, consider whether each stripe should be one `ObjectPatch` or the whole crosswalk group should be one — the current design is ambiguous on this.

---

### 10. [MINOR] `load_pipeline()` return type is annotated as `object` instead of the concrete type

**Description:** `def load_pipeline(device: str = "cuda") -> object` obscures the return type. All call sites cast this to `StableDiffusionXLControlNetPipeline`, and IDEs/type-checkers will provide no useful inference.

**Suggested fix:** Use `-> StableDiffusionXLControlNetPipeline` (with a `TYPE_CHECKING` import guard if needed to avoid circular imports or heavy startup cost).

---

### 11. [MINOR] No `torch.inference_mode()` context specified for SDXL inference

**Description:** `generate_sdxl_scene()` does not specify that inference should run under `torch.inference_mode()` (or at minimum `torch.no_grad()`). Without this, PyTorch retains the autograd graph for all intermediate activations, substantially increasing VRAM usage during a 1024×1024 SDXL forward pass — potentially causing OOM on 16 GB cards.

**Suggested fix:** Document that the pipeline call must be wrapped in `with torch.inference_mode():`. In practice diffusers' `__call__` applies this internally, but it should be stated explicitly so implementers do not inadvertently run the conditioning preprocessing outside the context.

---

### 12. [MINOR] Module attribution for `mask_to_ade20k_rgb()` and `mask_to_canny_edges()` is inconsistent

**Description:** The Dependencies section of component 05 lists both functions under `03_dataset_generator`. However, `00_system_overview.md` (Project Directory Structure) places them in `pipeline/layout/palette.py` — a separate file. This discrepancy will cause implementers to look in the wrong module.

**Suggested fix:** Update the Dependencies section to read `pipeline/layout/palette.py` (Component 03 palette sub-module) and match the path shown in the system overview.

---

### 13. [MINOR] No VAE tiling or memory-efficient attention noted for 1024×1024 inference on 16 GB VRAM

**Description:** The installation note states "16 GB minimum for float16 SDXL." SDXL at 1024×1024 with Multi-ControlNet (two ControlNet models loaded simultaneously) can exceed 16 GB without `enable_vae_tiling()` or `enable_xformers_memory_efficient_attention()`. Neither is mentioned, leaving implementers to discover OOM errors empirically.

**Suggested fix:** Add `pipeline.enable_vae_tiling()` to the `load_pipeline()` spec (safe for 1024×1024, no quality loss). Note that xFormers is optional but recommended if available. Update the VRAM estimate to reflect Multi-ControlNet overhead.
