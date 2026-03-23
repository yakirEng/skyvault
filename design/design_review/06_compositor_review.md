# Design Review — `design/06_compositor.md`

**Reviewer:** Senior ML Engineer
**Date:** 2026-03-06
**Files reviewed:** `design/06_compositor.md`, `design/00_system_overview.md`
**Scope:** `cv2.seamlessClone` edge-case handling, mask update pixel accuracy, rotation consistency, interface correctness.

---

## Findings

### 1. Rotation note in `composite()` contradicts the mask alignment guarantee — design incoherence

**Severity: CRITICAL**

`composite()` contains this note:
> "Allowed rotation: if patch needs rotation (0/90/180/270° only), rotate BOTH patch.image and patch.alpha by the same angle before blending."

This is internally contradictory with the component's own "Mask Alignment Guarantee" section, which states:
> "objects are pasted at those exact coordinates with no rescaling or position shift"

Three unresolved problems:

**a) When is rotation applied?** No trigger is specified. `ObjectPatch` has no `rotation_deg` field (see `00_system_overview.md` shared types). The compositor has no way to know whether a given patch needs rotation — the field doesn't exist.

**b) After rotation, `patch.bbox` is wrong.** `bbox = (x1, y1, x2, y2)` records the bounding box of the object in the layout at its original orientation. If `patch.image` is rotated 90° in the compositor, its shape changes from `(h, w, 3)` to `(w, h, 3)`. The line `src[y1:y2, x1:x2] = rotated_image` then fails with a shape mismatch: the destination slice has height `y2-y1` and width `x2-x1`, but the rotated image has swapped dimensions.

**c) Double-rotation.** The layout engine already places tennis courts at 0° or 90° (configured in `pipeline_config.yaml`). The SDXL generation is conditioned on the resulting layout mask — it generates the object at the orientation already baked into the mask. Rotating the patch in the compositor would rotate an object that SDXL already generated at the correct angle. The final image would show a 0° object composited onto a 90° mask region, breaking alignment.

The rotation note appears to be a leftover idea that conflicts with how the rest of the pipeline works.

**Suggested fix:** Remove the rotation note from `composite()` entirely. Object orientation is determined by the layout engine and baked into the SDXL conditioning mask. The compositor must not apply any additional rotation. If data augmentation via rotation is desired, it belongs exclusively in Stage 08 (`augmentor.py`), where both image and mask are transformed simultaneously.

---

### 2. Boundary guard checks `bbox` position, but `seamlessClone` constraint is on alpha pixel positions

**Severity: CRITICAL**

The design states:
> "Skip (log warning) if bbox is within 3px of image edge."

OpenCV's `seamlessClone` has a hard constraint: **no pixel where `mask > 0` may touch the image boundary** (row 0, row H-1, col 0, col W-1). If any mask pixel is on the outermost row or column of the destination, OpenCV throws `cv2::Error` at runtime.

The 3px bbox check is the wrong proxy for two reasons:

**a) A bbox 4px from the edge can still have alpha pixels at the boundary.** Consider a tennis court (475px wide) placed such that `x1 = 4`. The bbox is 4px from the left edge — it passes the 3px check. But if `patch.alpha[0, 0] == True` (the top-left pixel of the patch is an object pixel), then `clone_mask_full[y1, x1] = 255` — which is at absolute column `x1 = 4`, NOT at column 0. This is fine. But now consider `x1 = 1`: bbox is 1px from left edge, fails the 3px check, gets skipped. However, if `patch.alpha[:, 0]` is all `False` (the leftmost column of the patch has no object pixels), the actual nearest mask pixel might be at `x1 + 5 = 6`, well clear of the boundary.

The check over-rejects safe cases and under-protects against real boundary violations depending on object shape.

**b) The 3px value has no documented derivation.** It is neither the OpenCV hard constraint (1px) nor a principled visual margin. It will silently discard objects that could have been safely composited, and may still miss the failure mode for objects with alpha pixels near their bbox edge.

**Suggested fix:** Replace the bbox check with a direct check on the alpha pixel positions in full-image coordinates:
```
full_y, full_x = np.where(patch.alpha)
full_y += y1
full_x += x1
if full_y.min() < 1 or full_y.max() >= H-1 or full_x.min() < 1 or full_x.max() >= W-1:
    log warning and skip
```
This checks the actual hard constraint (no mask pixel on the outermost row/column) rather than an approximation. Also document separately a soft visual margin (e.g., warn if any alpha pixel is within 10px of the boundary) to catch blending halos.

---

### 3. NumPy chained indexing for mask update is semantically fragile — relies on undocumented view behavior

**Severity: WARNING**

The mask update step is:
```python
output_mask[y1:y2, x1:x2][patch.alpha] = patch.class_id
```

This is two sequential indexing operations:
1. `output_mask[y1:y2, x1:x2]` — basic slice indexing → returns a **view** (no copy)
2. `[patch.alpha]` with boolean mask on that view → assignment propagates to underlying array

This works in NumPy because basic slicing always returns a view, and assigning to a boolean-indexed subset of a view writes through to the original memory. However:

- This behaviour is non-obvious. Many experienced developers would expect the second `[]` to create a copy (since boolean indexing in NumPy is "advanced indexing" and normally produces copies in a read context). Any implementor who refactors this line risks introducing a silent no-op if they reverse the order or introduce an intermediate variable.
- It does NOT work if `output_mask[y1:y2, x1:x2]` is assigned to an intermediate variable first and then boolean-indexed in a separate Python expression. For example: `roi = output_mask[y1:y2, x1:x2]; roi[patch.alpha] = patch.class_id` — this form DOES work (roi is a view), but `roi = output_mask[y1:y2, x1:x2].copy(); roi[patch.alpha] = ...` does NOT.

**Suggested fix:** Use the unambiguous single-step form:
```python
alpha_full = np.zeros(output_mask.shape, dtype=bool)
alpha_full[y1:y2, x1:x2] = patch.alpha
output_mask[alpha_full] = patch.class_id
```
Or document the view semantics explicitly with a comment so implementors do not "fix" the chained form.

---

### 4. `blend_patch()` parameter named `background` — callers can corrupt the original source tile

**Severity: WARNING**

`blend_patch()` modifies its `background` parameter in place. The parameter name `background` is the same name used for the original real tile input to `composite()`. `composite()` correctly passes `image = background.copy()` to protect the original, but `blend_patch()` has no way to enforce this.

If a caller invokes `blend_patch(background, patch, mask)` passing the original background tile directly (not a copy), the tile is permanently corrupted. This is a hidden mutation trap. In the full pipeline, multiple samples reuse background tiles from a shared pool — corrupting one tile would contaminate all subsequent samples that draw from it.

**Suggested fix:** Rename the parameter to `canvas` or `dst` in `blend_patch()` to signal that it is a mutable working buffer, not the source tile. Add a docstring note: "caller must ensure this is a copy of the original tile; `blend_patch` modifies it in place." Alternatively, change `blend_patch()` to return the modified image instead of mutating in place, removing the mutation risk entirely.

---

### 5. No guard against empty `patch.alpha` (all-False mask)

**Severity: WARNING**

If `patch.alpha` is entirely `False` (no True pixels), then:
- `clone_mask_full` is all zeros
- `cv2.seamlessClone` receives an all-zero mask

OpenCV's behaviour with an all-zero mask is undefined in the documentation — in practice it either returns `dst` unchanged (acceptable) or raises an exception (not acceptable). This is not tested or guarded against in the design.

`05_controlnet_generation.md` specifies `MIN_COMPONENT_AREA` thresholds that should prevent degenerate patches from being created. However, the compositor has no guarantee that this contract is honoured — defensive validation is the caller's (compositor's) responsibility in a layered design.

**Suggested fix:** Add an early guard in `blend_patch()`:
```python
if not patch.alpha.any():
    # log warning: empty patch received, skipping
    return
```
Also specify this as an explicit pre-condition in the docstring.

---

### 6. `seamlessClone` mode `NORMAL_CLONE` vs. `MIXED_CLONE` is undocumented — may produce halos on high-contrast objects

**Severity: WARNING**

The design specifies `cv2.NORMAL_CLONE` without justification. The two modes differ in a way that matters for this pipeline:

- `NORMAL_CLONE`: copies `src` pixel values inside the mask and solves Poisson only at the boundary. Preserves SDXL-generated texture exactly.
- `MIXED_CLONE`: blends `src` and `dst` gradient fields inside the mask. Results look more natural when the object has transparent or background-like interior regions.

For this pipeline, objects like eucalyptus canopy (irregular blob with semi-transparent edges) and pools (high color contrast with arid background) are likely candidates for boundary halos under `NORMAL_CLONE`. A blue pool pasted onto ochre soil will produce a visible seam even after Poisson blending if the gradients are large.

`MIXED_CLONE` would partially absorb the background texture into the object interior, reducing halos but altering the SDXL-generated appearance.

**Suggested fix:** Document the deliberate choice of `NORMAL_CLONE` and its trade-off. Add a note that `MIXED_CLONE` should be considered for eucalyptus patches specifically, where canopy edges are naturally semi-opaque. Consider making the clone mode a config parameter (per class) to allow per-object tuning.

---

### 7. Full-size `src` buffer allocated per patch — significant memory pressure at scale

**Severity: WARNING**

Each call to `blend_patch()` allocates:
```python
src = np.zeros_like(background)  # 1024×1024×3 uint8 = 3.1 MB
```

For a sample with 10 patches, that is 31 MB of allocation and deallocation per sample. Across 1000 samples, this generates 31 GB of total GC pressure. Python's garbage collector will struggle with numpy array churn at this scale, and peak memory usage during a batch run may be unexpectedly high.

**Suggested fix:** Pre-allocate a single reusable `src` buffer in `composite()` and pass it to `blend_patch()` as a parameter, zeroing only the touched region before each use:
```python
src = np.zeros_like(image)
# in blend_patch: zero only the bbox before placing patch
src[y1:y2, x1:x2] = 0
src[y1:y2, x1:x2] = patch.image
```
Document this as a performance requirement in the design.

---

### 8. `center` is computed from bbox corners — integer truncation for even/odd dimensions

**Severity: WARNING**

```python
center = ((x1+x2)//2, (y1+y2)//2)
```

When `x2-x1` is odd (e.g., tennis court width = 475px), `(x1+x2)//2` truncates, placing the logical center half a pixel to the left of the geometric center. OpenCV's `seamlessClone` interprets `center` as the position where the CENTROID of the non-zero mask region is placed in `dst`. If the mask region's centroid doesn't precisely coincide with this truncated center, the Poisson solve will shift the blended result by ±1px relative to `output_mask`.

For tennis courts (475 × 219 px), the shift is at most 1px in each axis. For small crosswalk patches, this 1px shift may matter.

This doesn't break mask alignment (the mask update in step 5 is independent of seamlessClone — it writes directly to `output_mask[y1:y2, x1:x2]` based on `patch.alpha`), but it means the blended image pixels may be 1px offset from the mask pixels for odd-dimension objects. The image shows the object 1px left/up of where the mask says it is.

**Suggested fix:** Use the actual centroid of the alpha mask as the center:
```python
rows, cols = np.where(patch.alpha)
cy = int(rows.mean()) + y1
cx = int(cols.mean()) + x1
center = (cx, cy)
```
This makes the center match the actual object geometry in the `src` buffer, ensuring seamlessClone places the blend precisely over the mask region.

---

### 9. Output channel order (RGB/BGR) not specified for `CompositeResult.image`

**Severity: MINOR**

`background` (from SR) is documented in `04_super_resolution.md` as RGB. `patch.image` (from SDXL via PIL) is RGB. `cv2.seamlessClone` is channel-order-agnostic for blending purposes. However, `CompositeResult.image` does not document its channel order. Downstream components (`07_quality_gate` uses `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`) assume RGB input. If the compositor ever loads backgrounds via OpenCV directly (which returns BGR), the channel order would be wrong.

**Suggested fix:** Add to the `CompositeResult` definition and `composite()` docstring: "output image is RGB, matching the SR background tile input. Do not use `cv2.imread()` to load background tiles without BGR→RGB conversion."

---

### 10. Empty patches list behaviour not documented

**Severity: MINOR**

If `patches` is an empty list (all placement attempts in the layout engine failed), `composite()` performs no blending and returns:
- `image = background.copy()` — correct
- `mask = normalized_mask.copy()` — which would be all zeros (no objects) if export_mask was applied

This is technically correct, but undocumented. More importantly: the `normalized_mask` passed in would have `classes_present = [0]` only, and the quality gate should reject such samples. But the compositor should still handle the empty case gracefully (it does, implicitly), and this should be an explicit pre-condition note.

**Suggested fix:** Add a docstring note: "If `patches` is empty, returns the background as-is with the normalized_mask as the mask. The quality gate will reject such samples due to zero coverage for all non-background classes."

---

### 11. Pasting order in `composite()` sorts by class ID but crosswalk logic is wrong

**Severity: MINOR**

`composite()` specifies pasting order: tennis (2) → pool (3) → eucalyptus (4) → crosswalk (1).

The comment says "crosswalk last (it's on roads, not competing with others)." This ordering is correct for the mask — crosswalk should be the final writer so it isn't overwritten by pool or eucalyptus.

However, for the IMAGE blend, pasting crosswalk stripes LAST means `seamlessClone` uses the already-modified background (after tennis/pool/eucalyptus blending) as the `dst`. The Poisson solve for crosswalk stripes will blend against whatever was just pasted near the road — which is correct behavior and intentional.

The issue: the sort is documented as hardcoded by class ID order (2→3→4→1). If class IDs change in the config (e.g., if a new class is inserted), the pasting order breaks silently. The sort should be driven by a configurable list, not numeric class ID values.

**Suggested fix:** Define a `PASTE_ORDER: list[int] = [2, 3, 4, 1]` constant in the module (or in config) and sort `patches` by this list, not by class ID directly. Add a comment: "order matters — later patches overwrite earlier ones in both image and mask."

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | CRITICAL | Rotation note in `composite()` contradicts mask alignment guarantee; `ObjectPatch` has no rotation field; post-rotation `bbox` is wrong |
| 2 | CRITICAL | Boundary guard checks `bbox` position, not alpha pixel positions; 3px threshold undocumented; wrong failure mode |
| 3 | WARNING | Chained NumPy indexing `output_mask[...][alpha] = x` relies on view semantics; fragile to refactoring |
| 4 | WARNING | `background` parameter mutated in place; callers sharing tile pool risk silent corruption |
| 5 | WARNING | No guard for empty `patch.alpha`; `seamlessClone` behavior with all-zero mask is undefined |
| 6 | WARNING | `NORMAL_CLONE` vs `MIXED_CLONE` choice undocumented; likely produces halos on high-contrast objects (pool, eucalyptus) |
| 7 | WARNING | New 3MB `src` buffer allocated per patch; 31GB+ GC pressure for 1000-sample run |
| 8 | WARNING | `center` uses integer truncation for odd-dimension patches; up to 1px shift between blended image and mask |
| 9 | MINOR | `CompositeResult.image` channel order (RGB) not documented; downstream OpenCV calls assume RGB |
| 10 | MINOR | Empty `patches` list behavior not documented |
| 11 | MINOR | Pasting order hardcoded by class ID value; silently breaks if class IDs change |

**2 CRITICAL, 6 WARNING, 3 MINOR findings.**
