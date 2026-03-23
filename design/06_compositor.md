# Component 06 — Compositor

## Purpose

Paste SDXL object patches onto real background tiles using Poisson blending (`cv2.seamlessClone`). Simultaneously build the final segmentation mask by writing each object's class ID at its exact pixel locations.

**Fix #5 applied here:** The original design included ±10% scale jitter on pasted objects, which would cause mask-image misalignment (the mask records object positions but the pasted image would be scaled). Scale jitter is removed. Objects are pasted at their exact layout pixel coordinates. Background variety is achieved by randomly selecting different background tiles.

---

## Python Files

| File | Role |
|------|------|
| `pipeline/compositor/compositor.py` | `blend_patch()`, `composite()` |

---

## Dependencies

- `05_controlnet_generation` — `ObjectPatch`
- `03_dataset_generator` — `export_mask()` (normalized_mask as starting mask)
- External: `opencv-python`

---

## Inputs

### `blend_patch()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `background` | `np.ndarray` `(1024,1024,3)` `uint8` | Real tile — modified in place |
| `patch` | `ObjectPatch` | Object to paste |
| `output_mask` | `np.ndarray` `(1024,1024)` `uint8` | Accumulating mask — modified in place |

### `composite()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `background` | `np.ndarray` `(1024,1024,3)` `uint8` | Real background tile (from super-resolution) |
| `patches` | `list[ObjectPatch]` | All object patches from SDXL |
| `normalized_mask` | `np.ndarray` `(1024,1024)` `uint8` | From `export_mask()` — values 0-4 |

---

## Outputs

### `blend_patch()`

```
Returns: None (modifies background and output_mask in place)
```

### `composite()`

```
Returns: CompositeResult
    image: np.ndarray (1024,1024,3) uint8 — real background + SDXL objects blended
    mask:  np.ndarray (1024,1024)   uint8 — final segmentation mask, values 0-4 ONLY
```

**Saved to disk by the orchestration script (not by this module):**
```
images/sample_{i:04d}.png  — (1024,1024,3) uint8 RGB
masks/sample_{i:04d}.png   — (1024,1024)   uint8 grayscale, values 0-4
```

---

## Functions

```python
def blend_patch(
    background: np.ndarray,
    patch: ObjectPatch,
    output_mask: np.ndarray
) -> None:
    """
    Poisson-blend a single ObjectPatch onto the background image.

    Steps:
    1. Build float mask for seamlessClone:
           clone_mask = (patch.alpha.astype(np.uint8) * 255)  # (h,w) uint8, 0 or 255
    2. Compute paste center in full-image coordinates:
           x1,y1,x2,y2 = patch.bbox
           center = ((x1+x2)//2, (y1+y2)//2)  # (col, row) as required by OpenCV
    3. Create a full-size source image (same size as background) initialized to zeros,
       then place patch.image into it at patch.bbox:
           pipeline = np.zeros_like(background)
           pipeline[y1:y2, x1:x2] = patch.image
       Create full-size clone_mask_full analogously from clone_mask.
    4. Blend:
           result = cv2.seamlessClone(pipeline, background, clone_mask_full, center, cv2.NORMAL_CLONE)
           background[:] = result
    5. Update mask:
           output_mask[y1:y2, x1:x2][patch.alpha] = patch.class_id

    Note: seamlessClone requires the patch to be non-trivially inside the image boundary.
    Skip (log warning) if bbox is within 3px of image edge.
    """


def composite(
    background: np.ndarray,
    patches: list[ObjectPatch],
    normalized_mask: np.ndarray,
) -> CompositeResult:
    """
    Composite all object patches onto a real background tile.

    FIX #5: No scale jitter. Objects are pasted at exact layout pixel coordinates
    (patch.bbox comes directly from extract_object_patches, which used the layout mask).
    Mask and image remain pixel-aligned by construction.

    Steps:
    1. image = background.copy()
    2. output_mask = normalized_mask.copy()
    3. Sort patches by pasting order: tennis (2) → pool (3) → eucalyptus (4) → crosswalk (1)
       Rationale: larger objects first; crosswalk last (it's on roads, not competing with others)
    4. For each patch: blend_patch(image, patch, output_mask)
    5. Return CompositeResult(image=image, mask=output_mask)

    Allowed rotation: if patch needs rotation (0/90/180/270° only), rotate BOTH
    patch.image and patch.alpha by the same angle before blending.
    """
```

---

## Mask Alignment Guarantee

Because `extract_object_patches()` records `bbox` from the same `normalized_mask` that is passed here as `output_mask`, and objects are pasted at those exact coordinates with no rescaling or position shift, the `output_mask` at completion is pixel-perfectly aligned to `image`. No reprojection is needed.

---

## Background Variety

Diversity in the dataset comes from:
1. **Different background tiles** — each sample draws from a randomly selected real geographic crop
2. **Different layout seeds** — different object positions, counts, sizes each sample
3. **Augmentation ×4** — horizontal/vertical flips and 90° rotations (applied later in Stage 08)

Scale jitter was removed because it breaks mask alignment (Fix #5). If visual scale variation is desired, it should be applied in Stage 08 augmentation in a form that transforms both image and mask simultaneously.

---

## Implementation Notes

- `cv2.seamlessClone` requires `src`, `dst`, and `mask` to have the same size — hence the full-size `src` buffer approach in `blend_patch()`
- If a patch's `alpha` is entirely within a 3px border of the image, `seamlessClone` may fail — skip and log
- `output_mask` starts as a copy of `normalized_mask`, which already has crosswalk pixels correctly set (class 1 over road areas); `blend_patch` overwrites only pixels where `patch.alpha == True`
- Pasting order matters for the mask: later patches overwrite earlier ones in the mask (same as canvas placement order)
