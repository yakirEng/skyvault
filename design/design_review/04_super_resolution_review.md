# Design Review — 04_super_resolution.md

Reviewer: senior ML engineer
Date: 2026-03-06
Files reviewed: `design/00_system_overview.md`, `design/04_super_resolution.md`

---

## Findings

### 1. [CRITICAL] Model reloaded from disk on every crop call

**Description:** `upscale_to_5cm(tile, model_path)` takes a file path, implying the RealESRGAN weights are loaded fresh on every invocation. `process_geotiff_tiles` calls this function `n_crops_per_tile × len(geotiff_paths)` times. Loading a 67 MB `.pth` checkpoint takes several seconds per call; for 200 crops this is ~15–30 minutes of pure I/O with no GPU work.

**Suggested fix:** Change the signature to `upscale_to_5cm(tile, model: RealESRGANer)`. Load the model once in `process_geotiff_tiles` before the loop and pass the instantiated object. Alternatively, add an `@lru_cache`-wrapped `load_esrgan_model(model_path)` helper.

---

### 2. [CRITICAL] RGB/BGR channel mismatch between rasterio output and Real-ESRGAN input

**Description:** `load_and_crop_tile` output is documented as "BGR→RGB converted", meaning the array delivered to `upscale_to_5cm` is in RGB channel order. Real-ESRGAN's `RealESRGANer.enhance()` follows the OpenCV convention and **expects BGR**. Passing an RGB array causes the R and B channels to be silently swapped in the super-resolved output, producing colour-corrupted background tiles.

**Suggested fix:** Convert RGB→BGR immediately before calling `.enhance()` inside `upscale_to_5cm` (e.g. `img_bgr = tile[:, :, ::-1]`), then convert the BGR output back to RGB before returning. Document this explicitly in the function docstring.

---

### 3. [CRITICAL] Reprojection step is fatally underspecified

**Description:** Step 2 of `load_and_crop_tile` states "If CRS is not EPSG:2039, reproject on-the-fly using rasterio.warp" but omits every parameter needed to implement it correctly:

- **Target pixel resolution** — must be exactly 0.125 m (12.5 cm/px) in EPSG:2039 metres. Without specifying this, `rasterio.warp.reproject` will use the source file's native resolution, which may differ.
- **Resampling algorithm** — not stated (Bilinear? Lanczos? Nearest?). Wrong choice degrades texture for SR input.
- **Scope** — reprojecting the entire tile in memory before cropping can require gigabytes of RAM for large gov.il tiles. Windowed reprojection of just the crop area should be preferred but is not mentioned.
- **Nodata fill value** — not specified; default fill (0 = black) would be indistinguishable from valid dark pixels.

If the target resolution is wrong, `crop_size_px=410` no longer maps to 51.2 m of ground coverage and the entire GSD chain is invalidated.

**Suggested fix:** Add a "Reprojection spec" sub-section specifying `rasterio.warp.calculate_default_transform` parameters, `target_res=(0.125, 0.125)` in CRS units, `Resampling.lanczos`, and `nodata=0`. Document the windowed-reproject approach to avoid loading the full tile.

---

### 4. [WARNING] Source GSD is assumed, never validated

**Description:** The function hardcodes `crop_size_px=410` on the assumption that the input tile is 12.5 cm/px. `load_and_crop_tile` never reads `rasterio.transform` to verify the actual pixel size. gov.il provides tiles at multiple resolutions (10 cm and 25 cm datasets exist). A tile at 10 cm/px would require `crop_size_px=512` for the same ground coverage; using 410 would silently under-crop by ~20%.

**Suggested fix:** After opening the file, assert `abs(src.res[0] - 0.125) < 0.005` (tolerance for floating-point variation in EPSG:2039 metres). Raise `ValueError` with the observed resolution if the check fails, rather than producing wrong-scale output.

---

### 5. [WARNING] Band count assumption is unguarded

**Description:** The design assumes a 3-band (R, G, B) GeoTIFF and instructs "stack bands as (H,W,3)". gov.il orthophotos commonly ship with a 4th band (near-IR or alpha). Calling `src.read()` without explicit band selection returns shape `(4, H, W)`, and the subsequent transpose to `(H, W, 4)` will cause a shape error in `upscale_to_5cm` which expects `(410, 410, 3)`.

**Suggested fix:** Always select bands explicitly: `src.read([1, 2, 3])`. Add a guard that raises `ValueError` if the file has fewer than 3 bands.

---

### 6. [WARNING] Nodata / tile-edge pixels not handled

**Description:** Random crop origin selection checks only that `(row_off + crop_size_px) <= tile_height`, but GeoTIFF tiles routinely have nodata regions at their edges (georeferenced tiles that don't fill a rectangular pixel grid, or cloud-masked areas). A crop overlapping a nodata region will contain black or fill-value pixels. These propagate through Real-ESRGAN and appear as dark rectangles in composited backgrounds, corrupting training images.

**Suggested fix:** After reading the window, check `np.any(crop == nodata_value)` (or use `src.dataset_mask()`). If the crop contains nodata pixels, re-sample a new origin (with a retry limit). Document the retry count as a parameter.

---

### 7. [WARNING] No geographic metadata saved with output PNGs

**Description:** Output files are named `bg_{i:04d}.png` with no record of the source tile path, crop bounding box (in EPSG:2039 metres), or which geographic region they cover. The design calls out geographic diversity (north/centre/south) as critical, but after the batch run there is no way to verify distribution, detect duplicates across runs, or geo-locate a background for debugging.

**Suggested fix:** Save a sidecar JSON (`bg_{i:04d}.json`) per output PNG containing `{"source_tif": "...", "row_off": ..., "col_off": ..., "crs": "EPSG:2039", "bbox_m": [...]}`. This mirrors the `SampleMetadata` pattern used elsewhere in the pipeline.

---

### 8. [MINOR] "BGR→RGB converted" annotation is misleading for a rasterio reader

**Description:** rasterio's `read()` returns bands in the order they are stored in the file (typically R, G, B for standard RGB GeoTIFFs) — **not** BGR. The output note "BGR→RGB converted" implies an OpenCV-style BGR source that does not exist when using rasterio. A developer reading this may add an erroneous extra channel swap on the rasterio output, inverting R/B before the array even reaches `upscale_to_5cm`.

**Suggested fix:** Replace the note with "Bands read in file order via rasterio (expected: R, G, B). No channel reorder needed at this step; BGR conversion is handled inside `upscale_to_5cm` before passing to Real-ESRGAN."

---

### 9. [MINOR] Real-ESRGAN API call is unspecified

**Description:** The design names the weights file and pip package but provides no pseudocode for the `RealESRGANer` constructor or `.enhance()` call. Key parameters that affect output quality and memory use are omitted: `scale=4`, `tile=512` (tiling to avoid OOM on large inputs), `tile_pad=10`, `pre_pad=0`. Without this, an implementer must reverse-engineer the API and may choose `tile=0` (no tiling), which will OOM on any GPU with < 24 GB VRAM for a 1640×1640 intermediate.

**Suggested fix:** Add a "Real-ESRGAN usage" code block in the Implementation Notes specifying the `RealESRGANer` constructor and `.enhance()` signature, with recommended `tile=512`.

---

### 10. [MINOR] Crop size rounding is undocumented and inconsistent with target GSD

**Description:** The exact source crop for a 1024-pixel output at ×2.5 is `1024 / 2.5 = 409.6 px`. The design rounds to 410 without comment. This makes the effective output GSD `12.5 × 410 / 1024 ≈ 5.005 cm/px` rather than exactly 5.0 cm/px — a 0.1% error. While numerically harmless, it is inconsistent with the system's stated canvas spec of "5 cm/px" and will confuse future developers.

**Suggested fix:** Document the rounding explicitly: "409.6 rounds to 410; resulting GSD ≈ 5.005 cm/px, accepted as within 0.1% of target." Alternatively use 409 (giving 4.994 cm/px) and note the same. Either choice should be deliberate and recorded.

---

### 11. [MINOR] No RNG seeding strategy for reproducibility in `process_geotiff_tiles`

**Description:** `process_geotiff_tiles` has no `seed` parameter and the function creates or accepts no global RNG. With no seed, re-running the step produces different crops, making the background tile set non-reproducible. This is inconsistent with the layout engine's explicit `seed` parameter and will complicate debugging and dataset versioning.

**Suggested fix:** Add a `seed: int = 42` parameter to `process_geotiff_tiles`. Construct `rng = np.random.default_rng(seed + tile_index)` per tile so each tile's crops are independently reproducible without all crops clustering at the same offsets.
