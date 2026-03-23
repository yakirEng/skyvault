# Design Review — 02_layout_engine.md

Reviewer: senior ML engineer
Source files reviewed: `design/00_system_overview.md`, `design/02_layout_engine.md`

---

## Findings

---

### 1. [CRITICAL] `_free_mask` signature mismatch — will raise `TypeError` at runtime

**Description:**
`_free_mask` is defined with a single argument:
```python
def _free_mask(canvas: np.ndarray) -> np.ndarray:
```
But it is called with two arguments in two separate places:
- `place_eucalyptus` docstring: `paint_mask = (blob==1) & _free_mask(canvas, road_mask)`
- Spatial Constraints table: `paint_mask = (blob==1) & _free_mask(canvas, road_mask)`

Any implementation that follows this spec as written will fail with `TypeError: _free_mask() takes 1 positional argument but 2 were given`.

**Suggested fix:**
Pick one canonical signature and apply it consistently across all references. Since `_free_mask` returns `canvas == 0` and road pixels are 255 (not 0), `road_mask` is already implicitly excluded — the single-argument form `_free_mask(canvas)` is sufficient. Remove `road_mask` from both call sites in the eucalyptus docstring and the constraints table.

---

### 2. [CRITICAL] `pixel_counts` computation algorithm is unspecified — naive implementation produces invalid output

**Description:**
`LayoutResult.pixel_counts` must contain only keys 0–4, with 255-valued road pixels folded into key 0. The design states this requirement but never specifies HOW to compute it. A straightforward `np.unique(canvas, return_counts=True)` will produce `{255: N, ...}` — violating the contract (key 255 present, background count understated). No implementation note, helper function, or pseudocode is provided.

**Suggested fix:**
Add an explicit note to the `generate_layout` docstring (or a separate `_compute_pixel_counts(canvas)` spec) with the exact algorithm:
`counts[0] = np.sum((canvas == 0) | (canvas == 255))` then `counts[k] = np.sum(canvas == k)` for k in 1–4.

---

### 3. [WARNING] `place_crosswalks` max instance count is unspecified

**Description:**
Every other placer documents its count range: tennis=0–2, pools=0–3, eucalyptus=0–5. `place_crosswalks` has no count documented anywhere. Downstream components (dataset generator, quality gate, class balance analysis) cannot predict crosswalk density or pixel coverage without this.

**Suggested fix:**
Add a count range to the `place_crosswalks` docstring (e.g., "Places 0–N crosswalks per canvas") and add a corresponding `max_crosswalks` field to the YAML and `ObjectConfig`.

---

### 4. [WARNING] `stripe_pattern` construction is undefined — crosswalk geometry is ambiguous

**Description:**
The crosswalk placer specifies `np.where(road_mask & stripe_pattern, 1, canvas)` but never defines how `stripe_pattern` is constructed from `crosswalk_stripe_px`, `crosswalk_gap_px`, and `crosswalk_len_px`. Critical open questions:
- Is the stripe pattern aligned parallel or perpendicular to the road direction?
- How is road orientation (horizontal vs. vertical) detected to orient stripes correctly?
- Is `crosswalk_len_px` the extent across the road or along it?
- Is the pattern full-canvas or clipped to the road band width?

Without this, two implementations of `place_crosswalks` from this spec will produce incompatible results.

**Suggested fix:**
Add a "Stripe Pattern Construction" subsection that defines: (a) stripes run perpendicular to road travel direction, (b) pattern is a boolean array tiling `[stripe_px ones, gap_px zeros]` along the road axis, (c) `crosswalk_len_px` is the extent across the road (capped at road width), (d) one crosswalk = one contiguous stripe group placed at a random offset along the road.

---

### 5. [WARNING] `road_mask` parameter is passed to tennis/pool/eucalyptus placers but is functionally unused

**Description:**
`place_tennis_courts`, `place_pools`, and `place_eucalyptus` all accept `road_mask: np.ndarray` in their signatures. However, all three use `_free_mask(canvas)` (which returns `canvas == 0`) for placement gating — and since road pixels are 255, they are already excluded without consulting `road_mask`. The parameter has no documented use inside these three functions, creating a gap between the API and the implementation intent.

**Suggested fix:**
Either (a) remove `road_mask` from the signatures of the three non-crosswalk placers and update `generate_layout` calls accordingly, or (b) document explicitly why it is passed (e.g., "reserved for future hard-exclusion logic" or "used to compute eroded free mask that excludes road adjacency buffer").

---

### 6. [WARNING] Placer retry logic is unspecified — reproducibility claim is unverifiable

**Description:**
The design claims "100 samples, 0 violations" and full reproducibility via seed. However, no placer documents how many random placement attempts are made before logging `False` in `placement_log`. The number of RNG draws per failed attempt directly affects the RNG state, which means any difference in retry counts between implementations will break seed-for-seed reproducibility.

**Suggested fix:**
Add a `max_attempts: int` parameter (or a fixed constant) to each placer's docstring specifying the maximum number of random candidate positions tried before giving up on an instance. Alternatively, add it to `PipelineConfig` in 01_config.

---

### 7. [WARNING] Object instance counts (max_tennis=2, max_pools=3, max_eucalyptus=5) are hardcoded in placer docstrings, not in YAML/config

**Description:**
The design principle in 01_config.md is "single source of truth for all pipeline parameters." Object instance counts are pipeline parameters that control dataset density and class balance, but they are only found in prose docstrings inside the layout engine design — not in `pipeline_config.yaml` or any `ObjectConfig` field. Changing class balance requires editing the implementation, not the config.

**Suggested fix:**
Add `max_tennis_courts`, `max_pools`, `max_eucalyptus`, and `max_crosswalks` fields to the `objects` section of `pipeline_config.yaml` and corresponding fields to `ObjectConfig`. The placer docstrings should reference these config values rather than hardcoding counts.

---

### 8. [MINOR] `pixel_counts` description uses ambiguous phrasing for crosswalk-on-road pixels

**Description:**
The `LayoutResult` table says: "road pixels counted as 0 unless overwritten." After crosswalk placement, road pixels overwritten with class 1 are no longer 255 — they are correctly counted as class 1. But the phrasing "unless overwritten" could be misread as "unless overwritten by any class including roads themselves (re-init to 0)." A reader unfamiliar with the order of operations may miscount crosswalk pixels.

**Suggested fix:**
Rephrase to: "pixels that remain 255 after all placers run are counted as class 0 (background); pixels overwritten by classes 1–4 are counted under their respective class ID."

---

### 9. [MINOR] `place_pools` docstring omits explicit return type description

**Description:**
`place_tennis_courts` states "Returns (canvas, placement_log entries for tennis)" in prose. `place_pools` docstring says only "Never overlaps roads or tennis courts" with no explicit return description. Inconsistent documentation makes the return contract unclear for `place_pools` at a glance.

**Suggested fix:**
Add "Returns (canvas, placement_log entries for pools)." to the `place_pools` docstring, matching the pattern used by `place_tennis_courts`.

---

## Summary Table

| # | Severity | Short title |
|---|----------|-------------|
| 1 | CRITICAL | `_free_mask` called with 2 args, defined with 1 |
| 2 | CRITICAL | `pixel_counts` computation algorithm unspecified |
| 3 | WARNING  | Crosswalk instance count undocumented |
| 4 | WARNING  | `stripe_pattern` construction undefined |
| 5 | WARNING  | `road_mask` param passed to tennis/pool/eucalyptus but unused |
| 6 | WARNING  | Placer retry count unspecified — breaks reproducibility |
| 7 | WARNING  | Object instance counts hardcoded, not in YAML config |
| 8 | MINOR    | `pixel_counts` phrasing ambiguous for crosswalk-on-road pixels |
| 9 | MINOR    | `place_pools` docstring missing return description |
