# SleapCode — Code Review Session

**Date:** 2026-04-09  
**Reviewer:** Claude Sonnet 4.6  
**Scope:** Full codebase review, bug identification, prioritized improvement plan, incremental fixes

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Code Review](#2-code-review)
3. [How to Fix Bug 8 (Tail Position)](#3-how-to-fix-bug-8-tail-position)
4. [Improvement Plan](#4-improvement-plan)
5. [Fix Status After Manual Fixes](#5-fix-status-after-manual-fixes)
6. [Remaining Fixes Applied](#6-remaining-fixes-applied)
7. [Final Review — Items 6 7 9 10 11 17 19 20 23](#7-final-review--items-6-7-9-10-11-17-19-20-23)

---

## 1. Project Overview

### What the Project Does

SleapCode is a post-processing pipeline for Drosophila behavioral experiments. It takes pose-tracking output from [SLEAP](https://sleap.ai/) and computes 40+ behavioral features per fly per frame, then exports them to HDF5 and MATLAB formats for downstream classification in JAABA.

### Three-Stage Pipeline

| Stage | Script | Input → Output |
|---|---|---|
| 1 — Split arenas | `split_and_inference/split_experiment_to_arenas.py` | Multi-arena recording → per-arena folders |
| 2 — Inference | `split_and_inference/run_sleap_inference.py` | Arena video → `.slp` → `.analysis.h5` |
| 3 — Features | `run_pipeline.py` | `.analysis.h5` → `.features.h5` + `perframe/*.mat` + `trx.mat` |

### Folder and Module Map

```
SleapCode/
├── config.yaml                  # All paths, calibration constants, skeleton config
├── params.py                    # Config loader; exposes constants globally on import
├── dataset.py                   # Core feature orchestration + HDF5 writing
├── io_utils.py                  # SLEAP HDF5 loading
├── preprocessing.py             # Signal processing (fill_missing, normalize_to_egocentric, signed_angle)
├── run_pipeline.py              # ENTRY POINT (Stage 3) — GUI, drives full pipeline
│
├── features/                    # Feature computation (no __init__.py)
│   ├── registry.py              # Central registry: 40+ FeatureSpec entries, dependency resolver, validator
│   ├── appearance.py            # Body geometry (position, axes, area, eccentricity)
│   ├── locomotion.py            # Movement dynamics (heading, velocity, angular rate)
│   ├── position.py              # Velocity direction (phi, yaw)
│   ├── social.py                # Pairwise interactions (distances, bearing angles)
│   ├── wing_appearance.py       # Wing angles in egocentric frame
│   └── wing_movement.py         # Wing angle dynamics (rates of change)
│
├── perframe/                    # JAABA export (no __init__.py)
│   ├── create_perframe.py       # Export scalar features → perframe/*.mat
│   ├── trx_mat.py               # Export trajectory struct → trx.mat
│   └── ds_utils.py              # HDF5 iteration + atomic MAT file writing
│
└── split_and_inference/         # Upstream stages (no __init__.py)
    ├── split_experiment_to_arenas.py
    └── run_sleap_inference.py
```

### Data Flow

```
SLEAP .analysis.h5
  │
  ▼ io_utils.load_tracks()
  │  → raw pose array (frames, nodes, xy, flies)
  │
  ▼ dataset.make_expt_dataset()
  │  → registry.compute_item() for each feature (recursive, memoized)
  │     └─ preprocessing, appearance, locomotion, position, social, wings
  │  → writes *.features.h5
  │     /meta/, /pose/tracks/, /<scalar_key>/
  │
  ├─ create_perframe.export_perframe()
  │    → reads scalar datasets → safe_savemat() → perframe/<key>.mat
  │
  └─ trx_mat.save_trx()
       → reads /pose/tracks → safe_savemat() → trx.mat
```

### Key Architectural Conventions

- **Track array shape:** `(n_frames, n_nodes, 2, n_flies)` throughout.
- **Scalar feature shape:** `(n_frames, n_flies)`.
- **`a_mm` convention:** `a_mm = head_abdomen_dist / 4 / pxpermm`. The ellipse semi-major axis is `2*a_mm`, not `a_mm`.
- **NaN convention:** Frame 0 of any differential feature is always NaN (no prior frame). Never fill this.
- **`params.py`** loads `config.yaml` once at import time. Config changes during a session require restarting the interpreter.

---

## 2. Code Review

### Confirmed Bugs

| # | File | Location | Severity | Description |
|---|---|---|---|---|
| B1 | `io_utils.py` | line 33 | **High** | Off-by-one: `tracks[:last_fidx]` silently drops the last valid frame. Fix: `tracks[:last_fidx + 1]`. |
| B2 | `registry.py` | line 151 | **High** | Typo `"cte_ind"` instead of `"ctr_ind"` in `xy_mm` params. `NODE_IDX_THORAX` never applied; `x_mm`/`y_mm` always use node index 1. |
| B3 | `locomotion.py` | lines 233, 306 | **High** | Tail position sign error. `cos(-θ) = cos(θ)`, so the tail x-position is computed in the wrong direction. Affects `du_tail`, `dv_tail`, `velmag_tail`. |
| B4 | `registry.py` | line 224 | **Medium** | Intermediate key mismatch: `nose_tail_mm` declares `"x_nose_mm"` but `compute_nose_tail` returns `"nose_x_mm"`. |
| B5 | `dataset.py` | line 174 | **Medium** | Shape validation after destructuring: `tracks.shape` is unpacked before the validation check, producing a confusing error on bad input. |
| B6 | `trx_mat.py` | line 148 | Low | Duplicate dict key `"firstframe"` (same value, harmless, copy-paste error). |
| B7 | `create_perframe.py` | line 102 | Low | Dead code: both branches of `if/else` in `jaaba_data_cell` are identical. |
| B8 | `create_perframe.py` | line 63 | **Medium** | Incomplete JAABA units mapping: only 7 quantity types covered; most features export as `("unit", None)`. |

### Potential Bugs / Logic Issues

| # | File | Description |
|---|---|---|
| 10 | `locomotion.py:48` | `center_of_rotation2` determinant formula — needs verification against original MATLAB. *(Later determined: **not a bug**; formula is correct.)* |
| 12 | `social.py` | Four inconsistent "closest fly" definitions across features with no documentation. |

### Performance Issues

| # | File | Description |
|---|---|---|
| 9 | `locomotion.py:150` | Per-fly Python loops in `compute_corfrac` / `_cor_displacement`. |
| 13 | `social.py:83` | O(n_flies²) Python loops in social distance features. |

### Robustness Issues

| # | File | Description |
|---|---|---|
| 6 | `registry.py` | `validate_registry` doesn't catch enabled→disabled dependency chains at import time. |
| 7 | `registry.py` | `validate_registry` doesn't cross-check declared `intermediates` against function return keys. |
| 17 | `preprocessing.py:123` | `signed_angle` divides by norm with no zero-norm guard. |
| 19 | `create_perframe.py:142` | `n_frames`/`n_flies` inferred from first 2D dataset encountered — fragile traversal order dependency. |

### Documentation / Clarity Issues

| # | File | Description |
|---|---|---|
| 11 | `appearance.py` | `compute_ab` docstring calls `a_mm` "major semi-axis" but it is a quarter-length; note says "full major axis = 2*a_mm" which is also wrong. |
| 14 | `position.py:62` | Triple-duplicate `"+-pi = moving straight backward"` line in `compute_yaw` docstring. |
| 15 | `wing_movement.py:59` | `imbalancer[:-1] > imbalancel[:-1]` is `imbalancer > -imbalancer`, equivalent to `imbalancer > 0`. Unnecessarily indirect. |
| 16 | `preprocessing.py:120` | `signed_angle` docstring declares return shape `(n, 2)` but actual return is `(n,)`. |
| 20 | `trx_mat.py:83` | `movie_path` selected from unsorted glob with no uniqueness check. |
| 23 | `trx_mat.py:134` | `theta_mm = theta` — `_mm` suffix implies unit conversion that does not happen. |

### Structural Issues

| # | Description |
|---|---|
| 25 | `sys.path.insert(0, ...)` hack in Stage 1/2 scripts instead of proper package structure. |
| 26 | No `__init__.py` in `features/`, `perframe/`, `split_and_inference/`. |
| 27 | No `requirements.txt` or `pyproject.toml`. |
| 28 | `config.yaml` contains absolute paths to a specific machine. |

---

## 3. How to Fix Bug 8 (Tail Position)

The bug is that `sign` is applied **inside** `cos`/`sin` rather than **outside**. Since cosine is even, `cos(-θ) = cos(θ)` — so the tail x-component points the same direction as the nose.

### `_point_speed` (line 305–307)

```python
# Before
sign  = 1.0 if use_nose else -1.0
x_mm  = x_mm + 2 * np.cos(sign * theta) * a_mm
y_mm  = y_mm + 2 * np.sin(sign * theta) * a_mm

# After
sign  = 1.0 if use_nose else -1.0
x_mm  = x_mm + sign * 2 * np.cos(theta) * a_mm
y_mm  = y_mm + sign * 2 * np.sin(theta) * a_mm
```

### `_point_velocity` (line 232–234)

```python
# Before
if use_tail:
    a_mm = features["a_mm"]
    x_mm = x_mm + 2 * np.cos(-theta) * a_mm
    y_mm = y_mm + 2 * np.sin(-theta) * a_mm

# After
if use_tail:
    a_mm = features["a_mm"]
    x_mm = x_mm - 2 * np.cos(theta) * a_mm
    y_mm = y_mm - 2 * np.sin(theta) * a_mm
```

### Why

| Point | Correct formula | What the bug computed |
|---|---|---|
| Nose | `+ 2a·cos(θ), + 2a·sin(θ)` | `+ 2a·cos(θ), + 2a·sin(θ)` ✓ |
| Tail | `- 2a·cos(θ), - 2a·sin(θ)` | `+ 2a·cos(θ), - 2a·sin(θ)` ✗ |

Verify against `compute_nose_tail` in `appearance.py`, which correctly uses `- 2 * a_mm * np.cos(theta)` for the tail x.

---

## 4. Improvement Plan

### Tier 1 — Quick Wins
*Single-line or near-single-line fixes. Do these first — several are silent data errors.*

| Priority | Location | Fix |
|---|---|---|
| 1 | `io_utils.py:33` | `tracks[:last_fidx]` → `tracks[:last_fidx + 1]` |
| 2 | `registry.py:151` | Typo `"cte_ind"` → `"ctr_ind"` |
| 3 | `locomotion.py:233,306` | Tail position sign error (Bug 8) |
| 4 | `registry.py:224` | Fix intermediate key `"x_nose_mm"` → `"nose_x_mm"` |
| 5 | `dataset.py:174` | Move shape validation before destructuring |
| 6 | `trx_mat.py:149` | Remove duplicate `"firstframe"` key |
| 7 | `create_perframe.py:102` | Collapse dead if/else into one line |
| 8 | `position.py:62` | Remove triple-duplicated docstring line |
| 9 | `wing_movement.py:59` | `imbalancer[:-1] > imbalancel[:-1]` → `imbalancer[:-1] > 0` |
| 10 | `dataset.py:91` | Move `import warnings` to module top |

### Tier 2 — Medium Refactors

| Item | Description |
|---|---|
| 2a | Expand `jaaba_units_from_h5_dataset` mapping (7 → full coverage of all registry quantity strings) |
| 2b | Fix `n_frames`/`n_flies` inference in `export_perframe` — read from `/meta` instead of first 2D dataset |
| 2c | Add `signed_angle` zero-norm guard |
| 2d | Extend `validate_registry` to catch disabled→enabled dependency chains at import time |
| 2e | Sort `movie_path` selection and raise on ambiguity in `trx_mat.py` |
| 2f | Add `parse_selection` out-of-range warning in `run_sleap_inference.py` |
| 2g | Add `requirements.txt` |

### Tier 3 — Larger Architectural Improvements

| Item | Description |
|---|---|
| 3a | Add `__init__.py` to all sub-packages; remove `sys.path.insert` hacks |
| 3b | Replace `print()` with `logging` throughout |
| 3c | Write unit tests: `io_utils` round-trip, tail velocity cross-check, ellipse geometry, `safe_savemat` atomicity |
| 3d | Verify `center_of_rotation2` determinant against original MATLAB |
| 3e | Vectorize pairwise social features (only if n_flies > 2 becomes relevant) |
| 3f | Add `config.example.yaml`; add `config.yaml` to `.gitignore` |

### Recommended Order

```
Week 1 — All Tier 1 bugs (silent data errors, ship first)
Week 2 — Tier 2a–2d (correctness: units, robustness)
Week 3 — Tier 2g + Tier 3a + 3f (infrastructure, enables tooling)
Week 4 — Tier 3c (tests, run against Week 1 fixes)
Later  — Tier 3b, 3d, 3e
```

### Suggested Tooling

| Tool | Purpose |
|---|---|
| **pytest** | Unit tests for feature correctness |
| **ruff** | Fast linting + formatting |
| **mypy** | Static type checking (incremental) |
| **numpy.testing** | `assert_allclose` for array comparisons in tests |
| **cProfile + snakeviz** | Profile batch run before optimizing social features |

---

## 5. Fix Status After Manual Fixes

After the improvement plan was produced, several bugs were fixed manually. The following table records the state of each confirmed bug after the manual fixes:

| Bug | Status | Notes |
|---|---|---|
| **B1** Off-by-one in `load_tracks` | ✓ Fixed | `last_fidx + 1` |
| **B2** Typo `"cte_ind"` | ✓ Fixed | `"ctr_ind"` |
| **B3** Tail position in `_point_speed` | ✓ Fixed | Sign outside `cos`/`sin` |
| **B3** Tail position in `_point_velocity` | ⚠ Partially fixed — new problem introduced | Sign applied to whole expression but `sin(-θ) = -sin(θ)` meant y-component was now wrong |
| **B4** Intermediate key `"x_nose_mm"` | ✓ Fixed | `"nose_x_mm"` |
| **B5** Validation order + error message | ✓ Fixed (with poor error message) | Check before destructuring, but message said "not enough values to unpack" |
| **B6** Duplicate `"firstframe"` key | ✓ Fixed | |
| **B7** Dead if/else in `jaaba_data_cell` | ✓ Fixed (commented out) | |
| **B8** Incomplete units mapping | ✗ Still open | 7 entries, no additions |

Also fixed during this pass (non-bug items):

- `position.py` — triple-duplicate docstring line removed
- `wing_movement.py:59` — `imbalancer[:-1] > imbalancel[:-1]` → `imbalancer[:-1] > 0`
- `preprocessing.py` — `signed_angle` docstring return shape corrected to `(n,)`
- `preprocessing.py` — zero-norm guard added to `signed_angle`

---

## 6. Remaining Fixes Applied

After the status re-scan, two bugs and one incomplete mapping remained. All were resolved:

### B3 — `_point_velocity` tail y-component (the incomplete fix)

The manual fix had applied a leading minus to the whole expression, which correctly fixed x but broke y because `sin(-θ) = -sin(θ)`.

```python
# Broken intermediate state
x_mm = x_mm - 2 * np.cos(-theta) * a_mm   # x ok: -cos(-θ) = -cos(θ) ✓
y_mm = y_mm - 2 * np.sin(-theta) * a_mm   # y wrong: -sin(-θ) = +sin(θ) ✗

# Correct final state (verified in file)
x_mm = x_mm - 2 * np.cos(theta) * a_mm   # ✓
y_mm = y_mm - 2 * np.sin(theta) * a_mm   # ✓
```

### B5 — `dataset.py` error message

```python
# Before
if tracks.ndim != 4:
    raise ValueError("ValueError: not enough values to unpack")

# After
if tracks.ndim != 4:
    raise ValueError(
        f"Expected tracks with shape (time, joints, 2, fly). Got: {tracks.shape}"
    )
```

### B8 — JAABA units mapping expanded

`create_perframe.py` mapping extended from 7 to 42 entries covering all quantity strings in the registry:

| Group | Quantity strings added |
|---|---|
| Position / distance | `position`, `distance_change_rate`, `area`, `area_change_rate` |
| Orientation / angle | `orientation`, `velocity_direction`, `sideways_angle`, `yaw_angle`, `absolute_yaw_angle` |
| Angular rates | `angular_velocity`, `angular_speed`, `angle_change_rate`, `velocity_direction_change_rate` |
| Linear speeds | `speed`, all `*_velocity_cor/ctr/tail` variants, `a_change_rate`, `b_change_rate` |
| Dimensionless | `eccentricity`, `eccentricity_change_rate`, `fractional_offset`, `turn_sign`, `count`, `index` |

---

## 7. Final Review — Items 6, 7, 9, 10, 11, 17, 19, 20, 23

After all prior fixes, a targeted re-check was performed on nine specific review items.

### Item 6 — `validate_registry` doesn't catch disabled→enabled dependency chains

**Still relevant — robustness.**  
**Fixed.**

Added an `elif` branch inside the existing Check 1 loop:

```python
# Before
for dep in spec.requires:
    if dep not in registry:
        errors.append(...)

# After
for dep in spec.requires:
    if dep not in registry:
        errors.append(...)
    elif spec.enabled and not registry[dep].enabled:
        errors.append(
            f"  '{name}' (enabled) requires '{dep}', which is disabled. "
            f"Either disable '{name}' or re-enable '{dep}'."
        )
```

Any enabled feature that directly requires a disabled one now raises at import time instead of mid-run.

---

### Item 7 — Validator can't cross-check `intermediates` against function return keys

**Still relevant — robustness.**  
**Left unchanged.**

Properly verifying that declared `intermediates` match actual function return keys requires running the functions (no static solution). The known mismatch (B4) was fixed manually. This item needs test coverage, not a validator change.

---

### Item 9 — Per-fly Python loops in `compute_corfrac` / `_cor_displacement`

**Still relevant — performance.**  
**Left unchanged.**

Correct for 2 flies; vectorizing is a moderate refactor with regression risk. Defer until n_flies > 2 is a real requirement.

---

### Item 10 — `center_of_rotation2` determinant formula

**Not a bug.**  
**No change needed.**

Working backward from the inverse matrix entries in the same function:

- `m11 = dbcost/Z`, `m12 = -dasint/Z`, `m21 = dbsint/Z`, `m22 = dacost/Z`
- These imply M = `[[dacost, dasint], [-dbsint, dbcost]]`
- `det(M) = dacost·dbcost − dasint·(−dbsint) = dacost·dbcost + dasint·dbsint`
- The code's `Z = dacost·dbcost + dbsint·dasint` is algebraically identical ✓

---

### Item 11 — `compute_ab` docstring calls `a_mm` "major semi-axis"

**Still relevant — documentation.**  
**Fixed.**

`a_mm = head_abdomen_dist / 4 / pxpermm`. The ellipse semi-major axis is `2*a_mm`, not `a_mm`. The previous note "full major axis = 2*a_mm" was also wrong (full axis = 4*a_mm).

```python
# Before
a_mm : major semi-axis (mm)
    Half the distance from head to abdomen, divided by 2.
Note: the full major axis = 2*a_mm, full minor axis = 2*b_mm.

# After
a_mm : quarter body length (mm)
    One quarter of the head-to-abdomen distance (i.e. distance / 4 / pxpermm).
    The body ellipse semi-major axis = 2*a_mm.
Note: semi-major axis = 2*a_mm, semi-minor axis = 2*b_mm,
      full major axis = 4*a_mm, full minor axis = 4*b_mm.
```

---

### Item 17 — `signed_angle` zero-norm guard

**Already fixed (manually).**  
**No change needed.**

Current code:
```python
norm_a = np.linalg.norm(a, axis=1, keepdims=True)
norm_b = np.linalg.norm(b, axis=1, keepdims=True)
a = np.where(norm_a > 0, a / norm_a, 0.0)
b = np.where(norm_b > 0, b / norm_b, 0.0)
```

---

### Item 19 — `n_frames`/`n_flies` inferred from first 2D dataset

**Still relevant — robustness.**  
**Left unchanged.**

Given the known HDF5 structure (meta = 0D/1D scalars, pose = 3D arrays, feature datasets = 2D at root level), traversal order is safe in practice. A proper fix requires writing `n_frames`/`n_flies` to `/meta` in `dataset.py` and reading them in `create_perframe.py` — a two-file change deferred to a future session.

---

### Item 20 — `movie_path` selected without sorting

**Already fixed (manually).**  
**No change needed.**

Current `trx_mat.py`:
```python
movie_files = sorted(expt_dir.glob("movie.*"))
if len(movie_files) > 1:
    raise ValueError(f"Multiple movie files found in {expt_dir}: {movie_files}")
movie_path = movie_files[0] if movie_files else None
```

---

### Item 23 — `theta_mm = theta` misleading `_mm` suffix

**Still relevant — clarity.**  
**Fixed.**

```python
# Before
theta_mm = theta

# After
theta_mm = theta  # heading in radians; no unit conversion (angles are scale-invariant)
```

---

### Summary of Final Pass

| Item | Verdict | Action |
|---|---|---|
| 6 — Disabled dependency not caught at import | Robustness | **Fixed** — added check to `validate_registry` |
| 7 — Intermediates not validated against return keys | Robustness | Left unchanged — needs test coverage |
| 9 — Per-fly Python loops | Performance | Left unchanged — fine for 2 flies |
| 10 — CoR determinant formula | **Not a bug** | No change |
| 11 — `compute_ab` docstring incorrect | Documentation | **Fixed** — corrected terminology and note |
| 17 — `signed_angle` zero-norm guard | Already fixed | No change |
| 19 — Fragile `n_frames`/`n_flies` inference | Robustness | Left unchanged — deferred |
| 20 — Unsorted `movie_path` selection | Already fixed | No change |
| 23 — `theta_mm = theta` misleading | Clarity | **Fixed** — added inline comment |

### Files Modified in Final Pass

- `features/registry.py` — enabled→disabled dependency check in `validate_registry`
- `features/appearance.py` — corrected `compute_ab` docstring
- `perframe/trx_mat.py` — clarifying comment on `theta_mm`

### Remaining Open Items

- **Item 7** — Intermediate key validation requires test coverage.
- **Item 9** — Performance improvement deferred.
- **Item 19** — Fragile shape inference deferred pending `/meta` schema addition.

---

*End of review session.*
