# CLAUDE.md — SleapCode

## 1. Project Context

### What This Project Does
SleapCode is a post-processing pipeline for Drosophila behavioral experiments. It takes pose-tracking output from [SLEAP](https://sleap.ai/) and computes 40+ behavioral features per fly per frame, then exports them to HDF5 and MATLAB formats for downstream classification in JAABA.

### Pipeline Stages
| Stage | Script | Input → Output |
|---|---|---|
| 1 — Split arenas | `split_and_inference/split_experiment_to_arenas.py` | Multi-arena recording → per-arena folders |
| 2 — Inference | `split_and_inference/run_sleap_inference.py` | Arena video → `.slp` → `.analysis.h5` |
| 3 — Features | `run_pipeline.py` | `.analysis.h5` → `.features.h5` + `perframe/*.mat` + `trx.mat` |

Stage 3 is the primary focus of this codebase.

### Key Architectural Components

**`params.py`** — Loads `config.yaml` once at import time. Exposes calibration constants (`FPS`, `PXPERMM`) and skeleton node indices (`NODE_IDX_HEAD`, `NODE_IDX_THORAX`, etc.) used everywhere. Config changes during a session require restarting the interpreter.

**`features/registry.py`** — Central registry of all features as `FeatureSpec` dataclasses. Each entry declares its computation function, dependencies (`requires`), output keys, save mode, and units metadata. `compute_item()` resolves dependencies recursively and memoizes results in a `computed` dict. `validate_registry()` runs at import time and catches structural errors.

**`dataset.py`** — Orchestrates feature extraction for one experiment. Calls `compute_item()` for every enabled feature, then writes the results to an HDF5 file. All feature computation flows through here.

**`features/`** — Seven modules (`appearance`, `locomotion`, `position`, `social`, `wing_appearance`, `wing_movement`) containing the actual computation functions. Each function receives `(tracks, features, **kwargs)` where `features` is the shared `computed` dict.

**`perframe/`** — JAABA export: `create_perframe.py` writes scalar features as `.mat` files; `trx_mat.py` writes the trajectory struct; `ds_utils.py` provides atomic file writes and HDF5 iteration.

**`io_utils.py`** — Loads SLEAP `.analysis.h5` files into NumPy arrays.

**`preprocessing.py`** — Shared signal processing utilities: `fill_missing`, `normalize_to_egocentric`, `signed_angle`.

### Data Shape Convention
Tracks array shape throughout the pipeline: `(n_frames, n_nodes, 2, n_flies)`.
Scalar feature arrays: `(n_frames, n_flies)`.
These shapes are assumed but not always validated. Do not break them silently.

### Ellipse Axis Convention
`a_mm` is a **quarter-length** (= head-to-abdomen distance / 4 / pxpermm), not a standard semi-axis. The semi-major axis is `2 * a_mm`. This is consistent internally but non-standard. Do not normalize it without understanding all downstream uses.

---

## 2. Development Principles

- **Correctness over performance.** Silent data errors are worse than slow code. Fix bugs before optimizing.
- **Avoid breaking existing behavior.** Any change to a feature computation function affects every experiment that uses it. Consider downstream JAABA classifiers trained on existing data.
- **Fix issues incrementally.** One issue per change. Do not bundle unrelated fixes.
- **Do not refactor broadly unless explicitly asked.** Renaming, restructuring, or abstracting beyond the immediate issue is not permitted without a specific request.
- **Respect the dependency graph.** Features depend on other features via `requires`. Changing a foundational feature (e.g., `theta`, `xy_mm`, `body_scale`) has cascading effects on everything that depends on it.
- **Preserve NaN conventions.** Frame 0 of any differential feature is NaN by convention (no prior frame). Do not fill or shift this.

---

## 3. Code Guidelines

### Making Changes
- Read the function and its registry entry before suggesting any edit.
- Show only the lines being changed, not the whole file.
- Explain what the change does and why, and note any risk.
- If a fix touches a foundational feature (`theta`, `xy_mm`, `a_mm`, `b_mm`), flag the downstream impact explicitly before proceeding.

### Suggesting Fixes
- Prefer minimal edits: change one expression, not the whole function.
- Do not add logging, type annotations, or docstrings to code you are not otherwise modifying.
- Do not introduce new dependencies unless necessary.
- Do not add error handling for cases that cannot occur given the validated inputs.

### Registry Changes
- Every `FeatureSpec` entry must keep `outputs`, `intermediates`, and the function's actual return dict keys in sync. The validator does not check this automatically.
- `params` keys in a `FeatureSpec` must exactly match the argument names in the corresponding function. A typo here silently ignores the config value (see Bug: `"cte_ind"` / `"ctr_ind"`).
- Do not change `enabled=False` on a feature without checking which other features depend on it. A disabled dependency raises at runtime, not at import time.

### Tests
- When fixing a confirmed bug, note what a regression test for that fix would look like, even if tests are not being written yet.

---

## 4. Known Issues and Risks

These issues were identified during code review. Do not re-introduce them, and be aware of them when working nearby.

### Confirmed Bugs (not yet fixed)

| # | Location | Description |
|---|---|---|
| B1 | `io_utils.py:33` | **Off-by-one**: `tracks[:last_fidx]` silently drops the last valid frame. Fix: `tracks[:last_fidx + 1]`. |
| B2 | `registry.py:151` | **Typo** `"cte_ind"` instead of `"ctr_ind"` in `xy_mm` params. `NODE_IDX_THORAX` is never applied; `x_mm`/`y_mm` always use node index 1. |
| B3 | `locomotion.py:233,306` | **Tail position sign error**: `cos(-θ) = cos(θ)`, so the tail x-position is computed in the wrong direction. Affects `du_tail`, `dv_tail`, `velmag_tail`. Fix: multiply sign outside `cos`/`sin`, not inside. |
| B4 | `registry.py:224` | **Intermediate key mismatch**: `nose_tail_mm` declares `"x_nose_mm"` but `compute_nose_tail` returns `"nose_x_mm"`. No runtime crash, but the declaration is wrong. |
| B5 | `dataset.py:174` | **Validation after destructuring**: shape check runs after `tracks.shape` is unpacked, so a bad shape raises an obscure error, not the descriptive one. |
| B6 | `trx_mat.py:149` | **Duplicate dict key** `"firstframe"`. Harmless (same value), but a copy-paste error. |
| B7 | `create_perframe.py:102` | **Dead code**: both branches of `if/else` in `jaaba_data_cell` are identical. |
| B8 | `create_perframe.py:63` | **Incomplete units mapping**: most feature quantity strings (`"speed"`, `"angular_velocity"`, `"orientation"`, etc.) are not in the JAABA units map and silently export as `("unit", None)`. |

### Structural Risks

- **No `__init__.py`** in `features/`, `perframe/`, `split_and_inference/`. Imports depend on the CWD being the project root. Stage 1/2 scripts use `sys.path.insert` as a workaround.
- **No `requirements.txt`**. Dependencies must be inferred from imports.
- **`config.yaml` contains absolute paths** to a specific machine (`D:/`, `W:/`). Will not run on another machine without editing.
- **`params.py` loads at import time**. Config edits during a session are invisible until the interpreter restarts.
- **Feature output shapes are not validated** before writing to HDF5. A function returning `(n_frames, n_flies, 2)` instead of `(n_frames, n_flies)` silently produces a malformed file.
- **`center_of_rotation2` determinant** (`locomotion.py:48`): the formula `Z = dacost*dbcost + dbsint*dasint` has not been verified against the original MATLAB source. If wrong, `corfrac`, `dv_cor`, `du_cor`, `velmag`, and `flipdv_cor` are all affected.
- **Four different "closest fly" definitions** (`closestfly_ell2nose`, `closestfly_nose2ell`, `closestfly_anglesub`, `closestfly_center`). Derived features must use the correct index for their context. Mixing them is a silent logical error.

---

## 5. Review and Fixing Workflow

When asked to fix an issue:

1. **Read** the relevant file(s) first. Do not suggest a fix from memory.
2. **Identify** the exact lines involved and confirm the bug is present in the current code.
3. **Check dependencies**: does the change affect any downstream feature in the registry? Does it affect `trx_mat.py` or `create_perframe.py`?
4. **Propose** the fix with a before/after diff and a plain-language explanation of what changed and why.
5. **Wait for confirmation** before applying the fix.
6. **Apply** the fix to exactly the lines discussed. Do not make additional changes in the same edit.
7. **Note** what a regression test for this fix would check.

Do not move on to the next issue until the current one is confirmed correct.

---

## 6. Output Style

- Be concise. Lead with the answer or action.
- Use before/after code blocks for any proposed change.
- Use tables for structured comparisons (e.g., affected features, bug lists).
- Do not summarize what you just did at the end of a response.
- Do not add preamble ("Great question!", "Sure, let me...").
- Flag risk explicitly when a change touches foundational features or shared utilities.
