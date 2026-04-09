# Updated Project Review

**Date:** 2026-04-09  
**Based on:** Stage 2 code review + Stage 3 improvement plan  
**Method:** Full re-scan of current codebase; all statuses verified against live files

---

## Summary

| Metric | Count |
|---|---|
| Total issues tracked | 29 |
| Resolved | 18 |
| Still open | 10 |
| Partially resolved | 1 |
| Not a bug (closed after verification) | 1 |

---

## Detailed Status of Issues

Issues use the original identifiers from the Stage 2 review. Bugs are prefixed **B**, other issues are numbered.

---

### Confirmed Bugs

#### B1 — Off-by-one in `load_tracks`
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `io_utils.py:35` now reads `tracks[:last_fidx + 1]`. Last valid frame is correctly included.

---

#### B2 — Typo `"cte_ind"` in `xy_mm` registry params
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `registry.py:158` now correctly passes `"ctr_ind": NODE_IDX_THORAX`. The thorax node index from config is properly applied to `compute_xy`.

---

#### B3 — Tail position sign error (`_point_velocity` and `_point_speed`)
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `_point_velocity` (lines 235–236): `x_mm - 2 * np.cos(theta) * a_mm`, `y_mm - 2 * np.sin(theta) * a_mm` ✓  
- `_point_speed` (lines 308–309): `sign * 2 * np.cos(theta)`, `sign * 2 * np.sin(theta)` ✓  
- Both functions now agree with `compute_nose_tail`.

---

#### B4 — Intermediate key mismatch in `nose_tail_mm`
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `registry.py:226` now declares `intermediates=["nose_x_mm", "nose_y_mm", "tail_x_mm", "tail_y_mm"]`, matching the function return keys.

---

#### B5 — Shape validation after destructuring in `dataset.py`
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `dataset.py:178–181`: `ndim` check now precedes the unpacking, with a descriptive error message.

---

#### B6 — Duplicate dict key `"firstframe"` in `trx_mat.py`
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `trx_mat.py:152`: only one `"firstframe"` key present.

---

#### B7 — Dead if/else branches in `jaaba_data_cell`
- **Category:** Bug  
- **Status:** ⚠️ Partially resolved  
- The active code is correct (single `v_out = v.astype(np.float64, copy=False)` on line 140).  
- The original dead branches are still present as commented-out code (lines 141–144). Functionally harmless, but adds clutter.

---

#### B8 — Incomplete JAABA units mapping
- **Category:** Bug  
- **Status:** ✅ Resolved  
- `create_perframe.py:65–109`: mapping expanded from 7 to 42 entries, covering all quantity strings in the registry. Every feature now exports correct JAABA units.

---

### Robustness Issues

#### Issue 6 — `validate_registry` didn't catch enabled→disabled dependency chains
- **Category:** Robustness  
- **Status:** ✅ Resolved  
- `registry.py:81–85`: `elif spec.enabled and not registry[dep].enabled` check added inside the requires loop. Import-time detection now works.

---

#### Issue 7 — `validate_registry` can't cross-check `intermediates` against function return keys
- **Category:** Robustness  
- **Status:** 🔴 Still relevant  
- No static fix is possible without running the functions. The known mismatch (B4) was fixed manually, but future mismatches remain undetectable at registry load time. Requires test coverage to close properly.

---

#### Issue 13 / 17 — `signed_angle` zero-norm guard
- **Category:** Robustness  
- **Status:** ✅ Resolved  
- `preprocessing.py:125–129`: safe-divide guard added using `np.where(norm > 0, ...)`.

---

#### Issue 19 — Fragile `n_frames`/`n_flies` inference in `export_perframe`
- **Category:** Robustness  
- **Status:** 🔴 Still relevant  
- `create_perframe.py:178–184`: shape is still inferred from the first 2D dataset encountered during HDF5 traversal. This works reliably for the current file structure (meta = scalars, pose = 3D, features = 2D at root), but would silently break if a 2D metadata dataset were introduced. Fix requires writing `n_frames`/`n_flies` to `/meta` in `dataset.py` — a two-file change, deferred.

---

#### Issue 17 (run_pipeline) — `str.replace` used for path construction
- **Category:** Robustness  
- **Status:** 🔴 Still relevant  
- `run_pipeline.py:58`: `analysis_path.replace(".analysis.h5", ".features.h5")` operates on the full path string. If any parent directory name contains `.analysis.h5`, this would silently corrupt the output path.  
- **Low probability but easy to fix:** use `Path` operations instead.

---

### Performance Issues

#### Issue 9 — Per-fly Python loops in `compute_corfrac` / `_cor_displacement`
- **Category:** Performance  
- **Status:** 🔴 Still relevant (deferred)  
- `locomotion.py:150–157` and `164–190`: one Python loop per fly, each calling `center_of_rotation2` / `_rfrac2center`. Acceptable for 2 flies. Vectorizing is a moderate refactor; deferred until n_flies > 2 is needed.

---

#### Issue 12 — O(n_flies²) Python loops in social features
- **Category:** Performance  
- **Status:** 🔴 Still relevant (deferred)  
- `social.py`: `compute_dell2nose`, `compute_dnose2ell`, `compute_anglesub` each nest two fly loops over T frames with ellipse sampling. Fast enough for 2 flies. Deferred.

---

#### Issue 15 — `get_units_for_key` linear scan on every call
- **Category:** Performance  
- **Status:** 🔴 Still relevant (low priority)  
- `registry.py:821–826`: scans all specs for every output key. O(n_specs) per key. Negligible at 40 features; only matters if registry grows significantly.

---

### Structural Issues

#### Issue 21 — `sys.path.insert` hacks in Stage 1/2 scripts
- **Category:** Structure  
- **Status:** 🔴 Still relevant  
- `split_and_inference/split_experiment_to_arenas.py:10` and `run_sleap_inference.py:5`: both insert the parent directory into `sys.path` to import `params`. Fragile; breaks linting, IDE analysis, and `pytest` discovery. Requires `__init__.py` files to resolve properly.

---

#### Issue 26 — No `__init__.py` in `features/`, `perframe/`, `split_and_inference/`
- **Category:** Structure  
- **Status:** 🔴 Still relevant  
- Verified: no `__init__.py` files exist anywhere in the project. Imports work only because `run_pipeline.py` is executed from the project root. Tools like `pytest`, `mypy`, and IDE refactoring do not work correctly.

---

#### Issue 27 — No `requirements.txt` or `pyproject.toml`
- **Category:** Structure  
- **Status:** 🔴 Still relevant  
- Verified: neither file exists. Dependencies (`numpy`, `h5py`, `pandas`, `scipy`, `pyyaml`, `tkfilebrowser`) must be inferred from source.

---

#### Issue 28 — `config.yaml` contains hardcoded absolute paths
- **Category:** Structure  
- **Status:** 🔴 Still relevant  
- `config.yaml:3–6`: `D:/Galit'sLab Dropbox/...` and `W:/Rotem/...` paths are machine-specific. The project will not run on any other machine without manual editing. No `config.example.yaml` exists as a template.

---

#### Issue 29 — `params.py` loads config at import time
- **Category:** Structure  
- **Status:** 🔴 Still relevant (low priority / by design)  
- Config edits during a session are invisible until the interpreter restarts. Acceptable for current usage pattern but worth noting for future CLI or test harness development.

---

### Documentation / Clarity Issues

#### Issue 10 — `center_of_rotation2` determinant formula
- **Category:** Documentation  
- **Status:** ✅ Not a bug — verified  
- Working backward from the inverse matrix entries (`m11=dbcost/Z`, `m12=-dasint/Z`, `m21=dbsint/Z`, `m22=dacost/Z`), the original matrix is `[[dacost, dasint], [-dbsint, dbcost]]` and `det = dacost·dbcost + dasint·dbsint`. The code's formula `Z = dacost*dbcost + dbsint*dasint` is algebraically identical. No change needed.

---

#### Issue 11 — `compute_ab` docstring called `a_mm` a "major semi-axis"
- **Category:** Documentation  
- **Status:** ✅ Resolved  
- `appearance.py:16–30`: docstring now correctly describes `a_mm` as "quarter body length", states `semi-major axis = 2*a_mm`, and corrects the note to `full major axis = 4*a_mm`.

---

#### Issue 14 — `signed_angle` docstring incorrect return shape
- **Category:** Documentation  
- **Status:** ✅ Resolved  
- `preprocessing.py:120`: docstring now correctly states return shape `(n, )`.

---

#### Issue 16 — `import warnings` inside function body in `dataset.py`
- **Category:** Documentation / Style  
- **Status:** ✅ Resolved  
- `dataset.py:9`: `import warnings as _warnings` moved to module top.

---

#### Issue 22 — `movie_path` selection without sorting or uniqueness check
- **Category:** Documentation / Robustness  
- **Status:** ✅ Resolved  
- `trx_mat.py:85–88`: `sorted()` added; raises `ValueError` if more than one `movie.*` file is found.

---

#### Issue 23 — `theta_mm = theta` misleading `_mm` suffix
- **Category:** Documentation  
- **Status:** ✅ Resolved  
- `trx_mat.py:138`: clarifying comment added — `# heading in radians; no unit conversion (angles are scale-invariant)`.

---

#### Issue 24 — Redundant comparison in `compute_dwing_angle_imbalance`
- **Category:** Documentation / Clarity  
- **Status:** ✅ Resolved  
- `wing_movement.py:62`: `imbalancer[:-1] > imbalancel[:-1]` replaced with the equivalent `imbalancer[:-1] > 0`. Old line preserved as comment.

---

#### Issue 25 — Triple-duplicate docstring line in `compute_yaw`
- **Category:** Documentation  
- **Status:** ✅ Resolved  
- `position.py:62`: only one `"+-pi = moving straight backward"` line remains.

---

#### Issue 18 — `features=None` default never guarded in feature functions
- **Category:** Documentation / Robustness  
- **Status:** 🔴 Still relevant (low priority)  
- All feature functions accept `features=None` but immediately subscript `features[...]`. The default is misleading — it implies the argument is optional when it is required. No change has been made.

---

#### Issue 23b — `parse_selection` silently drops out-of-range arena indices
- **Category:** Documentation / UX  
- **Status:** 🔴 Still relevant  
- `run_sleap_inference.py:73`: indices outside `[1, max_index]` are silently discarded with no user feedback. Easy to add a warning.

---

#### Issue B7 dead code — Commented-out if/else in `jaaba_data_cell`
- **Category:** Documentation / Clarity  
- **Status:** ⚠️ Partially resolved  
- The fix collapsed the active logic to one line (correct), but the original dead branches remain as commented-out code (lines 141–144). No functional impact; minor clutter.

---

## Remaining Relevant Issues (Prioritized)

Issues that still matter, ordered by importance:

| Priority | ID | Category | Description |
|---|---|---|---|
| 1 | Issue 17 (run_pipeline) | Robustness | `str.replace` for path construction — potential silent path corruption |
| 2 | Issue 7 | Robustness | `validate_registry` cannot detect `intermediates` key mismatches — needs test coverage |
| 3 | Issue 19 | Robustness | `n_frames`/`n_flies` inferred from first 2D HDF5 dataset — fragile traversal order |
| 4 | Issue 26 | Structure | No `__init__.py` — not proper packages; breaks tools and pytest |
| 5 | Issue 27 | Structure | No `requirements.txt` or `pyproject.toml` — no reproducible install path |
| 6 | Issue 28 | Structure | `config.yaml` has hardcoded machine-specific absolute paths |
| 7 | Issue 21 | Structure | `sys.path.insert` hacks in Stage 1/2 scripts |
| 8 | Issue 23b | Clarity / UX | `parse_selection` drops out-of-range entries silently |
| 9 | Issue 9 | Performance | Per-fly Python loops in `compute_corfrac` / `_cor_displacement` |
| 10 | Issue 12 | Performance | O(n_flies²) Python loops in pairwise social features |
| 11 | Issue 18 | Clarity | `features=None` default is misleading — argument is actually required |
| 12 | Issue 15 | Performance | `get_units_for_key` linear scan (negligible at current scale) |
| 13 | Issue 29 | Structure | `params.py` loads at import time — config edits invisible during session |

---

## Notes

### Risks from Recent Fixes

- **B3 partial fix history:** `_point_velocity` went through an incorrect intermediate state during manual fixing (`x_mm - 2 * np.cos(-theta)`) where the x-component was fixed but the y-component was broken. The final state is correct (`cos(theta)` / `sin(theta)` without negation), but if any cached outputs were written during the intermediate state, they contain incorrect `du_tail` and `dv_tail` values. Any `.features.h5` files produced before this fix should be regenerated.

- **B8 units mapping:** The `"other"` fallback entry remains in the mapping. Any future registry quantity string that is not explicitly listed will silently export as `("unit", None)`. The mapping should be kept in sync with the registry as new features are added.

- **Issue 6 validator:** The new enabled→enabled check only detects **direct** dependency violations (A requires B, B is disabled). Transitive violations (A requires B (enabled), B requires C (disabled)) are still caught only at compute time, not at import time. Fully transitive checking would require a topological pass.

- **Dead code in B7:** The commented-out if/else block in `jaaba_data_cell` (lines 141–144) is harmless but should be removed in a cleanup pass to avoid confusion about intent.

### Items That Will Not Be Fixed Here

- **Issue 29** (params.py import-time loading): This is a design choice appropriate for the current usage pattern. Changing it would require a larger refactor of how constants are accessed.  
- **Issue 12 / 9** (performance loops): Both are deferred until experiments with more than 2 flies become a requirement.
