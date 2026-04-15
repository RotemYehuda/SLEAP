# SleapCode

A Python pipeline that converts [SLEAP](https://sleap.ai) pose-tracking outputs into
behavioral features and exports them in formats compatible with
[JAABA](https://jaaba.sourceforge.net) (a MATLAB-based behavior classifier).
---
  `conda create -n sleapcode python=3.9`  
  `conda activate sleapcode`  
  `pip install numpy h5py scipy pandas pyyaml tkfilebrowser`

OR

`pip install -r requirements.txt`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [End-to-End Workflow](#end-to-end-workflow)
3. [Project Structure](#project-structure)
4. [Installation and Dependencies](#installation-and-dependencies)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Feature System Architecture](#feature-system-architecture)
8. [Output Saving Behavior](#output-saving-behavior)
9. [Adding a New Feature](#adding-a-new-feature)
10. [Current Feature Registry](#current-feature-registry)
11. [Perframe Export and JAABA Compatibility](#perframe-export-and-jaaba-compatibility)
12. [Output Files](#output-files)

---

## Project Overview

This project solves the problem of going from raw multi-fly video recordings to structured, quantitative behavioral features that can be fed into a classifier such as JAABA.

The pipeline covers three stages:

1. **Preprocessing** splitting a raw multi-arena experiment recording into per-arena folders and running SLEAP pose estimation inference.
2. **Feature extraction** loading proofread SLEAP tracks and computing a library of kinematic and pairwise features per fly per frame.
3. **Export** writing features to an HDF5 file and converting them to JAABA-compatible MATLAB `.mat` files (`perframe/` directory and `trx.mat`).

---

## End-to-End Workflow

The end-to-end workflow has three independent stages, each invoked separately:

```
Raw multi-arena experiment folder
        ↓  split_and_inference/split_experiment_to_arenas.py
Per-arena folders  (movie.avi + metadata)
        ↓  split_and_inference/run_sleap_inference.py   [separate sleap conda env]
*.analysis.h5  (SLEAP pose tracks)
        ↓  run_pipeline.py
        ├── dataset.py          →  *.features.h5
        ├── create_perframe.py  →  perframe/*.mat
        └── trx_mat.py          →  trx.mat
```

The final outputs (`*.features.h5`, `perframe/`, `trx.mat`) are ready for import into
JAABA for behavioral classification.

---

## Project Structure


```
SleapCode/
├── config.yaml                        # All user configuration (edit this)
├── params.py                          # Loads config.yaml; exposes constants
├── dataset.py                         # Core feature computation and HDF5 writing
├── io_utils.py                        # SLEAP HDF5 loading utilities
├── preprocessing.py                   # Signal processing utilities (fill_missing,
│                                      #   normalize_to_egocentric, signed_angle)
├── run_pipeline.py                    # Top-level batch runner (GUI entry point)
│
├── features/
│   ├── registry.py                    # FeatureSpec, REGISTRY, compute_item()
│   ├── ego.py                         # Egocentric coordinate features
│   ├── kinematics.py                  # Individual kinematics (speed, acceleration)
│   ├── pairwise.py                    # Nearest-neighbor pairwise features
│   └── wings.py                       # Wing angle features
│
├── perframe/
│   ├── create_perframe.py             # Export features to JAABA perframe/*.mat
│   ├── trx_mat.py                     # Export trajectory struct to trx.mat
│   └── ds_utils.py                    # HDF5 iteration; safe atomic file writing
│
└── split_and_inference/
    ├── split_experiment_to_arenas.py  # Split multi-arena recordings into per-arena folders
    └── run_sleap_inference.py         # Run SLEAP inference; convert .slp → .analysis.h5
```

---

## Installation and Dependencies

**Main pipeline** (`run_pipeline.py` and everything it calls):

```
python >= 3.10
numpy
h5py
pandas
scipy
pyyaml
tkinter          # GUI file picker (standard library)
tkfilebrowser    # pip install tkfilebrowser

pip install numpy h5py pandas scipy pyyaml tkfilebrowser
```

**SLEAP inference** (`split_and_inference/run_sleap_inference.py`):
Run in a separate `sleap` conda environment with SLEAP installed. This script
does not import from the rest of the codebase.

---
## Configuration

All user-facing settings live in `config.yaml` at the project root. The file is
loaded once at import time by `params.py`, which exposes named constants to the
rest of the codebase. **Do not hardcode paths or calibration values anywhere else.**

### `config.yaml` reference

```yaml
# ── Paths ──────────────────────────────────────────────────────────────────────
# Default directory opened by the GUI file picker.
base_path: "D:/path/to/your/data"

# Parent directory for per-arena analysis folders.
arena_parent_dir: "W:/path/to/analysisData"

# SLEAP centroid and instance model paths (used by run_sleap_inference.py).
centroid_model:  "W:/path/to/centroid_model"
instance_model:  "W:/path/to/instance_model"

# Name of the SLEAP output file produced by inference.
sleap_output_name: "inference.slp"

# ── Calibration ────────────────────────────────────────────────────────────────
fps:      30      # frames per second
pxpermm:  10.5   # pixels per millimeter

# ── Skeleton ───────────────────────────────────────────────────────────────────
# Node names in index order, exactly as they appear in the SLEAP skeleton.
# These are validated at the start of every pipeline run.
# If the skeleton changes, update this list — all node indices update automatically.
node_names_expected:
  - "head"         # index 0  ← fwd_ind
  - "thorax"       # index 1  ← ctr_ind
  - "abdomen"      # index 2
  - "L_wing"       # index 3
  - "R_wing"       # index 4
  - "L_frontLeg"   # index 5
  - "R_frontLeg"   # index 6
  - "L_midLeg"     # index 7
  - "R_midLeg"     # index 8
  - "L_hindLeg"    # index 9
  - "R_hindLeg"    # index 10
```
These constants are imported by `dataset.py`, `trx_mat.py`, and `features/registry.py`.
If you retrain SLEAP with a different skeleton node order, update `node_names_expected`
in `config.yaml` — all indices update automatically throughout the pipeline.

At the start of each pipeline run, `dataset.py` validates that the node names in the
loaded `.analysis.h5` file match `node_names_expected`. A mismatch raises a descriptive
`ValueError` before any computation begins.
---

## Running the Pipeline

### Stage 1 — Split arenas (if needed)

```bash
python split_and_inference/split_experiment_to_arenas.py
```

Takes a multi-arena experiment folder and produces per-arena subdirectories.

### Stage 2 — SLEAP inference (separate environment)

```bash
conda activate sleap
python split_and_inference/run_sleap_inference.py
```

Runs pose estimation and produces `*.analysis.h5` files. Configuration is read
from `config.yaml` via `params.py` (the script adds the project root to `sys.path`
automatically).

### Stage 3 — Feature extraction and export

```bash
python run_pipeline.py
```

Opens a GUI folder picker. For each selected experiment folder it:

1. Locates the `*.analysis.h5` file
2. Calls `make_expt_dataset()` → writes `*.features.h5`
3. Calls `export_perframe()` → writes `perframe/*.mat`
4. Calls `save_trx()` → writes `trx.mat`

Errors in one experiment folder are caught and logged; processing continues
with the next folder.
---

## Feature System Architecture

Features are defined in `features/` using a **declarative registry with automatic
dependency resolution**.

### The registry

`features/registry.py` defines:

- **`FeatureSpec`** — a dataclass describing one feature entry
- **`REGISTRY`** — an ordered dict mapping string keys to `FeatureSpec` instances
- **`compute_item()`** — a recursive resolver that computes a feature and all its
  dependencies on demand
- **`validate_registry()`** — called automatically at import time; raises on structural
  errors such as invalid dependency references or duplicate output keys

### `FeatureSpec` fields

```python
@dataclass
class FeatureSpec:
    func:          Callable         # the feature function to call
    requires:      list[str]        # registry keys that must be computed first
    outputs:       list[str]        # keys written to HDF5 (must match return dict keys)
    units:         dict             # units metadata per output key (for JAABA export)
    enabled:       bool = True      # False = skip this feature entirely
    save_mode:     str  = "scalar"  # "scalar" | "pose_per_fly" | "none"
    intermediates: list[str] = []   # intermediate keys produced (documented, not saved)
    params:        dict = {}        # feature-specific config kwargs passed to func

```
---
## Output Saving Behavior

### HDF5 file structure (`*.features.h5`)

```
/
├── meta/
│   ├── fps            scalar float
│   ├── pxpermm        scalar float
│   ├── ctr_ind        scalar int
│   ├── fwd_ind        scalar int
│   ├── node_names     array of bytes strings
│   └── track_names    array of bytes strings
│
├── pose/
│   ├── tracks/
│   │   ├── fly_000    (n_frames, n_nodes, 2)  ← raw world-frame pose
│   │   └── fly_001
│   ├── ego_tracks/
│   │   ├── fly_000    (n_frames, n_nodes, 2)  ← egocentric pose
│   │   └── fly_001
│   └── ego_rel_nearest/
│       ├── fly_000    (n_frames, n_nodes, 2)  ← pose relative to nearest fly
│       └── fly_001
│
└── features/
    ├── FV             (n_frames, n_flies)
    ├── wingL          (n_frames, n_flies)
    ├── minDist        (n_frames, n_flies)
    └── ...            one dataset per output key
```

### `save_mode` controls how an output is written

| `save_mode` | What it does |
|---|---|
| `"scalar"` | Output array of shape `(n_frames, n_flies)` written as a single dataset under `/features/<key>`. Default for all leaf features. |
| `"pose_per_fly"` | Output array of shape `(n_frames, n_nodes, 2, n_flies)` written per-fly under `/pose/<key>/fly_NNN`. Use for any full-skeleton output. |
| `"none"` | Nothing is written. Use for features that exist only to supply intermediates to downstream features. The values are still available in `computed` during the run. |

### Unit attributes

For every scalar output, the HDF5 dataset receives attributes from the `units` dict:
`quantity`, `unit_raw`, `unit_si`, `scale_expr`. These are read back by
`export_perframe()` to apply unit conversion when writing JAABA MAT files.
---

## Adding a New Feature
### Step 1 — Write the feature function

Place the function in the appropriate file under `features/`:

| Category | File |
|---|---|
| Individual kinematics (speed, acceleration, angles) | `kinematics.py` |
| Pairwise / nearest-neighbor geometry | `pairwise.py` |
| Wing angles | `wings.py` |
| Egocentric coordinates | `ego.py` |
| New category | Create a new file and import it at the top of `registry.py` |

#### Required function contract

```python
def feature_my_feature(tracks, features=None, ctr_ind=1, fwd_ind=0, **kwargs):
    """One-line description.

    Args:
        tracks:   np.ndarray of shape (n_frames, n_nodes, 2, n_flies).
                  Raw world-frame pose coordinates. NaNs where tracking was missing.
        features: dict of previously computed results. Keys are output key names
                  from upstream registry entries (e.g. "FV", "nearestFlyIdx").
                  Always populated when called via compute_item().
        ctr_ind:  int. Index of the centroid (thorax) node. Sourced from config.yaml.
        fwd_ind:  int. Index of the forward (head) node. Sourced from config.yaml.
        **kwargs: Required. Absorbs any additional kwargs passed through the pipeline.

    Returns:
        dict mapping output key names to np.ndarray.
        Arrays intended for HDF5 export and perframe MAT files must have
        shape (n_frames, n_flies).
    """
    n_frames, n_nodes, _, n_flies = tracks.shape
    ...
    return {"my_output_key": result_array}
```

**The function must return a dict.** Every key in the returned dict that appears in
the registry `outputs` list will be saved to HDF5. Extra keys in the dict are ignored.

**`**kwargs` is required.** The pipeline passes `ctr_ind` and `fwd_ind` as keyword
arguments to every feature function. Omitting `**kwargs` causes a `TypeError` at
runtime if those parameters are not explicitly declared in the signature.

**Output shape for exported features must be `(n_frames, n_flies)`.** Arrays with any
other shape are written to HDF5 but silently skipped by `export_perframe()`.

#### Accessing upstream results

Access results from `requires` entries using their **output key names** from the
`features` dict — not registry key names:

```python
# If requires=["nearest_geom"], available keys include:
thx_v         = features["thx_v"]           # (n_frames, 2, n_flies)
nearest_dist  = features["nearest_dist"]    # (n_frames, n_flies)
nearest_valid = features["nearest_valid"]   # (n_frames, n_flies) bool

# If requires=["nearest_neighbor"], available keys include:
nearest       = features["nearestFlyIdx"]   # (n_frames, n_flies) int
min_dist      = features["minDist"]         # (n_frames, n_flies)

# If requires=["kinematics"], available keys include:
FV = features["FV"]                         # (n_frames, n_flies)
```

See [Current Feature Registry](#current-feature-registry) for all available output
keys and the registry entry that produces each one.
---

### Step 2 — Register the feature in `registry.py`

Add an entry to `REGISTRY` in `features/registry.py`. If your function is in a new
file, import it explicitly at the top of `registry.py`.

```python
"my_feature": FeatureSpec(
    func=feature_my_feature,
    requires=["nearest_geom"],        # registry key names, not output key names
    outputs=["my_output_key"],        # must exactly match your function's return dict keys
    units={
        "my_output_key": {
            "quantity":    "velocity",      # controls JAABA unit mapping; see table below
            "unit_raw":    "px/frame",
            "unit_si":     "mm/sec",
            "scale_expr":  "fps/pxpermm",  # evaluated with fps and pxpermm as variables
        },
    },
    enabled=True,
    save_mode="scalar",               # "scalar" | "pose_per_fly" | "none"
),
```

`validate_registry()` runs automatically when `registry.py` is imported. If any
`requires` key does not exist, any `outputs` key is duplicated, or any other structural
rule is violated, you get a `ValueError` at startup before any computation runs.

#### Valid `quantity` values for JAABA unit conversion

| `quantity` | JAABA units |
|---|---|
| `"distance"` | mm |
| `"velocity"` | mm/s |
| `"acceleration"` | mm/s² |
| `"angle"` | deg |
| `"rot_speed"` | deg/s |
| `"time"` | s |
| `"other"` | unit (no conversion) |

The `scale_expr` may use `fps` and `pxpermm` as variables and supports standard Python
arithmetic. `^` is treated as `**`.

#### Documenting intermediate keys with `intermediates`

If your feature produces values consumed by downstream features but not saved to HDF5,
list them in `intermediates` and set `save_mode="none"`:

```python
"my_geom": FeatureSpec(
    func=feature_my_geom,
    requires=["nearest_neighbor"],
    outputs=[],
    intermediates=["my_vec", "my_dist"],   # contract for downstream consumers
    save_mode="none",
),
```

Listing keys in `intermediates` does not change runtime behavior — it is a documented
contract validated at import time for duplicates and key collisions.

#### Per-feature configuration with `params`

If your feature has tunable parameters, declare them in `params` rather than
hardcoding defaults:

```python
"my_feature": FeatureSpec(
    func=feature_my_feature,
    requires=[],
    outputs=["my_key"],
    units={...},
    params={"window": 5, "threshold": 0.1},
),
```

Values in `params` are merged into `**kwargs` when the function is called. Declare
them explicitly in the function signature (with matching defaults as fallback):

```python
def feature_my_feature(tracks, features=None, window=5, threshold=0.1, **kwargs):
    ...
```

---

### Step 3 — That's it

You do **not** need to modify:

| File | Why untouched |
|---|---|
| `dataset.py` | The scalar save loop and pose write loop are fully registry-driven |
| `run_pipeline.py` | Calls `make_expt_dataset()` with no feature-specific logic |
| `create_perframe.py` | Reads all `(n_frames, n_flies)` datasets from HDF5 automatically |
| `trx_mat.py` | Only reads raw pose joints, not computed features |

---

## Current Feature Registry

All 13 registered features:

| Registry Key | Function | `requires` | `outputs` | `save_mode` |
|---|---|---|---|---|
| `ego_tracks` | `compute_ego_tracks` | — | — | `pose_per_fly` |
| `kinematics` | `compute_individual_kinematics` | — | `FV FA LV LA LS RS` | `scalar` |
| `wing` | `feature_wingLR` | `ego_tracks` | `wingL wingR` | `scalar` |
| `wing_minmax` | `feature_minmax_wing_angle` | `wing` | `minWingAng maxWingAng wingAmp` | `scalar` |
| `nearest_neighbor` | `feature_nearest_neighbor` | — | `minDist nearestFlyIdx` | `scalar` |
| `nearest_geom` | `feature_nearest_geom` | `nearest_neighbor` | — | `none` |
| `ego_rel_nearest` | `feature_ego_rel_nearest` | `nearest_neighbor` | — | `pose_per_fly` |
| `fv_to_nearest` | `feature_FV_to_nearest` | `nearest_geom` | `FV_to_nearest` | `scalar` |
| `relfv_to_nearest` | `feature_relFV_to_nearest` | `nearest_geom` | `relFV_to_nearest` | `scalar` |
| `ls_to_nearest` | `feature_LS_to_nearest` | `nearest_geom` | `LS_to_nearest` | `scalar` |
| `relLS_to_nearest` | `feature_relLS_to_nearest` | `nearest_geom` | `relLS_to_nearest` | `scalar` |
| `ang_to_nearest` | `feature_ang_to_nearest` | `nearest_geom` | `ang_to_nearest` | `scalar` |
| `wing_arc_to_nearest` | `feature_wing_arc_to_nearest` | `nearest_neighbor` | `arcThetaL_to_nearest arcThetaR_to_nearest` | `scalar` |

### Intermediate keys from `nearest_geom`

These are available in `features` to any entry declaring `requires=["nearest_geom"]`.
They are not written to HDF5.

| Key | Shape | Description |
|---|---|---|
| `thx_xy` | `(n_frames, 2, n_flies)` | Thorax position in world coordinates |
| `hd_xy` | `(n_frames, 2, n_flies)` | Head position in world coordinates |
| `thx_v` | `(n_frames, 2, n_flies)` | Thorax velocity vector |
| `nearest_dir_unit` | `(n_frames, 2, n_flies)` | Unit vector from focal fly head to nearest fly thorax |
| `nearest_dir_perp` | `(n_frames, 2, n_flies)` | Perpendicular to `nearest_dir_unit` |
| `nearest_dist` | `(n_frames, n_flies)` | Euclidean distance to nearest fly |
| `nearest_valid` | `(n_frames, n_flies)` | Boolean mask: True where a valid nearest neighbor exists |

---

## Perframe Export and JAABA Compatibility

`export_perframe()` reads `*.features.h5` and writes one MAT file per scalar feature.

**Only datasets with shape `(n_frames, n_flies)` are exported.** Any other shape is
skipped.

**Unit conversion** is applied using the `scale_expr` stored as an HDF5 attribute.
The expression is evaluated with `fps` and `pxpermm` as the only available variables.

**`trx.mat`** is written by `trx_mat.py` and contains per-fly trajectory data
(position, heading, body axes) derived from the pose joints. Joint indices are derived
from `node_names_expected` in `config.yaml` via `params.py` and update automatically
if the skeleton changes.
---

## Output Files

### `*.features.h5`

The primary output of `dataset.py`. Stored in the experiment folder alongside the source `*.analysis.h5` file. Structure:

```
features.h5
├── meta/
│   ├── fps            scalar float
│   ├── pxpermm        scalar float
│   ├── ctr_ind        scalar int
│   ├── fwd_ind        scalar int
│   ├── node_names     array of bytes strings
│   └── track_names    array of bytes strings
│
├── pose/
│   ├── tracks/
│   │   ├── fly_000    (n_frames, n_nodes, 2)  ← raw world-frame pose
│   │   └── fly_001
│   ├── ego_tracks/
│   │   ├── fly_000    (n_frames, n_nodes, 2)  ← egocentric pose
│   │   └── fly_001
│   └── ego_rel_nearest/
│       ├── fly_000    (n_frames, n_nodes, 2)  ← pose relative to nearest fly
│       └── fly_001
│
└── features/
    ├── FV             (n_frames, n_flies)
    ├── wingL          (n_frames, n_flies)
    ├── minDist        (n_frames, n_flies)
    └── ...            one dataset per output key
```

### `perframe/*.mat`

One MATLAB `.mat` file per exported scalar feature (e.g. `perframe/FV.mat`, `perframe/wingL.mat`). Each file contains:

- `data` — a MATLAB cell array of shape `(1, n_flies)`, where each cell holds a row vector of shape `(1, n_frames)` with `float64` values, optionally unit-converted using `scale_expr`.
- `units` — a MATLAB struct with `num` and `den` fields (JAABA convention).

Only datasets with shape exactly `(n_frames, n_flies)` in the HDF5 file are exported. Features stored in the `pose/` subgroup (3D arrays) are excluded.

### `trx.mat`

A MATLAB struct array with one entry per fly. Each entry contains trajectory fields expected by JAABA:

| Field | Description |
|---|---|
| `x`, `y` | Thorax position in pixels |
| `theta` | Heading angle in radians `[0, 2π)` |
| `a`, `b` | Half-lengths derived from head–abdomen and wing–wing distances |
| `x_mm`, `y_mm`, `a_mm`, `b_mm`, `theta_mm` | Millimetre-scaled equivalents |
| `timestamps`, `dt` | Frame timestamps and inter-frame intervals |
| `fps`, `pxpermm` | Calibration constants |
| `nframes`, `firstframe`, `endframe` | Frame range |
| `moviename`, `moviefile` | Path to the arena movie |
| `id`, `label` | Fly index and track name |
