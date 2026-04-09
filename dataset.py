# Core feature orchestration + HDF5 writing

import h5py
import numpy as np
from pathlib import Path
from io_utils import load_tracks, encode_hdf5_strings
from features.registry import compute_item, get_units_for_key, REGISTRY
from params import FPS, PXPERMM, NODE_NAMES_EXPECTED
import warnings as _warnings

_EXPECTED_NODE_ROLES = {
    "fwd_ind":        ("fwd_ind",  "head"),
    "ctr_ind":        ("ctr_ind",  "thorax"),
    "abdomen_index":  (2,          "abdomen"),
    "left_wing_index":(3,          "left_wing"),
    "right_wing_index":(4,         "right_wing"),
}


def _validate_node_indices(node_names: list, ctr_ind: int, fwd_ind: int) -> None:
    """Validate that node indices are consistent with the loaded node_names.

    Two checks are performed:

    1. If NODE_NAMES_EXPECTED is configured in config.yaml, the loaded
       node_names must match it exactly (same names, same order).

    2. The specific indices used throughout the pipeline (ctr_ind, fwd_ind,
       and the fixed abdomen/wing indices) are checked against the names
       at those positions.

    Args:
        node_names: List of node name strings loaded from the SLEAP file.
        ctr_ind:    Index of the centroid (thorax) node.
        fwd_ind:    Index of the forward (head) node.

    Raises:
        ValueError: with a descriptive message if any check fails.
    """
    errors = []

    # Check 1: full node list matches expected if configured
    if NODE_NAMES_EXPECTED is not None:
        if list(node_names) != list(NODE_NAMES_EXPECTED):
            errors.append(
                f"Node names do not match expected skeleton:\n"
                f"  Loaded:   {node_names}\n"
                f"  Expected: {list(NODE_NAMES_EXPECTED)}\n"
                f"  If the skeleton has changed, update 'node_names_expected' "
                f"in config.yaml."
            )
            # If the full list is wrong, skip per-index checks — they would
            # produce redundant errors.
            raise ValueError("Node name validation failed:\n" + "\n".join(errors))

    n_nodes = len(node_names)

    # Check 2: specific indices are in range
    named_indices = {
        "fwd_ind": fwd_ind,
        "ctr_ind": ctr_ind,
        "abdomen_index": 2,
        "left_wing_index": 3,
        "right_wing_index": 4,
    }
    for label, idx in named_indices.items():
        if idx >= n_nodes:
            errors.append(
                f"  {label}={idx} is out of range for skeleton with "
                f"{n_nodes} nodes: {node_names}"
            )

    if errors:
        raise ValueError("Node index validation failed:\n" + "\n".join(errors))

    # Check 3: named indices point to expected node names (soft check)
    expected_at_index = {
        fwd_ind:  "head",
        ctr_ind:  "thorax",
        2:        "abdomen",
        3:        "L_wing",
        4:        "R_wing",
    }
    warnings = []
    for idx, expected_name in expected_at_index.items():
        if idx < n_nodes:
            actual_name = node_names[idx]
            if actual_name != expected_name:
                warnings.append(
                    f"  index {idx}: expected '{expected_name}', "
                    f"got '{actual_name}'"
                )
    if warnings:
        _warnings.warn(
            "Node names at expected indices do not match defaults. "
            "Verify that ctr_ind/fwd_ind are set correctly for this skeleton.\n"
            + "\n".join(warnings),
            stacklevel=3,
        )


def _write_pose_per_fly(pose_group, group_name, array, n_flies):
    """Write a (n_frames, nodes, 2, n_flies) array as per-fly sub-datasets.

    Creates pose_group/<group_name>/fly_000, fly_001, ... each with shape
    (n_frames, nodes, 2) and axes/fly_index attributes.

    Args:
        pose_group: An open h5py Group (the 'pose' group).
        group_name: Name of the sub-group to create (e.g. 'ego_tracks').
        array:      np.ndarray of shape (n_frames, nodes, 2, n_flies).
        n_flies:    Number of individuals.
    """
    grp = pose_group.create_group(group_name)
    for fly in range(n_flies):
        ds = grp.create_dataset(
            f"fly_{fly:03d}",
            data=array[..., fly],
            compression=1,
        )
        ds.attrs["axes"] = ["time", "node", "xy"]
        ds.attrs["fly_index"] = fly


def make_expt_dataset(
    expt_folder: str,
    h5_file: str,
    output_path: str | None = None,
    overwrite: bool = False,
    ctr_ind: int = 1,
    fwd_ind: int = 0,
    fps: float | None = None,
    pxpermm: float | None = None
) -> str:
    """Gather experiment data into a single file.

    Args:
        pxpermm:
        fps:
        expt_folder: Full absolute path to the experiment folder.
        h5_file: Path to a SLEAP HDF5 analysis file ('*.analysis.h5').
        output_path: Path to save the resulting dataset to. Can be specified as a folder
            or full path ending with ".h5". Defaults to saving to current folder. If a
            folder is specified, the dataset filename will be the experiment folder
            name with ".h5".
        overwrite: If True, overwrite even if the output path already exists. Defaults
            to False.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.

    Returns:
        Path to output dataset.
    """

    # Resolve calibration — argument takes priority, then module default.
    fps = float(fps) if fps is not None else FPS
    pxpermm = float(pxpermm) if pxpermm is not None else PXPERMM

    # Resolve output path
    h5_path = Path(h5_file)
    expt_path = Path(expt_folder)

    if output_path is None:
        stem = h5_path.name
        if stem.endswith(".analysis.h5"):
            output_path = h5_path.with_name(stem[: -len(".analysis.h5")] + ".features.h5")
        else:
            output_path = h5_path.with_suffix(".features.h5")
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"Output path already exists and overwrite is set to False")
        return str(output_path)

    # Load tracking
    tracks, node_names, track_names = load_tracks(h5_file)
    if tracks.ndim != 4:
        raise ValueError(
            "ValueError: not enough values to unpack"
        )

    n_frames, n_nodes, _, n_flies = tracks.shape
    if tracks.shape[2] != 2:
        raise ValueError(
            f"Expected tracks with shape (time, joints, 2, fly). Got: {tracks.shape}"
        )

    if n_flies < 1:
        raise ValueError("No tracked individuals found (n_flies < 1).")

    # Ensure node_names length matches nodes axis
    if len(node_names) != n_nodes:
        raise ValueError(
            f"node_names length ({len(node_names)}) does not match nodes axis ({n_nodes})."
        )
    _validate_node_indices(node_names, ctr_ind=ctr_ind, fwd_ind=fwd_ind)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expt_name = expt_path.name
    print(f"\tCreating dataset for: {expt_name}")
    print(f"\tTracks: frames={n_frames}, nodes={n_nodes}, flies={n_flies}")

    computed = {}
    targets = [name for name, spec in REGISTRY.items() if spec.enabled]

    for name in targets:
        compute_item(
            name,
            tracks,
            REGISTRY,
            computed,
            ctr_ind=ctr_ind,
            fwd_ind=fwd_ind,
        )

    # Write output HDF5
    print(f"\tSaving to: {output_path}")

    with h5py.File(output_path, "w") as f:

        # Metadata
        meta = f.create_group("meta")

        meta.create_dataset("expt_name", data=np.bytes_(expt_name))
        meta.create_dataset("expt_folder", data=np.bytes_(expt_folder))
        meta.create_dataset("source_analysis_file", data=np.bytes_(h5_file))
        meta.create_dataset("node_names", data=encode_hdf5_strings(node_names))
        meta.create_dataset("track_names", data=track_names)
        meta.create_dataset("fps", data=fps)
        meta.create_dataset("pxpermm", data=pxpermm)

        meta.create_dataset("ctr_ind", data=ctr_ind)
        meta.create_dataset("fwd_ind", data=fwd_ind)

        # Pose
        pose = f.create_group("pose")

        _write_pose_per_fly(pose, "tracks", tracks, n_flies)
        # Registry-driven pose outputs.
        for name in targets:
            spec = REGISTRY[name]
            if spec.save_mode != "pose_per_fly":
                continue
            for key in spec.outputs:
                if key not in computed:
                    raise KeyError(
                        f"Registry entry '{name}' declared output '{key}' with "
                        f"save_mode='pose_per_fly' but '{key}' was not found in computed."
                    )
                _write_pose_per_fly(pose, key, computed[key], n_flies)

        save_keys = []
        for t in targets:
            spec = REGISTRY[t]
            if spec.save_mode == "scalar":  # only scalar outputs go here
                save_keys.extend(spec.outputs)

        save_keys = list(dict.fromkeys(save_keys))  # deduplicate, preserve order

        for k in save_keys:
            ds = f.create_dataset(k, data=computed[k], compression=1)
            u = get_units_for_key(k, REGISTRY)
            if u is not None:
                ds.attrs["quantity"] = u.get("quantity", "")
                ds.attrs["unit_raw"] = u.get("unit_raw", "")
                ds.attrs["unit_si"] = u.get("unit_si", "")
                ds.attrs["scale_expr"] = u.get("scale_expr", "")

    return str(output_path)