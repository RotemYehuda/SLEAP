# Stage 3: Fill missing values in .analysis.h5 files
#
# Requires in params.py:
#   MAX_GAP_BY_NODE = {
#       "default": 15,     # frames left un-filled if the interior gap is longer than this
#       "thorax":  5,      # example: stricter threshold for a centroid-like node
#       "head":    20,     # example: more tolerant threshold
#   }
#   (keys must match your skeleton's node names exactly; "default" is the fallback
#    for any node not listed. Adjust frame counts to your frame rate / behaviors of interest.)

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from params import ARENA_PARENT_DIR as _ARENA_PARENT_DIR
from params import SLEAP_OUTPUT_NAME
from params import MAX_GAP_BY_NODE

ARENA_PARENT_DIR = Path(_ARENA_PARENT_DIR) if _ARENA_PARENT_DIR else Path(".")
OUTPUT_NAME = SLEAP_OUTPUT_NAME

FILLED_TRACKS_KEY = "tracks_filled"
FILLED_MASK_KEY = "tracks_filled_mask"


def get_max_gap(node_name, max_gap_by_node):
    return max_gap_by_node.get(node_name, max_gap_by_node.get("default", None))


def fill_missing_1d(y, kind="linear", max_gap=None):
    """
    Fills missing values in a 1D array.

    Interior gaps are linearly interpolated; gaps longer than max_gap are left
    as NaN instead of being interpolated. Leading/trailing NaNs are filled with
    the nearest valid value (constant extrapolation), same as SLEAP's original.

    Returns (filled_array, was_filled_mask).
    """
    y = y.copy()
    n = len(y)
    valid = ~np.isnan(y)
    x = np.flatnonzero(valid)

    if len(x) < 2:
        # Not enough real data in this slice to interpolate anything
        return y, np.zeros(n, dtype=bool)

    f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
    missing = np.flatnonzero(np.isnan(y))
    if len(missing):
        y[missing] = f(missing)

    if max_gap is not None:
        idx = 0
        while idx < n:
            if not valid[idx]:
                start = idx
                while idx < n and not valid[idx]:
                    idx += 1
                end = idx  # exclusive
                is_interior = start > 0 and end < n
                if is_interior and (end - start) > max_gap:
                    y[start:end] = np.nan
            else:
                idx += 1

    still_missing = np.isnan(y)
    if still_missing.any() and (~still_missing).any():
        valid_idx = np.flatnonzero(~still_missing)
        y[still_missing] = np.interp(np.flatnonzero(still_missing), valid_idx, y[valid_idx])

    was_filled = (~valid) & (~np.isnan(y))
    return y, was_filled


def fill_missing(Y, node_names, max_gap_by_node, kind="linear"):
    """
    Y shape: (frames, nodes, 2, tracks)
    Returns (Y_filled, mask) with the same shape; mask[i] is True where the
    value was originally missing and got filled (interpolated or extrapolated).
    """
    Y = Y.copy()
    n_frames, n_nodes, n_coords, n_tracks = Y.shape
    mask = np.zeros(Y.shape, dtype=bool)

    for node_i in range(n_nodes):
        max_gap = get_max_gap(node_names[node_i], max_gap_by_node)
        for coord_i in range(n_coords):
            for track_i in range(n_tracks):
                y = Y[:, node_i, coord_i, track_i]
                y_filled, was_filled = fill_missing_1d(y, kind=kind, max_gap=max_gap)
                Y[:, node_i, coord_i, track_i] = y_filled
                mask[:, node_i, coord_i, track_i] = was_filled

    return Y, mask


def print_node_summary(Y, Y_filled, mask, node_names):
    print("\n  Node summary (missing / filled / left-NaN, % of frames):")
    for node_i, node_name in enumerate(node_names):
        orig_missing = np.isnan(Y[:, node_i, :, :]).any(axis=1)
        filled = mask[:, node_i, :, :].any(axis=1)
        still_nan = np.isnan(Y_filled[:, node_i, :, :]).any(axis=1)
        print(
            f"    {node_name:15s}  "
            f"missing: {100 * orig_missing.mean():5.1f}%   "
            f"filled: {100 * filled.mean():5.1f}%   "
            f"left NaN (gap>max): {100 * still_nan.mean():5.1f}%"
        )


def process_file(h5_path, max_gap_by_node, kind="nearest"):
    with h5py.File(h5_path, "r") as f:
        raw_tracks = f["tracks"][:]  # (instances, 2, nodes, frames) as stored by SLEAP
        node_names = [n.decode() if isinstance(n, bytes) else n for n in f["node_names"][:]]

    Y = raw_tracks.T  # (frames, nodes, 2, instances)

    Y_filled, mask = fill_missing(Y, node_names, max_gap_by_node, kind=kind)
    print_node_summary(Y, Y_filled, mask, node_names)

    tracks_filled = Y_filled.T  # back to (instances, 2, nodes, frames)
    mask_full = mask.T

    with h5py.File(h5_path, "r+") as f:
        for key in (FILLED_TRACKS_KEY, FILLED_MASK_KEY):
            if key in f:
                del f[key]
        f.create_dataset(FILLED_TRACKS_KEY, data=tracks_filled, compression="gzip")
        f.create_dataset(FILLED_MASK_KEY, data=mask_full, compression="gzip")


def parse_selection(selection, max_index):
    selected = set()
    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return [i for i in selected if 1 <= i <= max_index]