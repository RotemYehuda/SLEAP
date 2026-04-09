# SLEAP HDF5 loading

import numpy as np
import h5py


def load_tracks(track_file):
    """Load proofread and exported pose tracks.
    Args:
        track_file: Path to a SLEAP '*.analysis.h5' file containing tracked poses.
    Returns:
        tracks: NumPy array of shape (time, joints, 2, n_flies) with pose
            coordinates for all tracked individuals.
        node_names contains a list of string names for the joints.
        track_names: Array of track identifiers defining the order of
            individuals along the fly axis.
    """
    with h5py.File(track_file, "r") as f:
        tracks = np.transpose(f["tracks"][:])  # (frame, joint, xy, fly)
        node_names = f["node_names"][:]
        node_names = [x.decode() for x in node_names]
        track_names = f["track_names"][:]

    valid_frames = np.argwhere(
        np.isfinite(tracks.reshape(len(tracks), -1)).any(axis=-1)
    ).ravel()  # .ravel() is safe for empty, 1-element, and multi-element arrays

    if len(valid_frames) == 0:
        raise ValueError(
            f"No valid (finite) frames found in '{track_file}'. "
            f"The file may be empty or entirely NaN."
        )

    last_fidx = int(valid_frames[-1])
    tracks = tracks[:last_fidx + 1]

    return tracks, node_names, track_names


def encode_hdf5_strings(s: list[str]) -> list[np.bytes_]:
    """Encode a list of strings as numpy bytes for h5py compatibility.

    h5py requires strings to be encoded as numpy bytes_ objects (not Python
    str or bytes) when writing variable-length string datasets. Passing plain
    Python strings causes a TypeError in some h5py versions.

    Args:
        s: List of strings to encode.

    Returns:
        List of np.bytes_ objects suitable for h5py.create_dataset().
    """
    return [np.bytes_(x) for x in s]