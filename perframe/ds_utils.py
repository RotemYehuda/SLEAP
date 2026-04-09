# HDF5 iteration + atomic MAT file writing

from pathlib import Path
import h5py
import scipy.io

def iter_datasets(h5obj, prefix=""):
    """
    Yield (full_path, dataset) for all datasets under h5 obj (recursive).
    full_path uses POSIX separators: group/subgroup/name
    """
    for key in h5obj.keys():
        obj = h5obj[key]
        full = f"{prefix}{key}" if prefix == "" else f"{prefix}/{key}"
        if isinstance(obj, h5py.Dataset):
            yield full, obj
        elif isinstance(obj, h5py.Group):
            yield from iter_datasets(obj, prefix=full)


def safe_savemat(dest: Path, data_dict: dict) -> None:
    """Save a dict to a MATLAB .mat file using an atomic write pattern.

    Writes to a temporary file first, then renames it to the destination.
    This ensures that `dest` is never left in a partially-written state:
    - If scipy.io.savemat raises, the .tmp file is deleted and dest is untouched.
    - If the process is killed after the write but before the rename, a stale
      .tmp file may remain, but it will be cleaned up on the next run.
    - The rename is atomic on most filesystems (same-device move).

    Args:
        dest:      Path to the output .mat file.
        data_dict: Dictionary to save. Must be compatible with scipy.io.savemat.

    Raises:
        Any exception raised by scipy.io.savemat, after cleaning up the .tmp file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        scipy.io.savemat(tmp, data_dict)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    else:
        dest.unlink(missing_ok=True)
        tmp.rename(dest)