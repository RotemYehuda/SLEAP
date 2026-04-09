# Export scalar features → perframe/*.mat

import numpy as np
import h5py
from perframe.ds_utils import iter_datasets, safe_savemat

from params import FPS, PXPERMM

def _safe_eval_scale_expr(expr: str, fps: float, pxpermm: float) -> float:
    """Evaluate a scale expression using fps and pxpermm as variables.

    Supported variables: fps, pxpermm.
    Supports Python arithmetic operators; ^ is treated as ** (exponentiation).

    Raises:
        ValueError: if the expression cannot be evaluated, with the
                    offending expression included in the message.
    """
    cleaned = str(expr).strip().replace("^", "**")
    try:
        return float(eval(
            cleaned,
            {"__builtins__": {}},
            {"fps": float(fps), "pxpermm": float(pxpermm)},
        ))
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate scale_expr '{expr}': {e}. "
            f"Only 'fps' and 'pxpermm' are available as variables."
        ) from e


def _get_scale_from_ds(ds: h5py.Dataset, fps: float | None, pxpermm: float | None) -> float | None:
    if "scale_expr" not in ds.attrs:
        return None
    if fps is None or pxpermm is None:
        return None

    expr = ds.attrs["scale_expr"]
    if isinstance(expr, (bytes, np.bytes_)):
        expr = expr.decode()
    expr = str(expr).strip()
    if not expr:
        return None

    return _safe_eval_scale_expr(expr, fps=fps, pxpermm=pxpermm)


def jaaba_units_from_h5_dataset(ds, default_quantity="other"):
    """
    Build JAABA-style units struct:
      units.num = {'mm'} etc.
      units.den = {'s'} etc.

    We read from H5 attributes if present:
      quantity, unit_raw, unit_si, scale_expr
    We only export num/den into MAT (JAABA style).
    """
    # read attributes if exist
    quantity = ds.attrs.get("quantity", default_quantity)
    if isinstance(quantity, (bytes, np.bytes_)):
        quantity = quantity.decode()

    # map "quantity" to (num, den) for JAABA
    mapping = {
        # --- existing ---
        "distance":                     ("mm",   None),
        "velocity":                     ("mm",   "s"),
        "acceleration":                 ("mm",   "s^2"),
        "angle":                        ("rad",  None),
        "rot_speed":                    ("rad",  "s"),
        "time":                         ("s",    None),
        "other":                        ("unit", None),
        # --- position / distance ---
        "position":                     ("mm",   None),
        "distance_change_rate":         ("mm",   "s"),
        "area":                         ("mm^2", None),
        "area_change_rate":             ("mm^2", "s"),
        # --- orientation / angle ---
        "orientation":                  ("rad",  None),
        "velocity_direction":           ("rad",  None),
        "sideways_angle":               ("rad",  None),
        "yaw_angle":                    ("rad",  None),
        "absolute_yaw_angle":           ("rad",  None),
        # --- angular rates ---
        "angular_velocity":             ("rad",  "s"),
        "angular_speed":                ("rad",  "s"),
        "angle_change_rate":            ("rad",  "s"),
        "velocity_direction_change_rate": ("rad", "s"),
        # --- linear speeds / velocities ---
        "speed":                        ("mm",   "s"),
        "lateral_velocity_cor":         ("mm",   "s"),
        "lateral_speed_cor":            ("mm",   "s"),
        "forward_velocity_cor":         ("mm",   "s"),
        "forward_velocity_ctr":         ("mm",   "s"),
        "forward_velocity_tail":        ("mm",   "s"),
        "sideways_velocity_ctr":        ("mm",   "s"),
        "sideways_velocity_tail":       ("mm",   "s"),
        "signed_lateral_velocity_cor":  ("mm",   "s"),
        "a_change_rate":                ("mm",   "s"),
        "b_change_rate":                ("mm",   "s"),
        # --- dimensionless ---
        "eccentricity":                 ("unit", None),
        "eccentricity_change_rate":     ("unit", "s"),
        "fractional_offset":            ("unit", None),
        "turn_sign":                    ("unit", None),
        "count":                        ("unit", None),
        "index":                        ("unit", None),
    }

    num_str, den_str = mapping.get(str(quantity), ("unit", None))

    # JAABA wants MATLAB cell arrays (object arrays) of shape (1,1) or (1,0)
    num = np.empty((1, 1), dtype=object)
    num[0, 0] = num_str

    if den_str is None:
        den = np.empty((1, 0), dtype=object)
    else:
        den = np.empty((1, 1), dtype=object)
        den[0, 0] = den_str

    return {"num": num, "den": den}


def jaaba_data_cell(values_2d):
    """
    values_2d: np.ndarray shape (n_frames, n_flies)

    returns: MATLAB cell array shape (1, n_flies)
             each cell is row vector shape (1, n_frames)
    """
    n_frames, n_flies = values_2d.shape
    data_cell = np.empty((1, n_flies), dtype=object)

    for fly in range(n_flies):
        v = values_2d[:, fly]

        # JAABA usually expects doubles
        v_out = v.astype(np.float64, copy=False)
        # if np.issubdtype(v.dtype, np.floating):
        #     v_out = v.astype(np.float64, copy=False)
        # else:
        #     v_out = v.astype(np.float64, copy=False)

        data_cell[0, fly] = v_out.reshape(1, -1)

    return data_cell


def export_perframe(features_h5, perframe_dir, overwrite, convert_units=False) -> None:
    if not features_h5.is_file():
        raise FileNotFoundError(features_h5)

    # overwrite
    if perframe_dir.exists():
        if overwrite:
            for p in perframe_dir.rglob("*.mat"):
                p.unlink(missing_ok=True)
        else:
            raise FileExistsError(f"{perframe_dir} exists")
    perframe_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(features_h5, "r") as f:
        # collect candidate datasets
        all_ds = list(iter_datasets(f))

        fps = FPS
        pxpermm = PXPERMM

        if "meta" in f:
            meta = f["meta"]
            if "fps" in meta:
                fps = float(meta["fps"][()])
            if "pxpermm" in meta:
                pxpermm = float(meta["pxpermm"][()])

        # infer (frames, flies) from first 2D dataset anywhere
        n_frames = None
        n_flies = None
        for path, ds in all_ds:
            if ds.ndim == 2:
                n_frames, n_flies = ds.shape
                break

        if n_frames is None:
            raise RuntimeError("No 2D datasets found in the H5 file")

        # export only datasets exactly (frames, flies)
        for path, ds in all_ds:
            if ds.ndim != 2 or ds.shape != (n_frames, n_flies):
                continue

            values = ds[:]  # (frames, flies)

            if convert_units:
                scale = _get_scale_from_ds(ds, fps=fps, pxpermm=pxpermm)
                if scale is not None:
                    values = values * scale

            data_cell = jaaba_data_cell(values)
            units_struct = jaaba_units_from_h5_dataset(ds)

            ds_name = path.split("/")[-1]
            out_path = perframe_dir / f"{ds_name}.mat"
            safe_savemat(out_path, {"data": data_cell, "units": units_struct})