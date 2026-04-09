# Signal processing (fill_missing, egocentric normalization, signed_angle)

import numpy as np
import pandas as pd


def fill_missing(x, kind="nearest", warn_all_nan=False, **kwargs):
    """Fill missing values in a timeseries.

    Args:
        x: Timeseries of shape (time, _) or (_, time, _).
        kind: Type of interpolation to use. Defaults to "nearest".
        warn_all_nan: If True, emit a warning when a column remains entirely
            NaN after interpolation (i.e. it had no valid values to begin with).
            Defaults to False to preserve existing behaviour in batch pipelines.

    Returns:
        Timeseries of the same shape as the input with NaNs filled in.
    """
    if x.ndim == 3:
        return np.stack(
            [fill_missing(xi, kind=kind, warn_all_nan=warn_all_nan, **kwargs) for xi in x],
            axis=0,
        )

    result = (
        pd.DataFrame(x)
        .interpolate(kind=kind, axis=0, limit_direction="both", **kwargs)
        .to_numpy()
    )

    if warn_all_nan and np.isnan(result).all(axis=0).any():
        import warnings
        all_nan_cols = np.where(np.isnan(result).all(axis=0))[0]
        warnings.warn(
            f"fill_missing: {len(all_nan_cols)} column(s) are entirely NaN "
            f"after interpolation (column indices: {all_nan_cols.tolist()}). "
            f"These columns had no valid values to interpolate from.",
            stacklevel=2,
        )

    return result


def normalize_to_egocentric(x, rel_to=None, scale_factor=1, ctr_ind=1, fwd_ind=0, fill=True, return_angles=False):
    """Normalize pose estimates to egocentric coordinates.

    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2)
        rel_to: Pose to align x with of shape (joints, 2) or (time, joints, 2). Defaults
            to x if not specified.
        scale_factor: Spatial scaling to apply to coordinates after centering.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
        fill: If True, interpolate missing ctr and fwd coordinates. If False, timesteps
            with missing coordinates will be all NaN. Defaults to True.
        return_angles: If True, return angles with the aligned coordinates.

    Returns:
        Egocentrically aligned poses of the same shape as the input.

        If return_angles is True, also returns a vector of angles.
    """

    if rel_to is None:
        rel_to = x

    is_singleton = (x.ndim == 2) and (rel_to.ndim == 2)

    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if rel_to.ndim == 2:
        rel_to = np.expand_dims(rel_to, axis=0)

    # Find egocentric forward coordinates.
    ctr = rel_to[..., ctr_ind, :]  # (t, 2)
    fwd = rel_to[..., fwd_ind, :]  # (t, 2)
    if fill:
        ctr = fill_missing(ctr, kind="nearest")
        fwd = fill_missing(fwd, kind="nearest")
    ego_fwd = fwd - ctr

    # Compute angle.
    ang = np.arctan2(ego_fwd[..., 1], ego_fwd[..., 0])  # arctan2(y, x) -> radians in [-pi, pi]
    ca = np.cos(ang)  # (t,)
    sa = np.sin(ang)  # (t,)

    # Build rotation matrix.
    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1

    # Center and scale.
    x = x - np.expand_dims(ctr, axis=1)
    x /= scale_factor

    # Pad, rotate and crop.
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]

    if is_singleton:
        x = x[0]

    if return_angles:
        return x, ang
    else:
        return x

def signed_angle(a, b):
    """Finds the signed angle between two 2D vectors a and b.

    Args:
        a: Array of shape (n, 2).
        b: Array of shape (n, 2).

    Returns:
        The signed angles in degrees in vector of shape (n, ).

        This angle is positive if a is rotated clockwise to align to b and negative if
        this rotation is counter-clockwise.
    """
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    a = np.where(norm_a > 0, a / norm_a, 0.0)
    b = np.where(norm_b > 0, b / norm_b, 0.0)

    theta = np.arccos(np.around(np.sum(a * b, axis=1), decimals=4))
    # cross = np.cross(a, b, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    sign = np.zeros(cross.shape)
    sign[cross >= 0] = -1
    sign[cross < 0] = 1
    return np.rad2deg(theta) * sign