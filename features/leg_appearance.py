# Front-leg kinematics in the fly's egocentric frame

import numpy as np


def compute_front_leg_angles(tracks, features=None, ctr_ind=1, left_front_ind=5, right_front_ind=6, **kwargs,):
    """
    Egocentric angle of each front leg relative to the body axis (rad).

    Returns
    -------
    front_leg_L : (T, n_flies)  left front leg egocentric angle (rad)
    front_leg_R : (T, n_flies)  right front leg egocentric angle (rad)
    """
    x_ctr = tracks[:, ctr_ind, 0, :]  # (T, n_flies)
    y_ctr = tracks[:, ctr_ind, 1, :]
    theta = features["theta"]          # (T, n_flies)

    dx_L = tracks[:, left_front_ind,  0, :] - x_ctr
    dy_L = tracks[:, left_front_ind,  1, :] - y_ctr
    dx_R = tracks[:, right_front_ind, 0, :] - x_ctr
    dy_R = tracks[:, right_front_ind, 1, :] - y_ctr

    front_leg_L = ((np.arctan2(dy_L, dx_L) - theta) + np.pi) % (2 * np.pi) - np.pi
    front_leg_R = ((np.arctan2(dy_R, dx_R) - theta) + np.pi) % (2 * np.pi) - np.pi

    return {
        "front_leg_L_ang": front_leg_L.astype(np.float64),
        "front_leg_R_ang": front_leg_R.astype(np.float64),
    }


def compute_front_leg_angle_diff(tracks, features=None, **kwargs):
    """
    Left–right asymmetry of the front leg angles (rad).
        front_leg_angle_diff = front_leg_R - front_leg_L

    Interpretation:
         0  = symmetric posture (both legs at equal distances from body axis)
        >0  = right leg reaches further forward / outward than left
        <0  = left leg reaches further forward / outward than right
    """
    return {
        "front_leg_angle_diff": (
            features["front_leg_R_ang"] - features["front_leg_L_ang"]
        ).astype(np.float64),
    }


# Extension feature
def compute_front_leg_extension(
    tracks,
    features=None,
    ctr_ind=1,
    left_front_ind=5,
    right_front_ind=6,
    pxpermm=10.5,
    **kwargs,
):
    """
    Euclidean distance from thorax to each front leg tip (mm).

    Captures how far the leg is physically extended from the body centre,
    independent of direction.  Large values correspond to a reaching or
    outstretched leg; small values indicate a retracted leg.

    Returns
    -------
    front_leg_ext_L : (T, n_flies)  left front leg extension (mm)
    front_leg_ext_R : (T, n_flies)  right front leg extension (mm)
    """
    x_ctr = tracks[:, ctr_ind, 0, :]
    y_ctr = tracks[:, ctr_ind, 1, :]

    dx_L = tracks[:, left_front_ind,  0, :] - x_ctr
    dy_L = tracks[:, left_front_ind,  1, :] - y_ctr
    dx_R = tracks[:, right_front_ind, 0, :] - x_ctr
    dy_R = tracks[:, right_front_ind, 1, :] - y_ctr

    ext_L = np.sqrt(dx_L ** 2 + dy_L ** 2) / float(pxpermm)
    ext_R = np.sqrt(dx_R ** 2 + dy_R ** 2) / float(pxpermm)

    return {
        "front_leg_ext_L": ext_L.astype(np.float64),
        "front_leg_ext_R": ext_R.astype(np.float64),
    }