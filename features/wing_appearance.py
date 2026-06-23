# Wing angles in egocentric frame

import numpy as np

def compute_wing_angles(tracks, features=None, ctr_ind=1, left_ind=10, right_ind=9, **kwargs):
    """
    Wing angles in the fly's egocentric frame (radians).
    """
    # Thorax position and heading
    x_ctr = tracks[:, ctr_ind, 0, :]   # (T, n_flies)
    y_ctr = tracks[:, ctr_ind, 1, :]
    theta = features["theta"]

    # Left wing tip
    x_L = tracks[:, left_ind,  0, :]
    y_L = tracks[:, left_ind,  1, :]

    # Right wing tip
    x_R = tracks[:, right_ind, 0, :]
    y_R = tracks[:, right_ind, 1, :]

    # Vector from thorax to each wing tip
    dx_L = x_L - x_ctr
    dy_L = y_L - y_ctr
    dx_R = x_R - x_ctr
    dy_R = y_R - y_ctr

    # World-space angle of each wing tip, then subtract heading
    wingL = ((np.arctan2(dy_L, dx_L) - theta) % (2 * np.pi)) - np.pi
    wingR = ((np.arctan2(dy_R, dx_R) - theta) % (2 * np.pi)) - np.pi

    return {
        "wingL": wingL.astype(np.float64),
        "wingR": wingR.astype(np.float64),
    }

def compute_mean_wing_angle(tracks, features=None, **kwargs):
    """
    Signed mean wing angle (rad).
    (-wingL + wingR) / 2

    Both wings extended symmetrically → 0
    Right wing extended more          → positive
    Left wing extended more           → negative
    """
    return {
        "mean_wing_angle": ((-features["wingL"] + features["wingR"]) / 2).astype(np.float64)
    }

def compute_wing_angle_diff(tracks, features=None, **kwargs):
    """
    Difference between right and left wing angles (rad).
    wingR - wingL

    Both wings extended symmetrically → 0
    Right wing extended more          → positive
    Left wing extended more           → negative
    """
    return {
        "wing_angle_diff": (features["wingR"] - features["wingL"]).astype(np.float64)
    }

def compute_wing_angle_imbalance(tracks, features=None, **kwargs):
    """
    Absolute imbalance between left and right wing angles (rad).
    abs(wingR + wingL)

    Both wings extended symmetrically → 0  (they cancel)
    One wing extended, other folded   → large value
    """
    return {
        "wing_angle_imbalance": np.abs(features["wingR"] + features["wingL"]).astype(np.float64)
    }

def compute_minmax_wing_angle(tracks, features=None, **kwargs):
    """
    Minimum of (-wingL, wingR) per frame (rad).
    Low value = neither wing extended, or the less-extended wing.

    Maximum of (-wingL, wingR) per frame (rad).
    High value = the more-extended wing, regardless of side.
    """
    return {
        "min_wing_angle": np.minimum(-features["wingL"], features["wingR"]).astype(np.float64),
        "max_wing_angle": np.maximum(-features["wingL"], features["wingR"]).astype(np.float64)
    }