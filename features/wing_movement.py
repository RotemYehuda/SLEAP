# Wing angle dynamics (rates of change)

import numpy as np

def compute_dminmax_wing_angle(tracks, features=None, fps=30, **kwargs):
    """
    Rate of change of the more-extended (dmax) and less-extended (dmin)
    wing angles (rad/s).

    Tracks whichever wing is dominant/subordinate at each frame to avoid
    discontinuities when dominance switches.
    """
    wingL = features["wingL"]   # (T, n_flies)
    wingR = features["wingR"]

    danglel = np.diff(-wingL, axis=0)   # (T-1, n_flies)
    dangler = np.diff( wingR, axis=0)

    right_dominant = wingR[:-1] > -wingL[:-1]   # (T-1, n_flies)

    dmax = danglel.copy()
    dmax[right_dominant]  = dangler[right_dominant]

    dmin = danglel.copy()
    dmin[~right_dominant] = dangler[~right_dominant]

    dmax = dmax * fps
    dmin = dmin * fps

    nan_row = np.full((1, wingL.shape[1]), np.nan)
    return {
        "dmax_wing_angle": np.concatenate([nan_row, dmax], axis=0).astype(np.float64),
        "dmin_wing_angle": np.concatenate([nan_row, dmin], axis=0).astype(np.float64),
    }

def compute_dwing_angle_diff(tracks, features=None, fps=30, **kwargs):
    """Rate of change of wing angle difference (rad/s)."""
    return {
        "dwing_angle_diff": (np.diff(features["wing_angle_diff"], axis=0, prepend=np.nan) * fps).astype(np.float64)
    }

def compute_dwing_angle_imbalance(tracks, features=None, fps=30, **kwargs):
    """
    Rate of change of wing angle imbalance (rad/s).
    Tracks the dominant imbalance direction per frame to avoid
    discontinuities at sign switches.
    """
    wingL = features["wingL"]   # (T, n_flies)
    wingR = features["wingR"]

    imbalancer =  wingR + wingL    # (T, n_flies)
    imbalancel = -imbalancer

    dimbalancer = np.diff(imbalancer, axis=0)   # (T-1, n_flies)
    dimbalancel = np.diff(imbalancel, axis=0)

    # Default: use left imbalance derivative
    result = dimbalancel.copy()

    # Where right imbalance is dominant (wingR + wingL > 0)
    # right_dominant = imbalancer[:-1] > imbalancel[:-1]   # i.e. wingR+wingL > 0
    right_dominant = imbalancer[:-1] > 0
    result[right_dominant] = dimbalancer[right_dominant]

    result = result * fps

    nan_row = np.full((1, wingL.shape[1]), np.nan)
    return {
        "dwing_angle_imbalance": np.concatenate([nan_row, result], axis=0).astype(np.float64)
    }

def compute_minmax_absdwing_angle(tracks, features=None, fps=30, **kwargs):
    """
    Min and max of absolute wing angle rates across left and right wings (rad/s).

    min_absdwing_angle: whichever wing is changing slower
    max_absdwing_angle: whichever wing is changing faster
    """
    wingL = features["wingL"]   # (T, n_flies)
    wingR = features["wingR"]

    absdL = np.abs(np.diff(wingL, axis=0)) * fps   # (T-1, n_flies)
    absdR = np.abs(np.diff(wingR, axis=0)) * fps

    nan_row = np.full((1, wingL.shape[1]), np.nan)
    return {
        "min_absdwing_angle": np.concatenate([nan_row, np.minimum(absdL, absdR)], axis=0).astype(np.float64),
        "max_absdwing_angle": np.concatenate([nan_row, np.maximum(absdL, absdR)], axis=0).astype(np.float64),
    }

def compute_minmax_dwing_angle_in(tracks, features=None, fps=30, **kwargs):
    """
    Min and max of inward wing angular velocity across left and right wings (rad/s).

    Positive = wing moving toward body (inward/folding)
    Negative = wing moving away from body (extending)

    max_dwing_angle_in: whichever wing is folding faster (or extending slower)
    min_dwing_angle_in: whichever wing is extending faster (or folding slower)
    """
    wingL = features["wingL"]   # (T, n_flies)
    wingR = features["wingR"]

    # Inward velocity: positive = folding
    dL_in =  np.diff(wingL, axis=0) * fps   # (T-1, n_flies)
    dR_in = -np.diff(wingR, axis=0) * fps

    nan_row = np.full((1, wingL.shape[1]), np.nan)
    return {
        "max_dwing_angle_in": np.concatenate([nan_row, np.maximum(dL_in, dR_in)], axis=0).astype(np.float64),
        "min_dwing_angle_in": np.concatenate([nan_row, np.minimum(dL_in, dR_in)], axis=0).astype(np.float64),
    }

def compute_minmax_dwing_angle_out(tracks, features=None, fps=30, **kwargs):
    """
    Min and max of outward wing angular velocity across left and right wings (rad/s).

    Positive = wing moving away from body (extending/opening)
    Negative = wing moving toward body (folding)

    max_dwing_angle_out: whichever wing is extending faster
    min_dwing_angle_out: whichever wing is folding faster
    """
    wingL = features["wingL"]
    wingR = features["wingR"]

    # Outward velocity: positive = extending
    dL_out = -np.diff(wingL, axis=0) * fps   # (T-1, n_flies)
    dR_out =  np.diff(wingR, axis=0) * fps

    nan_row = np.full((1, wingL.shape[1]), np.nan)
    return {
        "max_dwing_angle_out": np.concatenate([nan_row, np.maximum(dL_out, dR_out)], axis=0).astype(np.float64),
        "min_dwing_angle_out": np.concatenate([nan_row, np.minimum(dL_out, dR_out)], axis=0).astype(np.float64),
    }