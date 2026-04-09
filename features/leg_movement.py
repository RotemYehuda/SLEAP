import numpy as np

# ---------------------------------------------------------------------------
# Angular velocity features
# ---------------------------------------------------------------------------

def compute_dfront_leg_angles(tracks, features=None, fps=30, **kwargs):
    """
    Angular velocity of each front leg in the egocentric frame (rad/s).

    Frame-to-frame difference of the egocentric leg angle, scaled by fps.
    Frame 0 is NaN (NaN-prepend convention — no prior frame available).

    Positive values indicate the leg sweeping in the direction of increasing
    angle (rightward / forward, depending on starting posture).
    """
    return {
        "dfront_leg_L": (
            np.diff(features["front_leg_L"], axis=0, prepend=np.nan) * fps
        ).astype(np.float64),
        "dfront_leg_R": (
            np.diff(features["front_leg_R"], axis=0, prepend=np.nan) * fps
        ).astype(np.float64),
    }


def compute_front_leg_abs_angular_speed(tracks, features=None, **kwargs):
    """
    Absolute angular speed of each front leg (rad/s).

    Unsigned version of dfront_leg_L / dfront_leg_R.  Useful when the
    direction of swing is not important, only its magnitude.
    Frame 0 is NaN (inherited from the velocity features).
    """
    return {
        "abs_dfront_leg_L": np.abs(features["dfront_leg_L"]).astype(np.float64),
        "abs_dfront_leg_R": np.abs(features["dfront_leg_R"]).astype(np.float64),
    }