# Velocity direction (phi, yaw)

import numpy as np

# velocity direction
def compute_phi(tracks, features=None, ctr_ind=1, **kwargs):
    """
    Direction of the velocity vector (movement direction) in radians [-pi, pi].
    """
    x = tracks[:, ctr_ind, 0, :]   # (T, n_flies)
    y = tracks[:, ctr_ind, 1, :]

    dx = np.diff(x, axis=0)        # (T-1, n_flies)
    dy = np.diff(y, axis=0)

    phi_diff = np.arctan2(dy, dx)  # (T-1, n_flies)

    # Prepend NaN row to keep shape (T, n_flies), consistent with theta
    nan_row = np.full((1, tracks.shape[-1]), np.nan)
    phi = np.concatenate([nan_row, phi_diff], axis=0)

    return {
        "phi": phi.astype(np.float64),
    }

# change in velocity direction
def compute_dphi(tracks, features=None, fps=30, **kwargs):
    """
    Rate of change of velocity direction (rad/s).
    """
    phi = features["phi"]

    dphi = np.diff(phi, axis=0, prepend=np.nan)
    dphi = ((dphi + np.pi) % (2 * np.pi) - np.pi) * fps

    return {
        "dphi": dphi.astype(np.float64),
    }

def compute_phisideways(tracks, features=None, **kwargs):
    """
    Sidewaysness of the velocity direction (rad), range [0, pi/2].
    Measures how much the fly is moving sideways vs forward/backward.
    """
    phi   = features["phi"]      # velocity direction
    theta = features["theta"]    # body orientation

    diff = phi - theta
    # wrap into [-pi/2, pi/2]: maps forward(0) and backward(pi) both to 0
    wrapped = ((diff + np.pi / 2) % np.pi) - np.pi / 2

    return {
        "phisideways": np.abs(wrapped).astype(np.float64)
    }

def compute_yaw(tracks, features=None, **kwargs):
    """
    Signed difference between velocity direction and body orientation (rad).
    Range [-pi, pi].

    0        = moving straight forward
    +pi/2    = moving purely left (sideways)
    +-pi     = moving straight backward

    Unsigned/folded version is phisideways.
    """
    phi   = features["phi"]
    theta = features["theta"]

    yaw = (phi - theta + np.pi) % (2 * np.pi) - np.pi

    return {
        "yaw": yaw.astype(np.float64)
    }

def compute_absyaw(tracks, features=None, **kwargs):
    """
    Absolute value of yaw (rad), range [0, pi].
    Measures how much the fly is moving forward/backward vs sideways.
    """
    yaw = features["yaw"]
    return {
        "absyaw": np.abs(yaw).astype(np.float64)
    }