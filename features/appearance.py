# Body geometry (position, axes, area, eccentricity)

import numpy as np

def compute_xy(tracks, features=None, ctr_ind=1, pxpermm=10.5, **kwargs):
    x = tracks[:, ctr_ind, 0, :] / float(pxpermm)
    y = tracks[:, ctr_ind, 1, :] / float(pxpermm)

    return {
        "x_mm": x.astype(np.float64),
        "y_mm": y.astype(np.float64),
    }

def compute_ab(tracks, features=None, fwd_ind=0, abdomen_idx=2, left_wing_idx=10, right_wing_idx=9, pxpermm=10.5, **kwargs):
    """
    Compute body axis quarter-lengths.

    a_mm : quarter body length (mm)
        One quarter of the head-to-abdomen distance (i.e. distance / 4 / pxpermm).
        The body ellipse semi-major axis = 2*a_mm.
        The nose is at centroid + 2*a_mm along theta,
        the tail at centroid - 2*a_mm along theta.

    b_mm : quarter wingspan (mm)
        One quarter of the left-to-right wing-tip distance.
        The body ellipse semi-minor axis = 2*b_mm.

    Note: semi-major axis = 2*a_mm, semi-minor axis = 2*b_mm,
          full major axis = 4*a_mm, full minor axis = 4*b_mm.
    """
    a = (np.sqrt(np.sum((tracks[:,fwd_ind,:,:] - tracks[:,abdomen_idx,:,:])**2, axis=1)) / 4).astype(np.float64)
    b = (np.sqrt(np.sum((tracks[:,left_wing_idx,:,:] - tracks[:,right_wing_idx,:,:])**2, axis=1)) / 4).astype(np.float64)

    a_mm = a / float(pxpermm)
    b_mm = b / float(pxpermm)

    return {
        "a_mm": a_mm,
        "b_mm": b_mm,
    }

def compute_dab(tracks, features=None, fps=30, **kwargs):
    """
    Compute the rate of change of the body semi-axes.

    da : rate of change of a_mm (mm/s)
    db : rate of change of b_mm (mm/s)
    """
    a_mm = features["a_mm"]
    b_mm = features["b_mm"]

    da = np.diff(a_mm, axis=0, prepend=np.nan) * fps
    db = np.diff(b_mm, axis=0, prepend=np.nan) * fps

    return {
        "da": da.astype(np.float64),
        "db": db.astype(np.float64),
    }

# Area of the ellipse
def compute_area(tracks, features=None, **kwargs):
    a_mm = features["a_mm"]
    b_mm = features["b_mm"]

    area = np.pi * (2 * a_mm) * (2 * b_mm)
    return {
        "area": area.astype(np.float64),
    }

# Change in area from frame t to t+1
def compute_darea(tracks, features=None, fps=30, **kwargs):
    area = features["area"]
    darea = np.diff(area, axis=0, prepend=np.nan) * fps
    return {
        "darea": darea.astype(np.float64),
    }

#  Eccentricity of the ellipse
def compute_ecc(tracks, features=None, **kwargs):
    a_mm = features["a_mm"]
    b_mm = features["b_mm"]

    ecc = b_mm / np.maximum(a_mm, 1e-6)
    # ecc = np.sqrt(1 - (b_mm / a_mm) ** 2)
    return {
        "ecc": ecc.astype(np.float64),
    }

# Change in the eccentricity of the ellipse from frame t to t+1
def compute_decc(tracks, features=None, fps=30, **kwargs):
    ecc = features["ecc"]
    decc = np.diff(ecc, axis=0, prepend=np.nan) * fps
    return {
        "decc": decc.astype(np.float64),
    }

def compute_nose_tail(tracks, features=None, **kwargs):
    """
    World-space nose position in mm.
    Nose = centroid + 2*a along forward heading (theta).
    Tail = centroid + 2*a along backward heading (-theta).
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    a_mm  = features["a_mm"]
    theta = features["theta"]

    return {
        "nose_x_mm": (x_mm + 2 * a_mm * np.cos(theta)).astype(np.float64),
        "nose_y_mm": (y_mm + 2 * a_mm * np.sin(theta)).astype(np.float64),
        "tail_x_mm": (x_mm - 2 * a_mm * np.cos(theta)).astype(np.float64),
        "tail_y_mm": (y_mm - 2 * a_mm * np.sin(theta)).astype(np.float64),
    }