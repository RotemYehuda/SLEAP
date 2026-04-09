# Movement dynamics (heading, velocity, angular rate)

import numpy as np

# heading relative to x-axis in radians [-pi, pi]
def compute_theta(tracks, features=None, ctr_ind=1, fwd_ind=0, **kwargs):
    delta = tracks[:,fwd_ind,:,:] - tracks[:,ctr_ind,:,:]
    theta = np.arctan2(delta[:,:,1], delta[:,:,0])

    return {
        "theta": theta.astype(np.float64),
    }

# change in body orientation
def compute_dtheta(tracks, features=None, fps=30, **kwargs):
    theta = features["theta"]
    dtheta = np.diff(theta, axis=0, prepend=np.nan)
    dtheta = ((dtheta + np.pi) % (2 * np.pi) - np.pi) * fps

    return {
        "dtheta": dtheta.astype(np.float64),
    }

# Angular speed
def compute_absdtheta(tracks, features=None, **kwargs):
    dtheta = features["dtheta"]
    absdtheta = np.abs(dtheta)

    return {
        "absdtheta": absdtheta.astype(np.float64),
    }

def center_of_rotation2(x_mm, y_mm, a_mm, b_mm, theta, N=100):
    """
    Vectorized Python port of center_of_rotation2.m.
    """
    T = len(x_mm)
    cost = np.cos(theta)
    sint = np.sin(theta)

    # --- Build the 2x2 system M per frame-pair ----------------------------
    # M encodes how rfrac maps to CoR world displacement.
    # The columns correspond to the major and minor axis contributions.
    dacost = 2 * np.diff(a_mm * cost)   # (T-1,)
    dbcost = 2 * np.diff(b_mm * cost)
    dasint = 2 * np.diff(a_mm * sint)
    dbsint = 2 * np.diff(b_mm * sint)

    # Determinant of M
    Z = dacost * dbcost + dbsint * dasint   # (T-1,)

    # M^{-1} stored as 4 rows: [m11, m12, m21, m22]
    safe_Z = np.where(Z == 0, 1.0, Z)          # avoid divide-by-zero
    m11 =  dbcost / safe_Z
    m21 =  dbsint / safe_Z
    m12 = -dasint / safe_Z
    m22 =  dacost / safe_Z

    # Right-hand side: centroid displacement
    dx = np.diff(x_mm)   # (T-1,)
    dy = np.diff(y_mm)

    # rfrac = -Minv @ [dx; dy]
    rfrac = np.zeros((2, T - 1), dtype=np.float64)
    rfrac[0] = -(m11 * dx + m12 * dy)   # maj component
    rfrac[1] = -(m21 * dx + m22 * dy)   # min component

    # NaN where Z==0 (no rotation) → clamp to 0 (body center)
    rfrac[:, Z == 0] = 0.0
    rfrac = np.nan_to_num(rfrac, nan=0.0)

    # --- Out-of-bounds fallback: search ellipse boundary ------------------
    isoutofbounds = (rfrac[0] ** 2 + rfrac[1] ** 2) > 1.0
    isonfly = ~isoutofbounds
    idx = np.where(isoutofbounds)[0]    # frame indices where CoR is outside

    if idx.size > 0:
        psi = np.linspace(0, 2 * np.pi, N, endpoint=False)  # (N,)
        cospsi = np.cos(psi)   # (N,)
        sinpsi = np.sin(psi)

        # Ellipse boundary world coords at frame t  (N, n_out)
        def ellipse_boundary(t_idx):
            # t_idx: 1-D array of frame indices
            # returns x_bnd, y_bnd of shape (N, len(t_idx))
            x_bnd = (x_mm[t_idx][None, :]
                     + 2 * a_mm[t_idx][None, :] * cost[t_idx][None, :] * cospsi[:, None]
                     - 2 * b_mm[t_idx][None, :] * sint[t_idx][None, :] * sinpsi[:, None])
            y_bnd = (y_mm[t_idx][None, :]
                     + 2 * a_mm[t_idx][None, :] * sint[t_idx][None, :] * cospsi[:, None]
                     + 2 * b_mm[t_idx][None, :] * cost[t_idx][None, :] * sinpsi[:, None])
            return x_bnd, y_bnd

        x1, y1 = ellipse_boundary(idx)        # frame t
        x2, y2 = ellipse_boundary(idx + 1)    # frame t+1

        # Squared distance between matching boundary points
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2  # (N, n_out)

        j = np.argmin(d, axis=0)               # (n_out,) best ψ index per pair
        rfrac[0, idx] = cospsi[j]
        rfrac[1, idx] = sinpsi[j]

    return rfrac, isonfly


def _rfrac2center(x_mm, y_mm, a_mm, b_mm, theta_mm, rfrac_maj, rfrac_min):
    """
    Python port of rfrac2center.m.

    rfrac_maj, rfrac_min : (T-1,) arrays — CoR fraction for each interval
    All other inputs     : (T,)   arrays — per-frame body state

    Returns x1, y1 (CoR at frame t) and x2, y2 (CoR at frame t+1),
    each of shape (T-1,).
    """
    ct1 = np.cos(theta_mm[:-1]);  st1 = np.sin(theta_mm[:-1])
    ct2 = np.cos(theta_mm[1:]);   st2 = np.sin(theta_mm[1:])

    x1 = (x_mm[:-1]
          + rfrac_maj * a_mm[:-1] * 2 * ct1
          - rfrac_min * b_mm[:-1] * 2 * st1)
    y1 = (y_mm[:-1]
          + rfrac_maj * a_mm[:-1] * 2 * st1
          + rfrac_min * b_mm[:-1] * 2 * ct1)

    x2 = (x_mm[1:]
          + rfrac_maj * a_mm[1:] * 2 * ct2
          - rfrac_min * b_mm[1:] * 2 * st2)
    y2 = (y_mm[1:]
          + rfrac_maj * a_mm[1:] * 2 * st2
          + rfrac_min * b_mm[1:] * 2 * ct2)

    return x1, y1, x2, y2


def compute_corfrac(tracks, features=None, **kwargs):
    """
    Center-of-rotation fractional offset along major and minor body axes.
    Shape: (T, n_flies), row 0 is NaN (no prior frame).
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    a_mm  = features["a_mm"]
    b_mm  = features["b_mm"]
    theta = features["theta"]

    T, n_flies = x_mm.shape
    corfrac_maj = np.full((T, n_flies), np.nan)
    corfrac_min = np.full((T, n_flies), np.nan)

    for f in range(n_flies):
        rfrac, _ = center_of_rotation2(
            x_mm[:, f], y_mm[:, f],
            a_mm[:, f], b_mm[:, f],
            theta[:, f],
        )
        corfrac_maj[1:, f] = rfrac[0]
        corfrac_min[1:, f] = rfrac[1]

    return {
        "corfrac_maj": corfrac_maj.astype(np.float64),
        "corfrac_min": corfrac_min.astype(np.float64),
    }

def _cor_displacement(tracks, features):
    """
    Compute per-fly CoR displacement vectors for consecutive frame pairs.
    Returns dx_cor, dy_cor of shape (T-1, n_flies).
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    theta = features["theta"]
    a     = features["a_mm"]
    b     = features["b_mm"]
    rfrac_maj = features["corfrac_maj"][1:, :]
    rfrac_min = features["corfrac_min"][1:, :]

    n_flies = x_mm.shape[1]
    dx_cor  = np.full((x_mm.shape[0] - 1, n_flies), np.nan)
    dy_cor  = np.full((x_mm.shape[0] - 1, n_flies), np.nan)

    for f in range(n_flies):
        x1, y1, x2, y2 = _rfrac2center(
            x_mm[:, f], y_mm[:, f],
            a[:, f], b[:, f], theta[:, f],
            rfrac_maj[:, f], rfrac_min[:, f],
        )
        dx_cor[:, f] = x2 - x1
        dy_cor[:, f] = y2 - y1

    return dx_cor, dy_cor

def _cor_velocity(tracks, features, fps, project_lateral):
    """
    project_lateral=False → project onto theta       (du_cor)
    project_lateral=True  → project onto theta+pi/2  (dv_cor)
    """
    dx_cor, dy_cor = _cor_displacement(tracks, features)
    theta_t = features["theta"][:-1, :]
    angle   = theta_t + (np.pi / 2 if project_lateral else 0.0)
    result  = (dx_cor * np.cos(angle) + dy_cor * np.sin(angle)) * fps

    nan_row = np.full((1, tracks.shape[-1]), np.nan)
    return np.concatenate([nan_row, result], axis=0).astype(np.float64)

def compute_dv_cor(tracks, features=None, fps=30, **kwargs):
    """Sideways velocity of the center of rotation (mm/s)."""
    return {"dv_cor": _cor_velocity(tracks, features, fps, project_lateral=True)}


def compute_absdv_cor(tracks, features=None, **kwargs):
    """Absolute sideways velocity of the center of rotation (mm/s)."""
    return {"absdv_cor": np.abs(features["dv_cor"]).astype(np.float64)}

def compute_du_cor(tracks, features=None, fps=30, **kwargs):
    """Forward velocity of the center of rotation (mm/s)."""
    return {"du_cor": _cor_velocity(tracks, features, fps, project_lateral=False)}

def _point_velocity(tracks, features, fps, lateral, use_tail):
    """
    Shared core for centroid/tail × forward/sideways velocity.

    use_tail=False  → track centroid
    use_tail=True   → track tail point (centroid - 2a along heading)
    lateral=False   → project onto theta        (forward)
    lateral=True    → project onto theta + pi/2 (sideways)
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    theta = features["theta"]

    if use_tail:
        a_mm = features["a_mm"]
        x_mm = x_mm - 2 * np.cos(-theta) * a_mm
        y_mm = y_mm - 2 * np.sin(-theta) * a_mm

    dx = np.diff(x_mm, axis=0)
    dy = np.diff(y_mm, axis=0)

    angle = theta[:-1, :] + (np.pi / 2 if lateral else 0.0)
    result = (dx * np.cos(angle) + dy * np.sin(angle)) * fps

    nan_row = np.full((1, tracks.shape[-1]), np.nan)
    return np.concatenate([nan_row, result], axis=0).astype(np.float64)

def compute_du_ctr(tracks, features=None, fps=30, **kwargs):
    """Forward velocity of the body center (mm/s)."""
    return {"du_ctr": _point_velocity(tracks, features, fps, lateral=False, use_tail=False)}

def compute_dv_ctr(tracks, features=None, fps=30, **kwargs):
    """Sideways velocity of the body center (mm/s)."""
    return {"dv_ctr": _point_velocity(tracks, features, fps, lateral=True,  use_tail=False)}

def compute_du_tail(tracks, features=None, fps=30, **kwargs):
    """Forward velocity of the tail point (mm/s)."""
    return {"du_tail": _point_velocity(tracks, features, fps, lateral=False, use_tail=True)}

def compute_dv_tail(tracks, features=None, fps=30, **kwargs):
    """Sideways velocity of the tail point (mm/s)."""
    return {"dv_tail": _point_velocity(tracks, features, fps, lateral=True,  use_tail=True)}

def compute_signdtheta(tracks, features=None, **kwargs):
    """Sign of body orientation change rate. +1 turning left, -1 turning right."""
    return {
        "signdtheta": np.sign(features["dtheta"]).astype(np.float64)
    }

def compute_flipdv_cor(tracks, features=None, **kwargs):
    """
    Sideways CoR velocity signed by turning direction (mm/s).
    Positive = sideways motion in the same direction as the body is turning.
    """
    return {
        "flipdv_cor": (features["dv_cor"] * features["signdtheta"]).astype(np.float64)
    }

def compute_velmag(tracks, features=None, fps=30, **kwargs):
    """Speed of center of rotation (mm/s). Falls back to velmag_ctr where CoR is NaN."""
    dx_cor, dy_cor = _cor_displacement(tracks, features)

    mag      = np.sqrt(dx_cor ** 2 + dy_cor ** 2) * fps
    bad      = np.isnan(dx_cor)
    for f in range(tracks.shape[-1]):
        mag[bad[:, f], f] = features["velmag_ctr"][1:, f][bad[:, f]]

    nan_row = np.full((1, tracks.shape[-1]), np.nan)
    return {
        "velmag": np.concatenate([nan_row, mag], axis=0).astype(np.float64)
    }

def _point_speed(tracks, features, fps, use_nose=False, use_tail=False):
    """
    Shared core for velmag_ctr / velmag_nose / velmag_tail.
    Computes speed (magnitude of displacement) of a body point.

    use_nose=False, use_tail=False → centroid
    use_nose=True                  → nose  (centroid + 2a along +theta)
    use_tail=True                  → tail  (centroid + 2a along -theta)
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    theta = features["theta"]

    if use_nose or use_tail:
        a_mm  = features["a_mm"]
        sign  = 1.0 if use_nose else -1.0
        x_mm  = x_mm + sign * 2 * np.cos(theta) * a_mm
        y_mm  = y_mm + sign * 2 * np.sin(theta) * a_mm

    dx = np.diff(x_mm, axis=0)   # (T-1, n_flies)
    dy = np.diff(y_mm, axis=0)

    speed   = np.sqrt(dx ** 2 + dy ** 2) * fps
    nan_row = np.full((1, tracks.shape[-1]), np.nan)
    return np.concatenate([nan_row, speed], axis=0).astype(np.float64)

def compute_velmag_ctr(tracks, features=None, fps=30, **kwargs):
    """Speed of body centroid (mm/s)."""
    return {"velmag_ctr": _point_speed(tracks, features, fps)}

def compute_velmag_nose(tracks, features=None, fps=30, **kwargs):
    """Speed of nose point — centroid + 2a along +theta (mm/s)."""
    return {"velmag_nose": _point_speed(tracks, features, fps, use_nose=True)}

def compute_velmag_tail(tracks, features=None, fps=30, **kwargs):
    """Speed of tail point — centroid + 2a along -theta (mm/s)."""
    return {"velmag_tail": _point_speed(tracks, features, fps, use_tail=True)}

