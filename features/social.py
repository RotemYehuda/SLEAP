# Pairwise interactions (distances, bearing angles)

import numpy as np


def _ellipse_boundary_samples(x, y, a, b, theta, n_samples=20):
    """
    Sample n_samples points uniformly around the boundary of an ellipse.
    """
    psi    = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)  # (S,)
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    ct     = np.cos(theta)
    st     = np.sin(theta)

    # broadcast: psi along axis 0, fly/frame along axis 1+
    # ellipse point = center + 2a*cos(psi)*[cos,sin](theta)
    #                        - 2b*sin(psi)*[-sin,cos](theta)
    px = x + 2 * a * ct * cospsi[:, None] - 2 * b * st * sinpsi[:, None]
    py = y + 2 * a * st * cospsi[:, None] + 2 * b * ct * sinpsi[:, None]
    return px, py

def _ell2point_dist(
    x_ell, y_ell, a_ell, b_ell, theta_ell,   # ellipse fly  (T,)
    x_pt,  y_pt,                               # point fly    (T,)
    valid,                                     # (T,) bool
    n_samples=20,
):
    """
    Minimum distance from an ellipse boundary to a single point, per frame.
    Returns d (T,) and j_min (T,) — index of closest boundary sample.
    NaN where not valid.
    """
    T   = len(x_ell)
    d   = np.full(T, np.nan)
    j   = np.full(T, -1, dtype=int)
    idx = np.where(valid)[0]

    if idx.size == 0:
        return d, j

    px, py = _ellipse_boundary_samples(
        x_ell[idx], y_ell[idx], a_ell[idx], b_ell[idx], theta_ell[idx],
        n_samples=n_samples,
    )
    nx = x_pt[idx][None, :]    # (1, n_valid)
    ny = y_pt[idx][None, :]

    dist       = np.sqrt((px - nx) ** 2 + (py - ny) ** 2)  # (S, n_valid)
    j_min      = np.argmin(dist, axis=0)                    # (n_valid,)
    d[idx]     = dist[j_min, np.arange(len(idx))]
    j[idx]     = j_min
    return d, j

def _pairwise_nose_ell(tracks, features):
    """
    Shared setup for compute_dell2nose and compute_dnose2ell.
    Returns x_mm, y_mm, a_mm, b_mm, theta, nose_x, nose_y — all (T, n_flies).
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    a_mm  = features["a_mm"]
    b_mm  = features["b_mm"]
    theta = features["theta"]
    nose_x = features["nose_x_mm"]
    nose_y = features["nose_y_mm"]
    return x_mm, y_mm, a_mm, b_mm, theta, nose_x, nose_y

def _dell2nose_pair(x1, y1, a1, b1, theta1, xnose2, ynose2, valid, n_samples=20):
    """Distance from fly1's ellipse to fly2's nose."""
    d, _ = _ell2point_dist(x1, y1, a1, b1, theta1, xnose2, ynose2, valid, n_samples)
    return d

def compute_dell2nose(tracks, features=None, n_samples=20, **kwargs):
    """
    For each fly: minimum distance from any point on its body ellipse
    to the nose of the nearest other fly (mm).
    """
    x_mm, y_mm, a_mm, b_mm, theta, nose_x, nose_y = _pairwise_nose_ell(tracks, features)
    T, n_flies = x_mm.shape

    dell2nose           = np.full((T, n_flies), np.nan)
    closestfly_ell2nose = np.full((T, n_flies), np.nan)

    for i1 in range(n_flies):
        # Distance from fly i1's ellipse to every other fly's nose
        d_all = np.full((n_flies, T), np.nan)

        for i2 in range(n_flies):
            if i1 == i2:
                continue

            # Frames where both flies are valid (non-NaN position)
            valid = (
                np.isfinite(x_mm[:, i1]) &
                np.isfinite(x_mm[:, i2])
            )

            d_all[i2, :] = _dell2nose_pair(
                x_mm[:, i1], y_mm[:, i1],
                a_mm[:, i1], b_mm[:, i1],
                theta[:, i1],
                nose_x[:, i2], nose_y[:, i2],
                valid=valid,
                n_samples=n_samples,
            )

        # Closest fly at each frame
        valid_frames = ~np.all(np.isnan(d_all), axis=0)
        closest_idx  = np.full(T, np.nan)
        min_d        = np.full(T, np.nan)
        if valid_frames.any():
            ci = np.nanargmin(d_all[:, valid_frames], axis=0)
            closest_idx[valid_frames] = ci.astype(float)
            min_d[valid_frames]       = d_all[ci, np.arange(T)[valid_frames]]

        dell2nose[:, i1]           = min_d
        closestfly_ell2nose[:, i1] = closest_idx

    return {
        "dell2nose":           dell2nose.astype(np.float64),
        "closestfly_ell2nose": closestfly_ell2nose.astype(np.float64),
    }

def _dnose2ell_pair(xnose1, ynose1, x2, y2, a2, b2, theta2, valid, n_samples=20):
    """Distance from fly1's nose to fly2's ellipse, plus boundary angle."""
    psi = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    d, j = _ell2point_dist(x2, y2, a2, b2, theta2, xnose1, ynose1, valid, n_samples)
    angle = np.where(j >= 0, psi[np.where(j >= 0, j, 0)] - np.pi, np.nan)
    return d, angle


def compute_dnose2ell(tracks, features=None, n_samples=20, **kwargs):
    """
    For each fly: minimum distance from its nose to the ellipse boundary
    of the nearest other fly (mm), plus the angle on that ellipse where
    the closest point lies.
    """
    x_mm, y_mm, a_mm, b_mm, theta, nose_x, nose_y = _pairwise_nose_ell(tracks, features)
    T, n_flies = x_mm.shape

    dnose2ell           = np.full((T, n_flies), np.nan)
    angleonclosestfly   = np.full((T, n_flies), np.nan)
    closestfly_nose2ell = np.full((T, n_flies), np.nan)

    for i1 in range(n_flies):
        mind_i1  = np.full(T, np.inf)
        angle_i1 = np.full(T, np.nan)
        closest  = np.zeros(T, dtype=float)

        for i2 in range(n_flies):
            if i1 == i2:
                continue

            valid = (
                np.isfinite(x_mm[:, i1]) &
                np.isfinite(x_mm[:, i2])
            )

            # fly1's nose → fly2's ellipse
            d_curr, angle_curr = _dnose2ell_pair(
                nose_x[:, i1], nose_y[:, i1],     # fly1 nose
                x_mm[:, i2], y_mm[:, i2],          # fly2 ellipse
                a_mm[:, i2], b_mm[:, i2],
                theta[:, i2],
                valid=valid,
                n_samples=n_samples,
            )

            # Update running minimum — matches MATLAB's element-wise min loop
            better = np.isfinite(d_curr) & (d_curr < mind_i1)
            mind_i1[better]  = d_curr[better]
            angle_i1[better] = angle_curr[better]
            closest[better]  = i2

        # Frames with no valid neighbor
        no_neighbor = ~np.isfinite(mind_i1)
        mind_i1[no_neighbor]  = np.nan
        closest[no_neighbor]  = np.nan

        dnose2ell[:, i1]           = mind_i1
        angleonclosestfly[:, i1]   = angle_i1
        closestfly_nose2ell[:, i1] = closest

    return {
        "dnose2ell":           dnose2ell.astype(np.float64),
        "angleonclosestfly":   angleonclosestfly.astype(np.float64),
        "closestfly_nose2ell": closestfly_nose2ell.astype(np.float64),
    }

def _anglesubtended(
    x1, y1,                      # fly1 centroid (n_valid,)
    x2, y2, a2, b2, theta2,      # fly2 ellipse  (n_valid,)
    fov=np.pi,
    n_samples=100,
):
    """
    Angle subtended by fly2's ellipse as seen from fly1's centroid.
    """
    # Sample fly2's ellipse boundary — (n_samples, n_valid)
    px, py = _ellipse_boundary_samples(
        x2, y2, a2, b2, theta2, n_samples=n_samples
    )

    # Vector from fly1 to each boundary point
    dx = px - x1[None, :]   # (n_samples, n_valid)
    dy = py - y1[None, :]

    # Angle of each boundary point as seen from fly1
    angles = np.arctan2(dy, dx)   # (n_samples, n_valid)

    # Angular spread = max - min angle, wrapped to [0, pi]
    # Use the circular range: sort angles and find largest gap,
    # then the subtended angle = 2*pi - largest_gap
    angles_sorted = np.sort(angles, axis=0)   # (n_samples, n_valid)

    # Gaps between consecutive angles (circular)
    gaps = np.diff(angles_sorted, axis=0)              # (n_samples-1, n_valid)
    last_gap = (angles_sorted[0] + 2*np.pi
                - angles_sorted[-1])[None, :]          # (1, n_valid)
    all_gaps = np.concatenate([gaps, last_gap], axis=0)  # (n_samples, n_valid)

    largest_gap = np.max(all_gaps, axis=0)             # (n_valid,)
    anglesub    = 2 * np.pi - largest_gap              # (n_valid,)

    # Clamp to fov
    anglesub = np.minimum(anglesub, 2 * fov)

    return anglesub


def _anglesub_pair(
    x1, y1,                      # fly1 centroid (T,)
    x2, y2, a2, b2, theta2,      # fly2 ellipse  (T,)
    valid,                       # (T,) bool
    fov=np.pi,
    n_samples=100,
):
    """
    Angle subtended by fly2's ellipse from fly1's position, per frame.
    Returns (T,) array, NaN where not valid.
    """
    T   = len(x1)
    out = np.full(T, np.nan)
    idx = np.where(valid)[0]

    if idx.size == 0:
        return out

    out[idx] = _anglesubtended(
        x1[idx], y1[idx],
        x2[idx], y2[idx], a2[idx], b2[idx], theta2[idx],
        fov=fov, n_samples=n_samples,
    )
    return out


def compute_anglesub(tracks, features=None, fov=np.pi, n_samples=100, **kwargs):
    """
    For each fly: angle subtended by the nearest other fly's ellipse
    as seen from its centroid (rad).

    Closest fly is defined as the one subtending the LARGEST angle.
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    a_mm  = features["a_mm"]
    b_mm  = features["b_mm"]
    theta = features["theta"]
    T, n_flies = x_mm.shape

    anglesub            = np.full((T, n_flies), np.nan)
    closestfly_anglesub = np.full((T, n_flies), np.nan)

    for i1 in range(n_flies):
        as_all = np.full((n_flies, T), np.nan)

        for i2 in range(n_flies):
            if i1 == i2:
                continue
            valid = (
                np.isfinite(x_mm[:, i1]) &
                np.isfinite(x_mm[:, i2])
            )
            as_all[i2] = _anglesub_pair(
                x_mm[:, i1], y_mm[:, i1],
                x_mm[:, i2], y_mm[:, i2],
                a_mm[:, i2], b_mm[:, i2],
                theta[:, i2],
                valid=valid, fov=fov, n_samples=n_samples,
            )

        # Closest = fly subtending the LARGEST angle (max, not min)
        valid_frames = ~np.all(np.isnan(as_all), axis=0)
        closest      = np.full(T, np.nan)
        maxas        = np.full(T, np.nan)
        if valid_frames.any():
            ci = np.nanargmax(as_all[:, valid_frames], axis=0)
            closest[valid_frames] = ci.astype(float)
            maxas[valid_frames]   = as_all[ci, np.arange(T)[valid_frames]]

        anglesub[:, i1]            = maxas
        closestfly_anglesub[:, i1] = closest

    return {
        "anglesub":            anglesub.astype(np.float64),
        "closestfly_anglesub": closestfly_anglesub.astype(np.float64),
    }

def compute_danglesub(tracks, features=None, fps=30, **kwargs):
    """Rate of change of angle subtended by nearest fly (rad/s)."""
    danglesub = np.diff(features["anglesub"], axis=0, prepend=np.nan) * fps
    return {
        "danglesub": danglesub.astype(np.float64)
    }

def _pairwise_min_dist(pt1_x, pt1_y, pt2_x, pt2_y):
    """
    For each focal fly i1: find the fly i2 minimizing distance
    from pt1[:,i1] to pt2[:,i2].

    Parameters
    ----------
    pt1_x, pt1_y : (T, n_flies) — source points (one per fly)
    pt2_x, pt2_y : (T, n_flies) — target points (one per fly)

    Returns
    -------
    mind    : (T, n_flies)  minimum distance to any other fly (mm)
    closest : (T, n_flies)  index of that fly (0-based float, NaN if none)
    """
    T, n_flies = pt1_x.shape
    mind    = np.full((T, n_flies), np.nan)
    closest = np.full((T, n_flies), np.nan)

    for i1 in range(n_flies):
        d_all = np.full((n_flies, T), np.nan)

        for i2 in range(n_flies):
            if i1 == i2:
                continue
            valid = np.isfinite(pt1_x[:, i1]) & np.isfinite(pt2_x[:, i2])
            dx = pt2_x[:, i2] - pt1_x[:, i1]
            dy = pt2_y[:, i2] - pt1_y[:, i1]
            d_all[i2, valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2)

        valid_frames = ~np.all(np.isnan(d_all), axis=0)
        ci           = np.full(T, np.nan)
        md           = np.full(T, np.nan)
        if valid_frames.any():
            c = np.nanargmin(d_all[:, valid_frames], axis=0)
            ci[valid_frames] = c.astype(float)
            md[valid_frames] = d_all[c, np.arange(T)[valid_frames]]
        mind[:, i1]    = md
        closest[:, i1] = ci

    return mind, closest

def compute_dcenter(tracks, features=None, **kwargs):
    """Centroid-to-centroid distance to nearest other fly (mm)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    mind, closest = _pairwise_min_dist(x_mm, y_mm, x_mm, y_mm)
    return {
        "dcenter":           mind.astype(np.float64),
        "closestfly_center": closest.astype(np.float64),
    }

def compute_ddcenter(tracks, features=None, fps=30, **kwargs):
    """Rate of change of distance to the closest fly by centroid (mm/s)."""
    return {
        "ddcenter": (np.diff(features["dcenter"], axis=0, prepend=np.nan) * fps).astype(np.float64)
    }

def compute_dnose2tail(tracks, features=None, **kwargs):
    """Distance from fly1's nose to nearest other fly's tail (mm)."""
    mind, closest = _pairwise_min_dist(
        features["nose_x_mm"], features["nose_y_mm"],
        features["tail_x_mm"], features["tail_y_mm"],
    )
    return {
        "dnose2tail":           mind.astype(np.float64),
        "closestfly_nose2tail": closest.astype(np.float64),
    }

def _abs_angle_diff_to_closest(angle, closestfly):
    """
    Shared core for absphidiff_* and absthetadiff_* features.

    For each fly i1 at each frame t, looks up the closest fly index,
    reads its angle, and computes |modrange(angle2 - angle1, -pi, pi)|.

    Parameters
    ----------
    angle       : (T, n_flies)  any circular angle in radians (phi or theta)
    closestfly  : (T, n_flies)  0-based float index of closest fly, NaN if none

    Returns
    -------
    out : (T, n_flies)  absolute angle difference in [0, pi], NaN where undefined
    """
    T, n_flies = angle.shape
    out = np.full((T, n_flies), np.nan)

    for i1 in range(n_flies):
        cf = closestfly[:, i1]           # (T,) closest fly index, float with NaN
        valid = np.isfinite(cf)          # frames where a closest fly exists
        if not valid.any():
            continue

        idx = np.where(valid)[0]
        i2s = cf[idx].astype(int)        # closest fly index per valid frame

        angle1 = angle[idx, i1]
        angle2 = angle[idx, i2s]         # vectorized: one index per frame

        diff = angle2 - angle1
        out[idx, i1] = np.abs(((diff + np.pi) % (2 * np.pi) - np.pi))

    return out.astype(np.float64)


def compute_absphidiff_anglesub(tracks, features=None, **kwargs):
    """
    Absolute velocity direction difference between focal fly and
    the fly subtending the largest angle (closestfly_anglesub).
    """
    return {
        "absphidiff_anglesub": _abs_angle_diff_to_closest(
            features["phi"],
            features["closestfly_anglesub"],
        )
    }


def compute_absphidiff_nose2ell(tracks, features=None, **kwargs):
    """
    Absolute velocity direction difference between focal fly and
    the nearest fly by nose-to-ellipse distance (closestfly_nose2ell).
    """
    return {
        "absphidiff_nose2ell": _abs_angle_diff_to_closest(
            features["phi"],
            features["closestfly_nose2ell"],
        )
    }

def compute_absthetadiff_anglesub(tracks, features=None, **kwargs):
    """
    Absolute body orientation difference between focal fly and
    the fly subtending the largest angle (closestfly_anglesub).
    """
    return {
        "absthetadiff_anglesub": _abs_angle_diff_to_closest(
            features["theta"],
            features["closestfly_anglesub"],
        )
    }


def compute_absthetadiff_nose2ell(tracks, features=None, **kwargs):
    """
    Absolute body orientation difference between focal fly and
    the nearest fly by nose-to-ellipse distance (closestfly_nose2ell).
    """
    return {
        "absthetadiff_nose2ell": _abs_angle_diff_to_closest(
            features["theta"],
            features["closestfly_nose2ell"],
        )
    }

def _anglefrom1to2(nose_x, nose_y, theta, x_mm, y_mm, closestfly):
    """
    Shared core for anglefrom1to2_* features.

    Bearing from fly1's nose to fly2's centroid, in fly1's egocentric frame.
    modrange(atan2(dy, dx) - theta1, -pi, pi)

    Parameters
    ----------
    nose_x, nose_y : (T, n_flies)  fly nose positions (mm)
    theta          : (T, n_flies)  fly heading (rad)
    x_mm, y_mm     : (T, n_flies)  fly centroid positions (mm)
    closestfly     : (T, n_flies)  0-based float index, NaN if none

    Returns
    -------
    out : (T, n_flies)  bearing angle in [-pi, pi], NaN where undefined
    """
    T, n_flies = theta.shape
    out = np.full((T, n_flies), np.nan)

    for i1 in range(n_flies):
        cf    = closestfly[:, i1]
        valid = np.isfinite(cf)
        if not valid.any():
            continue

        idx = np.where(valid)[0]
        i2s = cf[idx].astype(int)

        dx = x_mm[idx, i2s] - nose_x[idx, i1]
        dy = y_mm[idx, i2s] - nose_y[idx, i1]

        theta2 = np.arctan2(dy, dx)
        diff   = theta2 - theta[idx, i1]
        out[idx, i1] = (diff + np.pi) % (2 * np.pi) - np.pi

    return out.astype(np.float64)


def compute_anglefrom1to2_anglesub(tracks, features=None, **kwargs):
    """Bearing from fly1's nose to the closest fly (by anglesub) centroid, egocentric (rad)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    return {
        "anglefrom1to2_anglesub": _anglefrom1to2(
            features["nose_x_mm"], features["nose_y_mm"],
            features["theta"], x_mm, y_mm,
            features["closestfly_anglesub"],
        )
    }


def compute_anglefrom1to2_nose2ell(tracks, features=None, **kwargs):
    """Bearing from fly1's nose to the closest fly (by nose2ell) centroid, egocentric (rad)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    return {
        "anglefrom1to2_nose2ell": _anglefrom1to2(
            features["nose_x_mm"], features["nose_y_mm"],
            features["theta"], x_mm, y_mm,
            features["closestfly_nose2ell"],
        )
    }

def compute_absanglefrom1to2_nose2ell(tracks, features=None, **kwargs):
    """Absolute bearing from fly1's nose to closest fly (by nose2ell), egocentric (rad)."""
    return {
        "absanglefrom1to2_nose2ell": np.abs(features["anglefrom1to2_nose2ell"]).astype(np.float64)
    }

def _magveldiff(x_mm, y_mm, closestfly, fps):
    """
    Shared core for magveldiff_* features.

    Magnitude of velocity vector difference between fly1 and its closest fly.
    ||(dx1,dy1) - (dx2,dy2)|| * fps  in mm/s.

    Parameters
    ----------
    x_mm, y_mm  : (T, n_flies)  centroid positions in mm
    closestfly  : (T, n_flies)  0-based float index, NaN if none
    fps         : float

    Returns
    -------
    out : (T, n_flies), NaN where undefined. Row 0 is NaN (no prior frame).
    """
    T, n_flies = x_mm.shape
    out = np.full((T, n_flies), np.nan)

    # Frame-to-frame displacement — (T-1, n_flies), aligned to frames 1..T-1
    dx = np.diff(x_mm, axis=0)   # dx[t] = x[t+1] - x[t]
    dy = np.diff(y_mm, axis=0)

    for i1 in range(n_flies):
        # closestfly is defined at each frame; use frames 0..T-2
        cf    = closestfly[:-1, i1]
        valid = np.isfinite(cf)
        if not valid.any():
            continue

        idx = np.where(valid)[0]     # indices into 0..T-2
        i2s = cf[idx].astype(int)

        # fly1 velocity at interval idx→idx+1
        vx1 = dx[idx, i1]
        vy1 = dy[idx, i1]

        # fly2 velocity at same interval
        vx2 = dx[idx, i2s]
        vy2 = dy[idx, i2s]

        # magnitude of difference, scaled to mm/s
        # store at idx+1 to match NaN-prepend convention (frame 0 = NaN)
        out[idx + 1, i1] = np.sqrt((vx1 - vx2) ** 2 + (vy1 - vy2) ** 2) * fps

    return out.astype(np.float64)


def compute_magveldiff_anglesub(tracks, features=None, fps=30, **kwargs):
    """Velocity difference magnitude vs closest fly by anglesub (mm/s)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    return {
        "magveldiff_anglesub": _magveldiff(
            x_mm, y_mm, features["closestfly_anglesub"], fps
        )
    }


def compute_magveldiff_nose2ell(tracks, features=None, fps=30, **kwargs):
    """Velocity difference magnitude vs closest fly by nose2ell (mm/s)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    return {
        "magveldiff_nose2ell": _magveldiff(
            x_mm, y_mm, features["closestfly_nose2ell"], fps
        )
    }

def _veltoward(x_mm, y_mm, closestfly, fps):
    """
    Shared core for veltoward_* features.

    Signed velocity of fly1 in the direction of its closest fly (mm/s).
    Positive = moving toward, negative = moving away.

    Parameters
    ----------
    x_mm, y_mm  : (T, n_flies)  centroid positions in mm
    closestfly  : (T, n_flies)  0-based float index, NaN if none
    fps         : float

    Returns
    -------
    out : (T, n_flies), NaN where undefined. Row 0 is NaN.
    """
    T, n_flies = x_mm.shape
    out = np.full((T, n_flies), np.nan)

    dx = np.diff(x_mm, axis=0)   # (T-1, n_flies) — fly1 velocity
    dy = np.diff(y_mm, axis=0)

    for i1 in range(n_flies):
        cf    = closestfly[:-1, i1]      # evaluate at frames 0..T-2
        valid = np.isfinite(cf)
        if not valid.any():
            continue

        idx = np.where(valid)[0]         # into 0..T-2
        i2s = cf[idx].astype(int)

        # Unit vector from fly1 to fly2 at frame idx
        dx2 = x_mm[idx, i2s] - x_mm[idx, i1]
        dy2 = y_mm[idx, i2s] - y_mm[idx, i1]
        dz2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

        # Normalize — zero distance → zero unit vector (flies on top of each other)
        safe = dz2 > 0
        ux = np.where(safe, dx2 / np.where(safe, dz2, 1.0), 0.0)
        uy = np.where(safe, dy2 / np.where(safe, dz2, 1.0), 0.0)

        # Project fly1 velocity onto unit vector, scale to mm/s
        out[idx + 1, i1] = (dx[idx, i1] * ux + dy[idx, i1] * uy) * fps

    return out.astype(np.float64)


def compute_veltoward_anglesub(tracks, features=None, fps=30, **kwargs):
    """Signed velocity toward the closest fly by anglesub (mm/s)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    return {
        "veltoward_anglesub": _veltoward(
            x_mm, y_mm, features["closestfly_anglesub"], fps
        )
    }


def compute_veltoward_nose2ell(tracks, features=None, fps=30, **kwargs):
    """Signed velocity toward the closest fly by nose2ell (mm/s)."""
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    return {
        "veltoward_nose2ell": _veltoward(
            x_mm, y_mm, features["closestfly_nose2ell"], fps
        )
    }

def compute_nflies_close(tracks, features=None, nbodylengths_near=2.0, **kwargs):
    """
    Number of other flies within nbodylengths_near body lengths of each fly.
    "Close" = centroid distance <= nbodylengths_near * 4 * a_mm(fly1),
    where 4*a_mm is the full body length (2 * major axis).
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    a_mm = features["a_mm"]
    T, n_flies = x_mm.shape

    nflies_close = np.zeros((T, n_flies), dtype=np.float64)

    for i1 in range(n_flies):
        for i2 in range(n_flies):
            if i1 == i2:
                continue

            valid = np.isfinite(x_mm[:, i1]) & np.isfinite(x_mm[:, i2])
            if not valid.any():
                continue

            dx = x_mm[:, i2] - x_mm[:, i1]
            dy = y_mm[:, i2] - y_mm[:, i1]

            # Normalize by fly1's full body length (4 * a_mm)
            body_length = 4.0 * a_mm[:, i1]
            normalized_dist = np.sqrt(dx ** 2 + dy ** 2) / np.where(
                body_length > 0, body_length, np.inf
            )

            isclose = valid & (normalized_dist <= nbodylengths_near)
            nflies_close[isclose, i1] += 1.0

    return {
        "nflies_close": nflies_close.astype(np.float64)
    }