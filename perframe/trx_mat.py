# Export trajectory struct → trx.mat

from pathlib import Path
import h5py
import numpy as np

from perframe.ds_utils import safe_savemat
from params import FPS, PXPERMM, NODE_IDX_HEAD, NODE_IDX_THORAX, NODE_IDX_ABDOMEN, NODE_IDX_L_WING, NODE_IDX_R_WING

def save_trx(features_source: Path, trx_dest: Path, timestamps: np.ndarray | None = None, overwrite: bool = False) -> Path:
    features_h5 = Path(features_source)
    trx_dest = Path(trx_dest)

    if not features_h5.is_file():
        raise FileNotFoundError(f"features file not found: {features_h5}")

    if trx_dest.exists():
        if overwrite:
            trx_dest.unlink()
        else:
            raise FileExistsError(f"trx already exists: {trx_dest}")

    head_index       = NODE_IDX_HEAD
    thorax_index     = NODE_IDX_THORAX
    abdomen_index    = NODE_IDX_ABDOMEN
    left_wing_index  = NODE_IDX_L_WING
    right_wing_index = NODE_IDX_R_WING

    fields = [
        "moviename",
        "moviefile",
        "nframes",
        "firstframe",
        "endframe",
        "off",
        "id",
        "label",
        "fps",
        "pxpermm",
        "timestamps",
        "dt",
        "x",
        "y",
        "a",
        "b",
        "theta",
        "x_mm",
        "y_mm",
        "a_mm",
        "b_mm",
        "theta_mm",
        "xwingl",
        "ywingl",
        "xwingr",
        "ywingr",
    ]

    with h5py.File(features_h5, "r") as f:
        # ----- meta -----
        fps = FPS
        pxpermm = PXPERMM
        track_names = None

        if "meta" in f:
            meta = f["meta"]
            if "fps" in meta:
                fps = float(meta["fps"][()])
            if "pxpermm" in meta:
                pxpermm = float(meta["pxpermm"][()])
            if "track_names" in meta:
                track_names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                               for x in meta["track_names"][:]]
            if "node_names" in meta:
                stored_names = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                                for x in meta["node_names"][:]]
                node_map = {name: idx for idx, name in enumerate(stored_names)}
                head_index       = node_map.get("head",    head_index)
                thorax_index     = node_map.get("thorax",  thorax_index)
                abdomen_index    = node_map.get("abdomen", abdomen_index)
                left_wing_index  = node_map.get("L_wing",  left_wing_index)
                right_wing_index = node_map.get("R_wing",  right_wing_index)

        # ----- pose/tracks -----
        if "pose" not in f or "tracks" not in f["pose"]:
            raise KeyError("Expected group '/pose/tracks' in features.h5")

        tracks_group = f["pose"]["tracks"]
        fly_keys = sorted(list(tracks_group.keys()))  # fly_000, fly_001, ...
        if len(fly_keys) == 0:
            raise ValueError("No fly datasets found in /pose/tracks")

        # Try to infer movie file in the experiment folder
        expt_dir = features_h5.parent
        movie_files = sorted(expt_dir.glob("movie.*"))
        if len(movie_files) > 1:
            raise ValueError(f"Multiple movie files found in {expt_dir}: {movie_files}")
        movie_path = movie_files[0] if movie_files else None

        trx_list = []
        for i, fly_key in enumerate(fly_keys):
            track = tracks_group[fly_key][:]  # (frames, nodes, xy)
            if track.ndim != 3 or track.shape[2] != 2:
                raise ValueError(f"Unexpected track shape for {fly_key}: {track.shape}")

            nframes = int(track.shape[0])

            if track_names is not None and i < len(track_names):
                label_name = track_names[i]
            else:
                label_name = f"fly_{i}"

            # timestamps
            if timestamps is not None:
                ts = np.asarray(timestamps).squeeze()
                if ts.shape[0] < nframes:
                    raise ValueError("Provided timestamps shorter than nframes")
                ts = ts[:nframes]
            else:
                ts = np.arange(nframes, dtype=np.float64) / float(fps)

            dt = np.diff(ts)

            # core pose fields
            thorax_pos = track[:, thorax_index, :]  # (n,2)
            head_pos = track[:, head_index, :]
            abdomen_pos = track[:, abdomen_index, :]
            left_wing_pos = track[:, left_wing_index, :]
            right_wing_pos = track[:, right_wing_index, :]

            x = thorax_pos[:, 0].astype(np.float64)
            y = thorax_pos[:, 1].astype(np.float64)

            # heading relative to x-axis in radians [-π, π]
            direction_vec = head_pos - thorax_pos
            theta = np.arctan2(direction_vec[:, 1], direction_vec[:, 0]).astype(np.float64)
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            # "a" and "b"
            a = (np.sqrt(np.sum((head_pos - abdomen_pos) ** 2, axis=1)) / 4).astype(np.float64)
            b = (np.sqrt(np.sum((left_wing_pos - right_wing_pos) ** 2, axis=1)) / 4).astype(np.float64)

            # px -> mm fields
            x_mm = x / float(pxpermm)
            y_mm = y / float(pxpermm)
            a_mm = a / float(pxpermm)
            b_mm = b / float(pxpermm)
            theta_mm = theta  # heading in radians; no unit conversion (angles are scale-invariant)

            xwingl = left_wing_pos[:, 0]
            ywingl = left_wing_pos[:, 1]

            xwingr = right_wing_pos[:, 0]
            ywingr = right_wing_pos[:, 1]

            trx_entry: dict[str, object] = {"moviename": movie_path.name if movie_path else "movie",
                                            "moviefile": str(movie_path) if movie_path else "movie",
                                            "fps": float(fps),
                                            "pxpermm": float(pxpermm),
                                            "id": int(i + 1),
                                            "label": label_name,
                                            "firstframe": 1.0,
                                            "off": 0.0,
                                            "nframes": float(nframes),
                                            "endframe": float(nframes),
                                            "timestamps": ts,
                                            "dt": dt,
                                            "x": x,
                                            "y": y,
                                            "theta": theta,
                                            "a": a, "b": b,
                                            "x_mm": x_mm,
                                            "y_mm": y_mm,
                                            "theta_mm": theta_mm,
                                            "a_mm": a_mm,
                                            "b_mm": b_mm,
                                            "xwingl": xwingl,
                                            "ywingl": ywingl,
                                            "xwingr": xwingr,
                                            "ywingr": ywingr,}


            trx_list.append(trx_entry)

    # MATLAB struct array
    composite_dtype = [(fld, "O") for fld in fields]
    structured_vals = [tuple(entry[fld] for fld in fields) for entry in trx_list]
    trx_struct_array = np.array(structured_vals, dtype=composite_dtype)

    save_dict = {
        "trx": trx_struct_array,
        "timestamps": trx_list[0]["timestamps"],
    }

    safe_savemat(trx_dest, save_dict)
    return trx_dest