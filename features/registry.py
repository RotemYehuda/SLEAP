# Central registry: 40+ FeatureSpec entries, dependency resolver, validator

from typing import Callable, List, Dict, Literal
from dataclasses import dataclass, field
import numpy as np

from params import (NODE_IDX_HEAD, NODE_IDX_THORAX, NODE_IDX_ABDOMEN,
                    NODE_IDX_L_WING, NODE_IDX_R_WING,
                    NODE_IDX_L_FRONT_LEG, NODE_IDX_R_FRONT_LEG,
                    PXPERMM, FPS)

from features.appearance import (compute_xy, compute_ab, compute_dab,
                                 compute_area, compute_darea,
                                 compute_ecc, compute_decc,
                                 compute_nose_tail)

from features.locomotion import (compute_theta, compute_dtheta, compute_absdtheta,
                                 compute_corfrac, compute_dv_cor, compute_absdv_cor,
                                 compute_dv_ctr, compute_dv_tail,
                                 compute_du_cor, compute_du_ctr, compute_du_tail,
                                 compute_signdtheta, compute_flipdv_cor,
                                 compute_velmag_ctr, compute_velmag, compute_velmag_tail, compute_velmag_nose)

from features.position import (compute_phi, compute_dphi,
                               compute_phisideways, compute_yaw, compute_absyaw)

from features.social import (compute_dell2nose, compute_dnose2ell,
                             compute_anglesub, compute_danglesub,
                             compute_dcenter, compute_ddcenter,
                             compute_dnose2tail,
                             compute_absphidiff_anglesub, compute_absphidiff_nose2ell,
                             compute_absthetadiff_anglesub, compute_absthetadiff_nose2ell,
                             compute_anglefrom1to2_anglesub, compute_anglefrom1to2_nose2ell,
                             compute_absanglefrom1to2_nose2ell,
                             compute_magveldiff_anglesub, compute_magveldiff_nose2ell,
                             compute_veltoward_anglesub, compute_veltoward_nose2ell,
                             compute_nflies_close)

from features.wing_appearance import (compute_wing_angles,compute_mean_wing_angle,
                                      compute_wing_angle_diff, compute_wing_angle_imbalance,
                                      compute_minmax_wing_angle)

from features.wing_movement import (compute_dminmax_wing_angle, compute_minmax_absdwing_angle,
                                    compute_dwing_angle_diff, compute_dwing_angle_imbalance,
                                    compute_minmax_dwing_angle_in, compute_minmax_dwing_angle_out)

from features.leg_appearance import (compute_front_leg_angles, compute_front_leg_angle_diff, compute_front_leg_extension)
from features.leg_movement import (compute_dfront_leg_angles, compute_front_leg_abs_angular_speed)


@dataclass
class FeatureSpec:
    func: Callable
    requires: List[str]
    outputs: List[str]  # keys to save
    units: Dict[str, Dict[str, str]] = field(default_factory=dict)  # per-output metadata
    enabled: bool = True
    save_mode: Literal["scalar", "pose_per_fly", "none"] = "scalar"
    # "scalar"       -> shape (n_frames, n_flies), written as a flat dataset
    #                   via the existing save_keys loop in dataset.py
    # "pose_per_fly" -> shape (n_frames, nodes, 2, n_flies), written as
    #                   per-fly sub-datasets under pose/<registry_key>/fly_NNN
    # "none"         -> intermediate only, not written to HDF5
    intermediates: List[str] = field(default_factory=list)
    # Keys returned by the function that are available in `computed` for
    # downstream features but are NOT written to HDF5.
    params: Dict[str, object] = field(default_factory=dict)
    # Feature-specific configuration parameters passed as kwargs to func.
    # Values here override the function's own defaults.

def validate_registry(registry: Dict[str, FeatureSpec]) -> None:
    errors = []
    seen_outputs: Dict[str, str] = {}
    seen_intermediates: Dict[str, str] = {}

    for name, spec in registry.items():

        # Check 1: valid requires keys; also catch enabled→disabled dependency chains
        for dep in spec.requires:
            if dep not in registry:
                errors.append(
                    f"  '{name}' requires '{dep}', which is not a registered key.\n"
                    f"    Available keys: {sorted(registry.keys())}"
                )
            elif spec.enabled and not registry[dep].enabled:
                errors.append(
                    f"  '{name}' (enabled) requires '{dep}', which is disabled. "
                    f"Either disable '{name}' or re-enable '{dep}'."
                )

        # Check 2: no duplicate output keys
        for out_key in spec.outputs:
            if out_key in seen_outputs:
                errors.append(
                    f"  '{name}' declares output '{out_key}', already declared "
                    f"by '{seen_outputs[out_key]}'."
                )
            else:
                seen_outputs[out_key] = name

        # Check 2b: no duplicate intermediate keys
        for int_key in spec.intermediates:
            if int_key in seen_intermediates:
                errors.append(
                    f"  '{name}' declares intermediate '{int_key}', already declared "
                    f"by '{seen_intermediates[int_key]}'."
                )
            else:
                seen_intermediates[int_key] = name

        # Check 2c: intermediates must not overlap with outputs
        for int_key in spec.intermediates:
            if int_key in seen_outputs:
                errors.append(
                    f"  '{name}' declares '{int_key}' as an intermediate, but it is "
                    f"already declared as an output by '{seen_outputs[int_key]}'."
                )

        # Check 3: save_mode consistency
        if spec.save_mode == "pose_per_fly" and len(spec.outputs) == 0:
            errors.append(
                f"  '{name}' has save_mode='pose_per_fly' but outputs=[].\n"
                f"    Declare the output key(s) this feature writes."
            )
        if spec.save_mode == "scalar" and len(spec.outputs) == 0 and spec.enabled:
            errors.append(
                f"  '{name}' has save_mode='scalar' (default) but outputs=[].\n"
                f"    Set save_mode='none' explicitly if this entry is intentionally intermediate-only."
            )
        # Check 4: no self-dependency
        if name in spec.requires:
            errors.append(
                f"  '{name}' lists itself in requires. Self-dependencies cause "
                f"infinite recursion in compute_item()."
            )

    # Check 5: registry key names do not collide with output key names (unchanged)
    for name in registry.keys():
        if name in seen_outputs and seen_outputs[name] != name:
            errors.append(
                f"  Registry key '{name}' is also declared as an output key by "
                f"'{seen_outputs[name]}'. This causes a collision in the computed dict."
            )

    if errors:
        raise ValueError(
            f"Feature registry validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors)
        )


REGISTRY = {
    # Appearance features
    "xy_mm": FeatureSpec(
        func=compute_xy,
        requires=[],
        outputs=[],
        intermediates=["x_mm", "y_mm"],
        units={},
        enabled=True,
        save_mode="none",
        params={"ctr_ind": NODE_IDX_THORAX, "pxpermm": PXPERMM},
    ),
    "body_scale": FeatureSpec(
        func=compute_ab,
        requires=[],
        outputs=["a_mm", "b_mm"],
        units={},
        enabled=True,
        save_mode="scalar",
        params={"fwd_ind": NODE_IDX_HEAD,
                "abdomen_idx": NODE_IDX_ABDOMEN,
                "leftW_idx": NODE_IDX_L_WING,
                "rightW_idx": NODE_IDX_R_WING,
                "pxpermm": PXPERMM},
    ),
    "dab": FeatureSpec(
        func=compute_dab,
        requires=["body_scale"],
        outputs=["da", "db"],
        units={
            "da": {"quantity": "a_change_rate", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
            "db": {"quantity": "b_change_rate", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS}
    ),
    "area": FeatureSpec(
        func=compute_area,
        requires=["body_scale"],
        outputs=["area"],
        units={
            "area": {"quantity": "area", "unit_raw": "mm^2", "unit_si": "mm^2", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "darea": FeatureSpec(
        func=compute_darea,
        requires=["area"],
        outputs=["darea"],
        units={
            "darea": {"quantity": "area_change_rate", "unit_raw": "mm^2/sec", "unit_si": "mm^2/sec", "scale_expr": "1" },
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS}
    ),
    "eccentricity": FeatureSpec(
        func=compute_ecc,
        requires=["body_scale"],
        outputs=["ecc"],
        units={
            "ecc": {"quantity": "eccentricity", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "deccentricity": FeatureSpec(
        func=compute_decc,
        requires=["eccentricity"],
        outputs=["decc"],
        units={
            "decc": {"quantity": "eccentricity_change_rate", "unit_raw": "unit/sec", "unit_si": "unit/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS}
    ),
    "nose_tail_mm": FeatureSpec(
        func=compute_nose_tail,
        requires=["body_scale", "theta", "xy_mm"],
        outputs=[],
        intermediates=["nose_x_mm", "nose_y_mm", "tail_x_mm", "tail_y_mm"],
        units={
            "nose_x_mm": {"quantity": "position", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "nose_y_mm": {"quantity": "position", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "tail_x_mm": {"quantity": "position", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "tail_y_mm": {"quantity": "position", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="none",
    ),


    # Locomotion features
    "theta": FeatureSpec(
        func=compute_theta,
        requires=[],
        outputs=[],
        intermediates=["theta"],
        units={
            "theta": {"quantity": "orientation", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="none",
        params={"fwd_ind": NODE_IDX_HEAD, "ctr_ind": NODE_IDX_THORAX},
    ),
    "dtheta": FeatureSpec(
        func=compute_dtheta,
        requires=["theta"],
        outputs=["dtheta"],
        units={
            "dtheta": {"quantity": "angular_velocity", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS}
    ),
    "absdtheta": FeatureSpec(
        func=compute_absdtheta,
        requires=["dtheta"],
        outputs=["absdtheta"],
        units={
            "absdtheta": {"quantity": "angular_speed", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "corfrac": FeatureSpec(
        func=compute_corfrac,
        requires=["body_scale", "theta", "xy_mm"],
        outputs=["corfrac_maj", "corfrac_min"],
        units={
            "corfrac_maj": {"quantity": "fractional_offset", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
            "corfrac_min": {"quantity": "fractional_offset", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "dv_cor": FeatureSpec(
        func=compute_dv_cor,
        requires=["corfrac", "body_scale", "theta"],
        outputs=["dv_cor"],
        units={
            "dv_cor": {"quantity": "lateral_velocity_cor", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "absdv_cor": FeatureSpec(
        func=compute_absdv_cor,
        requires=["dv_cor"],
        outputs=["absdv_cor"],
        units={
            "absdv_cor": {
                "quantity": "lateral_speed_cor", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1",
            },
        },
        enabled=True,
        save_mode="scalar",
    ),
    "du_cor": FeatureSpec(
        func=compute_du_cor,
        requires=["corfrac", "body_scale", "theta"],
        outputs=["du_cor"],
        units={
            "du_cor": {"quantity": "forward_velocity_cor", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "du_ctr": FeatureSpec(
        func=compute_du_ctr,
        requires=["theta", "xy_mm"],
        outputs=["du_ctr"],
        units={
            "du_ctr": {"quantity": "forward_velocity_ctr", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "du_tail": FeatureSpec(
        func=compute_du_tail,
        requires=["body_scale", "theta", "xy_mm"],
        outputs=["du_tail"],
        units={
            "du_tail": {"quantity": "forward_velocity_tail", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "dv_ctr": FeatureSpec(
        func=compute_dv_ctr,
        requires=["theta", "xy_mm"],
        outputs=["dv_ctr"],
        units={
            "dv_ctr": {"quantity": "sideways_velocity_ctr", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "dv_tail": FeatureSpec(
        func=compute_dv_tail,
        requires=["body_scale", "theta", "dv_ctr"],
        outputs=["dv_tail"],
        units={
            "dv_tail": {"quantity": "sideways_velocity_tail", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "signdtheta": FeatureSpec(
        func=compute_signdtheta,
        requires=["dtheta"],
        outputs=[],
        intermediates=["signdtheta"],
        units={
            "signdtheta": {"quantity": "turn_sign", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="none",
    ),
    "flipdv_cor": FeatureSpec(
        func=compute_flipdv_cor,
        requires=["dv_cor", "signdtheta"],
        outputs=["flipdv_cor"],
        units={
            "flipdv_cor": {"quantity": "signed_lateral_velocity_cor", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "velmag_ctr": FeatureSpec(
        func=compute_velmag_ctr,
        requires=["theta", "xy_mm"],
        outputs=["velmag_ctr"],
        units={"velmag_ctr": {"quantity": "speed", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"}},
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "velmag": FeatureSpec(
        func=compute_velmag,
        requires=["corfrac", "body_scale", "theta", "velmag_ctr"],
        outputs=["velmag"],
        units={"velmag": {"quantity": "speed", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"}},
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "velmag_nose": FeatureSpec(
        func=compute_velmag_nose,
        requires=["body_scale", "theta", "xy_mm"],
        outputs=["velmag_nose"],
        units={"velmag_nose": {"quantity": "speed", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"}},
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "velmag_tail": FeatureSpec(
        func=compute_velmag_tail,
        requires=["body_scale", "theta", "xy_mm"],
        outputs=["velmag_tail"],
        units={"velmag_tail": {"quantity": "speed", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"}},
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),


    # Position features
    "phi": FeatureSpec(
        func=compute_phi,
        requires=[],
        outputs=["phi"],
        units={
            "phi": {"quantity": "velocity_direction", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"ctr_ind": NODE_IDX_THORAX},
    ),
    "dphi": FeatureSpec(
        func=compute_dphi,
        requires=["phi"],
        outputs=["dphi"],
        units={
            "dphi": {"quantity": "velocity_direction_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "phisideways": FeatureSpec(
        func=compute_phisideways,
        requires=["phi", "theta"],
        outputs=["phisideways"],
        units={
            "phisideways": {"quantity": "sideways_angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "yaw": FeatureSpec(
        func=compute_yaw,
        requires=["phi", "theta"],
        outputs=["yaw"],
        units={
            "yaw": {"quantity": "yaw_angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "absyaw": FeatureSpec(
        func=compute_absyaw,
        requires=["yaw"],
        outputs=["absyaw"],
        units={
            "absyaw": {"quantity": "absolute_yaw_angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),


    # Social features
    "dell2nose": FeatureSpec(
        func=compute_dell2nose,
        requires=["body_scale", "theta"],
        outputs=["dell2nose", "closestfly_ell2nose"],
        units={
            "dell2nose": {"quantity": "distance", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "closestfly_ell2nose": {"quantity": "index", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"n_samples": 20},
    ),
    "dnose2ell": FeatureSpec(
        func=compute_dnose2ell,
        requires=["body_scale", "theta"],
        outputs=["dnose2ell", "angleonclosestfly", "closestfly_nose2ell"],
        units={
            "dnose2ell": {"quantity": "distance", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "angleonclosestfly": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
            "closestfly_nose2ell": {"quantity": "index", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"n_samples": 20},
    ),
    "anglesub": FeatureSpec(
        func=compute_anglesub,
        requires=["body_scale", "theta", "xy_mm"],
        outputs=["anglesub", "closestfly_anglesub"],
        units={
            "anglesub": {"quantity": "angle","unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
            "closestfly_anglesub": {"quantity": "index","unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fov": np.pi, "n_samples": 100},
    ),
    "danglesub": FeatureSpec(
        func=compute_danglesub,
        requires=["anglesub"],
        outputs=["danglesub"],
        units={
            "danglesub": {"quantity": "angle_change_rate","unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "dcenter": FeatureSpec(
        func=compute_dcenter,
        requires=["xy_mm"],
        outputs=["dcenter", "closestfly_center"],
        units={
            "dcenter": {"quantity": "distance", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "closestfly_center": {"quantity": "index", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "ddcenter": FeatureSpec(
        func=compute_ddcenter,
        requires=["dcenter"],
        outputs=["ddcenter"],
        units={
            "ddcenter": {"quantity": "distance_change_rate","unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "dnose2tail": FeatureSpec(
        func=compute_dnose2tail,
        requires=["nose_tail_mm"],
        outputs=["dnose2tail", "closestfly_nose2tail"],
        units={
            "dnose2tail": {"quantity": "distance", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "closestfly_nose2tail": {"quantity": "index", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "absphidiff_anglesub": FeatureSpec(
        func=compute_absphidiff_anglesub,
        requires=["phi", "anglesub"],
        outputs=["absphidiff_anglesub"],
        units={
            "absphidiff_anglesub": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "absphidiff_nose2ell": FeatureSpec(
        func=compute_absphidiff_nose2ell,
        requires=["phi", "dnose2ell"],
        outputs=["absphidiff_nose2ell"],
        units={
            "absphidiff_nose2ell": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "absthetadiff_anglesub": FeatureSpec(
        func=compute_absthetadiff_anglesub,
        requires=["theta", "anglesub"],
        outputs=["absthetadiff_anglesub"],
        units={
            "absthetadiff_anglesub": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "absthetadiff_nose2ell": FeatureSpec(
        func=compute_absthetadiff_nose2ell,
        requires=["theta", "dnose2ell"],
        outputs=["absthetadiff_nose2ell"],
        units={
            "absthetadiff_nose2ell": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "anglefrom1to2_anglesub": FeatureSpec(
        func=compute_anglefrom1to2_anglesub,
        requires=["theta", "nose_tail_mm", "anglesub", "xy_mm"],
        outputs=["anglefrom1to2_anglesub"],
        units={
            "anglefrom1to2_anglesub": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "anglefrom1to2_nose2ell": FeatureSpec(
        func=compute_anglefrom1to2_nose2ell,
        requires=["theta", "nose_tail_mm", "dnose2ell", "xy_mm"],
        outputs=["anglefrom1to2_nose2ell"],
        units={
            "anglefrom1to2_nose2ell": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "absanglefrom1to2_nose2ell": FeatureSpec(
        func=compute_absanglefrom1to2_nose2ell,
        requires=["anglefrom1to2_nose2ell"],
        outputs=["absanglefrom1to2_nose2ell"],
        units={
            "absanglefrom1to2_nose2ell": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "magveldiff_anglesub": FeatureSpec(
        func=compute_magveldiff_anglesub,
        requires=["anglesub", "xy_mm"],
        outputs=["magveldiff_anglesub"],
        units={
            "magveldiff_anglesub": {"quantity": "speed", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "magveldiff_nose2ell": FeatureSpec(
        func=compute_magveldiff_nose2ell,
        requires=["dnose2ell", "xy_mm"],
        outputs=["magveldiff_nose2ell"],
        units={
            "magveldiff_nose2ell": {"quantity": "speed", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "veltoward_anglesub": FeatureSpec(
        func=compute_veltoward_anglesub,
        requires=["anglesub", "xy_mm"],
        outputs=["veltoward_anglesub"],
        units={
            "veltoward_anglesub": {"quantity": "velocity", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "veltoward_nose2ell": FeatureSpec(
        func=compute_veltoward_nose2ell,
        requires=["dnose2ell", "xy_mm"],
        outputs=["veltoward_nose2ell"],
        units={
            "veltoward_nose2ell": {"quantity": "velocity", "unit_raw": "mm/sec", "unit_si": "mm/sec", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "nflies_close": FeatureSpec(
        func=compute_nflies_close,
        requires=["body_scale", "xy_mm"],
        outputs=["nflies_close"],
        units={
            "nflies_close": {"quantity": "count", "unit_raw": "unit", "unit_si": "unit", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
        params={"nbodylengths_near": 2.0},
    ),


    # wing appearance
    "wing_angles": FeatureSpec(
        func=compute_wing_angles,
        requires=["theta"],
        outputs=[],
        intermediates=["wingL", "wingR"],
        units={
            "wingL": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
            "wingR": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="none",
        params={"ctr_ind": NODE_IDX_THORAX, "left_ind": NODE_IDX_L_WING, "right_ind": NODE_IDX_R_WING},
    ),
    "mean_wing_angle": FeatureSpec(
        func=compute_mean_wing_angle,
        requires=["wing_angles"],
        outputs=["mean_wing_angle"],
        units={
            "mean_wing_angle": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "wing_angle_diff": FeatureSpec(
        func=compute_wing_angle_diff,
        requires=["wing_angles"],
        outputs=["wing_angle_diff"],
        units={
            "wing_angle_diff": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "wing_angle_imbalance": FeatureSpec(
        func=compute_wing_angle_imbalance,
        requires=["wing_angles"],
        outputs=["wing_angle_imbalance"],
        units={
            "wing_angle_imbalance": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "minmax_wing_angle": FeatureSpec(
        func=compute_minmax_wing_angle,
        requires=["wing_angles"],
        outputs=["min_wing_angle", "max_wing_angle"],
        units={
             "min_wing_angle": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
             "max_wing_angle": {"quantity": "angle", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),


    # wing movement
    "dminmax_wing_angle": FeatureSpec(
        func=compute_dminmax_wing_angle,
        requires=["wing_angles"],
        outputs=["dmax_wing_angle", "dmin_wing_angle"],
        units={
            "dmax_wing_angle": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
            "dmin_wing_angle": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "minmax_absdwing_angle": FeatureSpec(
        func=compute_minmax_absdwing_angle,
        requires=["wing_angles"],
        outputs=["min_absdwing_angle", "max_absdwing_angle"],
        units={
            "min_absdwing_angle": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
            "max_absdwing_angle": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "dwing_angle_diff": FeatureSpec(
        func=compute_dwing_angle_diff,
        requires=["wing_angle_diff"],
        outputs=["dwing_angle_diff"],
        units={
            "dwing_angle_diff": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "dwing_angle_imbalance": FeatureSpec(
        func=compute_dwing_angle_imbalance,
        requires=["wing_angles"],
        outputs=["dwing_angle_imbalance"],
        units={
            "dwing_angle_imbalance": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1",},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "minmax_dwing_angle_in": FeatureSpec(
        func=compute_minmax_dwing_angle_in,
        requires=["wing_angles"],
        outputs=["max_dwing_angle_in", "min_dwing_angle_in"],
        units={
            "max_dwing_angle_in": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
            "min_dwing_angle_in": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "minmax_dwing_angle_out": FeatureSpec(
        func=compute_minmax_dwing_angle_out,
        requires=["wing_angles"],
        outputs=["max_dwing_angle_out", "min_dwing_angle_out"],
        units={
            "max_dwing_angle_out": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
            "min_dwing_angle_out": {"quantity": "angle_change_rate", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),

    # Front-leg kinematics
    "front_leg_angles": FeatureSpec(
        func=compute_front_leg_angles,
        requires=["theta"],
        outputs=["front_leg_L_ang", "front_leg_R_ang"],
        units={
            "front_leg_L_ang": {"quantity": "orientation", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
            "front_leg_R_ang": {"quantity": "orientation", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"ctr_ind": NODE_IDX_THORAX,
                "left_front_ind": NODE_IDX_L_FRONT_LEG,
                "right_front_ind": NODE_IDX_R_FRONT_LEG},
    ),
    "front_leg_angle_diff": FeatureSpec(
        func=compute_front_leg_angle_diff,
        requires=["front_leg_angles"],
        outputs=["front_leg_angle_diff"],
        units={
            "front_leg_angle_diff": {"quantity": "orientation", "unit_raw": "rad", "unit_si": "rad", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "dfront_leg_angles": FeatureSpec(
        func=compute_dfront_leg_angles,
        requires=["front_leg_angles"],
        outputs=["dfront_leg_L", "dfront_leg_R"],
        units={
            "dfront_leg_L": {"quantity": "angular_velocity", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
            "dfront_leg_R": {"quantity": "angular_velocity", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"fps": FPS},
    ),
    "front_leg_abs_angular_speed": FeatureSpec(
        func=compute_front_leg_abs_angular_speed,
        requires=["dfront_leg_angles"],
        outputs=["abs_dfront_leg_L", "abs_dfront_leg_R"],
        units={
            "abs_dfront_leg_L": {"quantity": "angular_speed", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
            "abs_dfront_leg_R": {"quantity": "angular_speed", "unit_raw": "rad/sec", "unit_si": "rad/sec", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
    ),
    "front_leg_extension": FeatureSpec(
        func=compute_front_leg_extension,
        requires=[],
        outputs=["front_leg_ext_L", "front_leg_ext_R"],
        units={
            "front_leg_ext_L": {"quantity": "distance", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
            "front_leg_ext_R": {"quantity": "distance", "unit_raw": "mm", "unit_si": "mm", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar",
        params={"ctr_ind": NODE_IDX_THORAX,
                "left_front_ind": NODE_IDX_L_FRONT_LEG,
                "right_front_ind": NODE_IDX_R_FRONT_LEG,
                "pxpermm": PXPERMM},
    ),
}

# Validate registry consistency at import time.
# This raises ValueError immediately if any dependency is missing or
# any output key is declared more than once.
validate_registry(REGISTRY)

def get_units_for_key(key: str, registry: dict[str, FeatureSpec]) -> dict[str, str] | None:
    """Return units metadata dict for a saved output key, or None if not found."""
    for spec in registry.values():
        if key in spec.units:
            return spec.units[key]
    return None


def compute_item(
    name: str,
    tracks: np.ndarray,
    registry: dict[str, FeatureSpec],
    computed: dict[str, object],
    **kwargs: object,
) -> None:
    _COMPUTED_SENTINEL = f"__computed_{name}__"
    if _COMPUTED_SENTINEL in computed:
        return

    spec = registry[name]

    if not spec.enabled:
        raise RuntimeError(
            f"Feature '{name}' is disabled (enabled=False) but was requested "
            f"either directly or as a dependency. "
            f"To resolve this, either re-enable '{name}' or also disable the "
            f"features that depend on it."
        )

    for dep in spec.requires:
        compute_item(dep, tracks, registry, computed, **kwargs)

    merged_kwargs = {**kwargs, **spec.params}
    value = spec.func(tracks, features=computed, **merged_kwargs)

    if spec.outputs:
        if not isinstance(value, dict):
            raise TypeError(
                f"Feature '{name}' declares outputs {spec.outputs} but its function "
                f"returned {type(value).__name__} instead of a dict. "
                f"Functions with non-empty outputs must return a dict."
            )
        missing = [k for k in spec.outputs if k not in value]
        if missing:
            raise KeyError(
                f"Feature '{name}' declared output key(s) {missing} but its function "
                f"did not return them. "
                f"Returned keys: {sorted(value.keys())}"
            )

    # Store only the individual output keys
    if isinstance(value, dict):
        for k, v in value.items():
            computed[k] = v
    else:
        # Bare-array return: only valid when outputs=[]
        computed[name] = value

    # Mark this item as computed so repeated calls are no-ops.
    computed[_COMPUTED_SENTINEL] = True