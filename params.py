# Config loader; exposes constants globally on import

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

_config = _load_config()

# GUI default directory for file pickers.
# Falls back to empty string, which opens the picker in the current directory.
BASE_PATH = _config.get("base_path", "")

# SLEAP inference parent directory (used by run_sleap_inference.py only).
ARENA_PARENT_DIR = _config.get("arena_parent_dir", "")
# CENTROID_MODEL    = _config.get("centroid_model", "")
# INSTANCE_MODEL    = _config.get("instance_model", "")
BOTTOMUP_MODEL    = _config.get("bottomup_model", "")
SLEAP_OUTPUT_NAME = _config.get("sleap_output_name", "inference.slp")

# Calibration constants — used as defaults when not stored in the HDF5 file.
FPS: float = float(_config.get("fps", 30))
PXPERMM: float = float(_config.get("pxpermm", 10.5))

# Expected node names in skeleton order
NODE_NAMES_EXPECTED: list | None = _config.get("node_names_expected", None)

# Node indices — derived from NODE_NAMES_EXPECTED.
# Fallback integers match the default skeleton order if config.yaml is absent.
_nne = NODE_NAMES_EXPECTED or []

def _node_idx(name: str, fallback: int) -> int:
    try:
        return _nne.index(name)
    except ValueError:
        return fallback

NODE_IDX_HEAD         = _node_idx("head",    0)
NODE_IDX_THORAX       = _node_idx("thorax",  1)
NODE_IDX_ABDOMEN      = _node_idx("abdomen", 2)
NODE_IDX_L_WING       = _node_idx("L_wing",     10)
NODE_IDX_R_WING       = _node_idx("R_wing",      9)
NODE_IDX_L_FRONT_LEG  = _node_idx("L_frontLeg",  4)
NODE_IDX_R_FRONT_LEG  = _node_idx("R_frontLeg",  3)
NODE_IDX_L_MIDDLE_LEG = _node_idx("L_midLeg",    6)
NODE_IDX_R_MIDDLE_LEG = _node_idx("R_midLeg",    5)

MAX_GAP_BY_NODE = {
    "default"   : 15,  # frames left un-filled if the interior gap is longer than this
    "R_midLeg"  : 5,
    "L_midLeg"  : 5,
    "R_hindLeg" : 2,
    "L_hindLeg" : 2,
    "thorax"    : 10,
    "head"      : 10,
}