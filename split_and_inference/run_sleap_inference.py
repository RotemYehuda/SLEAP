# Stage 2: Run SLEAP, convert .slp → .analysis.h5

import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from params import ARENA_PARENT_DIR as _ARENA_PARENT_DIR
# from params import CENTROID_MODEL   as _CENTROID_MODEL
# from params import INSTANCE_MODEL   as _INSTANCE_MODEL
from params import BOTTOMUP_MODEL   as _BOTTOMUP_MODEL
from params import SLEAP_OUTPUT_NAME

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov", ".mkv"]

ARENA_PARENT_DIR = Path(_ARENA_PARENT_DIR) if _ARENA_PARENT_DIR else Path(".")
BOTTOMUP_MODEL   = Path(_BOTTOMUP_MODEL)   if _BOTTOMUP_MODEL   else None

# CENTROID_MODEL   = Path(_CENTROID_MODEL)   if _CENTROID_MODEL   else None
# INSTANCE_MODEL   = Path(_INSTANCE_MODEL)   if _INSTANCE_MODEL   else None

OUTPUT_NAME = SLEAP_OUTPUT_NAME

def find_movie(arena_dir):
    for ext in VIDEO_EXTENSIONS:
        candidate = arena_dir / f"movie{ext}"
        if candidate.exists():
            return candidate
    return None


def run_sleap(movie_path, output_path):
    cmd = [
        "sleap-track",
        str(movie_path),
        "--model", str(BOTTOMUP_MODEL),
        "--output", str(output_path),
        "--tracking.match", "hungarian",
        "--tracking.max_tracking", "1",
        "--tracking.max_tracks", "2",
        "--tracking.target_instance_count", "2",
        "--tracking.post_connect_single_breaks", "1",
        "--tracking.similarity", "instance",
        "--tracking.track_window", "5",
        "--tracking.tracker", "flowmaxtracks",
    ]
    subprocess.run(cmd, check=True)


def convert_to_h5(slp_path):
    cmd = [
        "sleap-convert",
        str(slp_path),
        "--format", "analysis"
    ]

    print("Converting to H5:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def parse_selection(selection, max_index):
    selected = set()

    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))

    return [i for i in selected if 1 <= i <= max_index]


def main():
    # if CENTROID_MODEL is None or not CENTROID_MODEL.exists():
    #     print(f"ERROR: CENTROID_MODEL is not configured or does not exist: {CENTROID_MODEL}")
    #     print("Please set 'centroid_model' in config.yaml (or CENTROID_MODEL in params.py).")
    #     return
    # if INSTANCE_MODEL is None or not INSTANCE_MODEL.exists():
    #     print(f"ERROR: INSTANCE_MODEL is not configured or does not exist: {INSTANCE_MODEL}")
    #     print("Please set 'instance_model' in config.yaml (or INSTANCE_MODEL in params.py).")
    #     return

    if BOTTOMUP_MODEL is None or not BOTTOMUP_MODEL.exists():
        print(f"ERROR: BOTTOMUP_MODEL is not configured or does not exist: {BOTTOMUP_MODEL}")
        print("Please set 'bottomup_model' in config.yaml (or BOTTOMUP_MODEL in params.py).")
        return
    arena_dirs = sorted(
        [d for d in ARENA_PARENT_DIR.iterdir() if d.is_dir()]
    )

    if not arena_dirs:
        print("No arena folders found")
        return

    print("\nAvailable arena folders:\n")
    for i, d in enumerate(arena_dirs, start=1):
        print(f"{i:2d}. {d.name}")

    selection = input(
        "\nSelect arenas to run (e.g. 1,3,5-7 or ENTER for all): "
    ).strip()

    if selection:
        indices = parse_selection(selection, len(arena_dirs))
        selected_dirs = [arena_dirs[i - 1] for i in indices]
    else:
        selected_dirs = arena_dirs

    print(f"\nSelected {len(selected_dirs)} arenas\n")

    for arena_dir in selected_dirs:
        print(f"Processing {arena_dir.name}")

        movie = find_movie(arena_dir)
        if movie is None:
            print("  No movie found, skipping")
            continue

        output = arena_dir / OUTPUT_NAME
        if output.exists():
            print("  Inference already exists, skipping")
            continue

        try:
            run_sleap(movie, output)
            print("  Inference done")

            convert_to_h5(output)
            print("  H5 conversion done")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {e}")

    print("\nInference completed")


if __name__ == "__main__":
    main()