# Stage 1: Split multi-arena recordings

import sys
from tkfilebrowser import askopendirnames
import tkinter as tk
from pathlib import Path
import shutil
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from params import BASE_PATH

def split_experiment_to_arenas(experiment_dir):
    """
    Split a single experiment folder into per arena folders.

    For each arena movie (e.g. R1C1), create a new folder:
    <experiment_name>_<arena_id>

    The folder will contain:
    - the full movie (original name)
    - the arena movie renamed to 'movie.<ext>'
    - all auxiliary experiment files
    """
    experiment_dir = Path(experiment_dir)
    exp_name = experiment_dir.name

    print(f"\nProcessing experiment folder: {exp_name}")

    # Collect auxiliary files
    aux_files = list(experiment_dir.glob(f"{exp_name}*"))
    aux_files = [
        f for f in aux_files
        if f.is_file() and f.suffix.lower() not in [".mp4", ".avi", ".mov"]
    ]

    details_file = experiment_dir / "experiment_details.JSON"
    if details_file.exists():
        aux_files.append(details_file)
        print("Found experiment_details.JSON")

    print(f"Found {len(aux_files)} auxiliary files")

    # Identify full movie
    full_movies = [
        f for f in experiment_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in [".mp4", ".avi", ".mov"]
        and not re.search(r"R\d+C\d+", f.stem)
    ]

    if len(full_movies) != 1:
        raise ValueError(
            f"Expected exactly one full movie, found {len(full_movies)}"
        )

    full_movie = full_movies[0]
    print(f"Full movie identified: {full_movie.name}")

    # Identify arena movies
    arena_movies = [
        f for f in experiment_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in [".mp4", ".avi", ".mov"]
        and re.search(r"R\d+C\d+", f.stem)
    ]

    print(f"Found {len(arena_movies)} arena movies")

    for arena_movie in arena_movies:
        arena_id = re.search(r"R\d+C\d+", arena_movie.stem).group()
        arena_dir = experiment_dir.parent / f"{exp_name}_{arena_id}"

        print(f"\nCreating arena folder: {arena_dir.name}")
        arena_dir.mkdir(exist_ok=True)

        shutil.copy2(full_movie, arena_dir / full_movie.name)
        print(f"  Copied full movie")

        shutil.copy2(
            arena_movie,
            arena_dir / f"movie{arena_movie.suffix}"
        )
        print(f"  Copied arena movie as movie{arena_movie.suffix}")

        for aux in aux_files:
            shutil.copy2(aux, arena_dir / aux.name)
            print(f"  Copied aux file: {aux.name}")

    print(f"Finished processing experiment: {exp_name}")

def main():
    root = tk.Tk()
    root.withdraw()  # hide empty Tk window

    selected_dirs = askopendirnames(
        title="Select one or more experiment group folders",
        initialdir=BASE_PATH
    )

    if not selected_dirs:
        print("No folders selected")
        return

    for folder in selected_dirs:
        try:
            split_experiment_to_arenas(folder)
        except Exception as e:
            print(f"ERROR processing {Path(folder).name}: {e}")

    print("\nAll processing completed")


if __name__ == "__main__":
    main()