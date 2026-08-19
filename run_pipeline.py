# ENTRY POINT (Stage 3) — GUI, drives full pipeline

import tkinter as tk
from pathlib import Path
from tkfilebrowser import askopendirnames

import traceback

from dataset import make_expt_dataset
from preprocessing import process_file
from perframe.create_perframe import export_perframe
from perframe.trx_mat import save_trx
from params import BASE_PATH, FPS, PXPERMM, MAX_GAP_BY_NODE

def main(fill_missing: bool = False):
    """
    Entry point for batch extraction of features from SLEAP analysis files.

    Opens a GUI to select experiment folders, locates a single '*.analysis.h5'
    file in each folder, and generates a corresponding '*.features.h5' file
    using `make_expt_dataset`.

    Handles only user interaction, input validation, and per-experiment
    orchestration. Assumes fixed, proofread identities in the input files.

    Creates one output file per experiment and reports progress to stdout.

    Args:
        fill_missing: If True, runs the gap-filling step on each
            selected experiment's .analysis.h5 before feature extraction,
            and feature computation uses the filled tracks matrix
            (tracks_filled) instead of the raw tracks.

    Returns:
        None
    """
    root = tk.Tk()
    root.withdraw()

    expt_folders = askopendirnames(
        title="Select one or more experiment folders",
        initialdir=BASE_PATH
    )

    if not expt_folders:
        print("No experiment folders selected")
        return

    total_folders = len(expt_folders)
    for i, expt_folder in enumerate(expt_folders, start=1):
        expt_folder = expt_folder.strip()
        print(f"\n[{i}/{total_folders}] Processing experiment folder: {expt_folder}")

        # Decide which inference file exists
        analysis_files = list(Path(expt_folder).glob("*.analysis.h5"))

        if len(analysis_files) == 0:
            print("\tNo inference H5 found, skipping")
            continue
        elif len(analysis_files) > 1:
            print("\tMultiple analysis H5 files found, skipping")
            continue

        analysis_path = str(analysis_files[0])

        if fill_missing:
            print("\tFilling missing values in tracking data...")
            try:
                process_file(Path(analysis_path), MAX_GAP_BY_NODE)
            except Exception as e:
                print(f"\tERROR during fill_missing: {e}")
                traceback.print_exc()

        features_path = analysis_path.replace(".analysis.h5", ".features.h5")

        # Stage 1: feature extraction
        features_h5 = None
        try:
            features_h5 = make_expt_dataset(
                expt_folder,
                h5_file=analysis_path,
                output_path=features_path,
                overwrite=True,
                fps=FPS,
                pxpermm=PXPERMM,
                use_filled_tracks=fill_missing,
            )
        # except Exception as e:
        #     print(f"\tERROR during feature extraction: {e}")
        except Exception as e:
            print(f"\tERROR during feature extraction: {e}")
            traceback.print_exc()

        if features_h5 is None or not Path(features_h5).is_file():
            print("\tSkipping export: no valid features file produced.")
            continue

        # Stage 2: perframe and trx export
        features_h5 = Path(features_h5)
        perframe_dir = Path(expt_folder) / "perframe"

        try:
            print(f"\tUsing features file: {features_h5.name}")
            print(f"\tCreating perframe directory: {perframe_dir}")

            export_perframe(
                features_h5=features_h5,
                perframe_dir=perframe_dir,
                overwrite=True,
            )

            trx_path = Path(expt_folder) / "trx.mat"
            save_trx(features_h5, trx_path, timestamps=None, overwrite=True)

        except Exception as e:
            print(f"\tERROR during export: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main(fill_missing=True)