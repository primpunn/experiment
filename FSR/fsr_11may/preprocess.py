#!/usr/bin/env python3
"""
preprocess.py — Apply Hall 2008 pipeline to recorded session CSVs  (fsr_11may)

Input CSV columns: timestamp_ms, V1_mv, V2_mv, V3_mv, V4_mv, V5_mv, V6_mv [,event]
Output CSV adds:   F_FSR1_N … F_FSR6_N, F_calf_N, F_foot_N

Usage:
    conda activate massage
    cd /home/primpunn/experiment/FSR/fsr_11may

    # Process one file (output saved as <input>_forces.csv)
    python preprocess.py sessions/session_001.csv

    # Process all CSVs in the sessions directory
    python preprocess.py sessions/

    # Specify custom calibration file
    python preprocess.py sessions/session_001.csv --cal my_coeffs.json

    # Specify output path
    python preprocess.py sessions/session_001.csv -o processed/session_001_out.csv
"""

import argparse
import json
import sys
from pathlib import Path

from config import CALIBRATIONS_DIR
from fsr_pipeline import process_session_csv, find_latest_calibration_path


def process_file(csv_path, cal, output_path=None, verbose=True):
    out = process_session_csv(csv_path, cal["sensors"], output_path)
    if verbose:
        print(f"  {Path(csv_path).name}  →  {Path(out).name}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Post-process FSR session CSVs with Hall 2008 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input",
                    help="Path to CSV file or directory containing CSV files")
    ap.add_argument("-o", "--output", default=None,
                    help="Output CSV path (single-file mode only)")
    ap.add_argument("--cal", default=None,
                    help="Path to calibration_coeffs.json (default: latest in calibrations/)")
    args = ap.parse_args()

    if args.cal:
        cal_path = Path(args.cal)
    else:
        cal_path = find_latest_calibration_path()

    if cal_path is None or not cal_path.exists():
        sys.exit("No calibration found.\n"
                 "Run: python calibrate.py  to create one first.\n"
                 "Or specify: python preprocess.py <csv> --cal calibrations/<folder>/calibration_coeffs.json")

    try:
        cal = json.loads(cal_path.read_text())
    except Exception as e:
        sys.exit(f"Failed to load calibration: {e}")

    if "sensors" not in cal or not cal["sensors"]:
        sys.exit("Calibration file contains no sensor data.")

    calibrated = sorted(cal["sensors"].keys())
    print(f"  Calibration: {cal_path.parent.name}  ({len(calibrated)} sensors: {calibrated})")

    inp = Path(args.input)
    if inp.is_dir():
        csv_files = sorted(inp.glob("*.csv"))
        if not csv_files:
            sys.exit(f"No CSV files found in {inp}")
        print(f"  Processing {len(csv_files)} file(s) in {inp}/\n")
        for f in csv_files:
            if f.name.endswith("_forces.csv"):
                continue  # skip already-processed files
            try:
                process_file(f, cal)
            except Exception as e:
                print(f"  ERROR processing {f.name}: {e}")
    else:
        if not inp.exists():
            sys.exit(f"File not found: {inp}")
        try:
            out = process_file(inp, cal, args.output)
            print(f"\n  Done. Output: {out}")
        except Exception as e:
            sys.exit(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
