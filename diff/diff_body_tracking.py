#!/usr/bin/env python3
"""
diff_body_tracking.py

Compare raw FlyTracker output (flytracker_trx.mat) vs human-fixed FlyTracker output
(fixed_trx.mat) on a per-fly, per-frame basis.

This script assumes both files come from the same FlyTracker pipeline, so fly indices
already correspond. It therefore compares fly i in flytracker_trx.mat directly against
fly i in fixed_trx.mat, without Hungarian matching.

Classified error types:
  - missing_in_flytracker  : fixed has a fly at this frame, but flytracker is NaN
  - missing_in_fixed       : flytracker has a fly at this frame, but fixed is NaN
  - position_diff          : position differs by more than DIST_THRESHOLD pixels
  - orientation_diff       : heading differs by more than ANGLE_THRESHOLD degrees
  - possible_id_switch     : position differs by more than ID_SWITCH_DIST_THRESHOLD pixels
  - possible_jitter        : raw flytracker motion jumps sharply relative to previous frame
                             but fixed tracking stays smooth
"""

import argparse
import os
import sys
import numpy as np
import h5py
import scipy.io
from pathlib import Path


def _load_inputs_env(filename="inputs.env"):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = Path.cwd()
    for _ in range(4):
        candidate = p / filename
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return
        p = p.parent


DIST_THRESHOLD = 10.0            # pixels
ANGLE_THRESHOLD = 20.0           # degrees
ID_SWITCH_DIST_THRESHOLD = 40.0  # pixels
JITTER_JUMP_THRESHOLD = 25.0     # pixels


def angle_diff(a, b):
    """Absolute angular difference in radians, wrapped to [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def _flatten_numeric(x):
    """Convert MATLAB-loaded numeric data into a 1D numpy array."""
    return np.asarray(x).reshape(-1)


def _read_h5_ref_array(f, ds):
    """
    Read a MATLAB v7.3 field stored as an HDF5 dataset of object references.

    Example:
        trx['x'] is shape (n_flies, 1), where each entry is a reference
        to a numeric dataset containing that fly's trajectory.
    """
    refs = np.asarray(ds).reshape(-1)
    out = []
    for ref in refs:
        arr = np.asarray(f[ref]).reshape(-1)
        out.append(arr)
    return out


def _read_h5_scalar_ref_array(f, ds):
    """
    Read a MATLAB v7.3 field stored as references to scalar datasets.
    Returns a list of Python scalars.
    """
    refs = np.asarray(ds).reshape(-1)
    out = []
    for ref in refs:
        val = np.asarray(f[ref]).reshape(-1)
        if val.size != 1:
            raise ValueError(f"Expected scalar dataset, got shape {val.shape}")
        out.append(val[0].item())
    return out


def load_trx_h5(path):
    """
    Load trx-style tracking from MATLAB v7.3 HDF5 trx format.
    Supports both flytracker_trx.mat and fixed_trx.mat.

    Returns:
        X, Y, TH arrays of shape (n_frames, n_flies)
    """
    with h5py.File(path, "r") as f:
        if "trx" not in f:
            raise ValueError(f"No 'trx' group found in {path}. Keys: {list(f.keys())}")

        trx = f["trx"]
        if not isinstance(trx, h5py.Group):
            raise ValueError(f"Expected 'trx' to be an HDF5 group, got {type(trx)}")

        required_fields = ["x", "y", "theta", "firstframe", "endframe"]
        missing = [k for k in required_fields if k not in trx]
        if missing:
            raise ValueError(f"Missing required trx fields in {path}: {missing}")

        xs = _read_h5_ref_array(f, trx["x"])
        ys = _read_h5_ref_array(f, trx["y"])
        thetas = _read_h5_ref_array(f, trx["theta"])
        firstframes = [int(v) for v in _read_h5_scalar_ref_array(f, trx["firstframe"])]
        endframes = [int(v) for v in _read_h5_scalar_ref_array(f, trx["endframe"])]

        n_flies = len(xs)
        n_frames = max(endframes)

        X = np.full((n_frames, n_flies), np.nan)
        Y = np.full((n_frames, n_flies), np.nan)
        TH = np.full((n_frames, n_flies), np.nan)

        for i in range(n_flies):
            f0 = firstframes[i] - 1
            f1 = endframes[i]

            x = xs[i]
            y = ys[i]
            theta = thetas[i]

            expected_len = f1 - f0
            if not (len(x) == len(y) == len(theta) == expected_len):
                raise ValueError(
                    f"{path}, fly {i}: inconsistent trajectory lengths. "
                    f"Expected {expected_len}, got x={len(x)}, y={len(y)}, theta={len(theta)}"
                )

            X[f0:f1, i] = x
            Y[f0:f1, i] = y
            TH[f0:f1, i] = theta

        return X, Y, TH


def load_trx_legacy(path):
    """
    Load trx-style tracking from legacy non-v7.3 MATLAB files.
    Supports fixed_trx.mat-style struct arrays.
    """
    d = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)

    if "trx" not in d:
        raise ValueError(f"Unrecognized legacy MATLAB trx format in {path}")

    trx = d["trx"]
    if np.ndim(trx) == 0:
        trx = [trx]

    n_flies = len(trx)
    n_frames = max(int(fly.endframe) for fly in trx)

    X = np.full((n_frames, n_flies), np.nan)
    Y = np.full((n_frames, n_flies), np.nan)
    TH = np.full((n_frames, n_flies), np.nan)

    for i, fly in enumerate(trx):
        f0 = int(fly.firstframe) - 1
        f1 = int(fly.endframe)
        X[f0:f1, i] = _flatten_numeric(fly.x)
        Y[f0:f1, i] = _flatten_numeric(fly.y)
        TH[f0:f1, i] = _flatten_numeric(fly.theta)

    return X, Y, TH


def load_trx(path):
    """
    Unified trx loader.

    Supports:
      - MATLAB v7.3 HDF5 trx files (flytracker_trx.mat, fixed_trx.mat)
      - legacy scipy-readable trx files
    """
    try:
        return load_trx_legacy(path)
    except NotImplementedError:
        return load_trx_h5(path)


def classify_frame_fly(
    frame_idx,
    fly_idx,
    ux,
    uy,
    uth,
    fx,
    fy,
    fth,
    dist_threshold,
    angle_threshold_rad,
    id_switch_dist_threshold,
    jitter_jump_threshold,
):
    """
    Classify the error for a single fly on a single frame.
    Returns a dict or None if no error.
    """
    u_valid = not (np.isnan(ux) or np.isnan(uy) or np.isnan(uth))
    f_valid = not (np.isnan(fx) or np.isnan(fy) or np.isnan(fth))

    if f_valid and not u_valid:
        return {
            "frame": frame_idx + 1,
            "fly": fly_idx,
            "type": "missing_in_flytracker",
            "dist": None,
            "angle_deg": None,
            "x": float(fx),
            "y": float(fy),
        }

    if u_valid and not f_valid:
        return {
            "frame": frame_idx + 1,
            "fly": fly_idx,
            "type": "missing_in_fixed",
            "dist": None,
            "angle_deg": None,
            "x": float(ux),
            "y": float(uy),
        }

    if not u_valid and not f_valid:
        return None

    dist = float(np.hypot(ux - fx, uy - fy))
    ang = float(angle_diff(uth, fth))
    angle_deg = float(np.degrees(ang))

    # Strongest condition first
    if dist > id_switch_dist_threshold:
        return {
            "frame": frame_idx + 1,
            "fly": fly_idx,
            "type": "possible_id_switch",
            "dist": dist,
            "angle_deg": angle_deg,
            "x": float(fx),
            "y": float(fy),
        }

    if dist > dist_threshold:
        return {
            "frame": frame_idx + 1,
            "fly": fly_idx,
            "type": "position_diff",
            "dist": dist,
            "angle_deg": angle_deg,
            "x": float(fx),
            "y": float(fy),
        }

    if ang > angle_threshold_rad:
        return {
            "frame": frame_idx + 1,
            "fly": fly_idx,
            "type": "orientation_diff",
            "dist": dist,
            "angle_deg": angle_deg,
            "x": float(fx),
            "y": float(fy),
        }

    return None


def detect_jitter(
    frame_idx,
    fly_idx,
    ux,
    uy,
    fx,
    fy,
    dist,
    angle_deg,
    jitter_jump_threshold,
):
    """
    Detect likely jitter by comparing motion jump from previous frame.
    Returns a dict or None.
    """
    if frame_idx == 0:
        return None

    prev_u_valid = not (np.isnan(ux[frame_idx - 1, fly_idx]) or np.isnan(uy[frame_idx - 1, fly_idx]))
    prev_f_valid = not (np.isnan(fx[frame_idx - 1, fly_idx]) or np.isnan(fy[frame_idx - 1, fly_idx]))
    curr_u_valid = not (np.isnan(ux[frame_idx, fly_idx]) or np.isnan(uy[frame_idx, fly_idx]))
    curr_f_valid = not (np.isnan(fx[frame_idx, fly_idx]) or np.isnan(fy[frame_idx, fly_idx]))

    if not (prev_u_valid and prev_f_valid and curr_u_valid and curr_f_valid):
        return None

    raw_jump = float(
        np.hypot(
            ux[frame_idx, fly_idx] - ux[frame_idx - 1, fly_idx],
            uy[frame_idx, fly_idx] - uy[frame_idx - 1, fly_idx],
        )
    )
    fixed_jump = float(
        np.hypot(
            fx[frame_idx, fly_idx] - fx[frame_idx - 1, fly_idx],
            fy[frame_idx, fly_idx] - fy[frame_idx - 1, fly_idx],
        )
    )

    if raw_jump > jitter_jump_threshold and fixed_jump < jitter_jump_threshold / 2.0:
        return {
            "frame": frame_idx + 1,
            "fly": fly_idx,
            "type": "possible_jitter",
            "dist": dist,
            "angle_deg": angle_deg,
            "x": float(fx[frame_idx, fly_idx]),
            "y": float(fy[frame_idx, fly_idx]),
            "raw_jump": raw_jump,
            "fixed_jump": fixed_jump,
        }

    return None


def diff_tracking(
    flytracker_path,
    fixed_path,
    dist_threshold=DIST_THRESHOLD,
    angle_threshold_deg=ANGLE_THRESHOLD,
    id_switch_dist_threshold=ID_SWITCH_DIST_THRESHOLD,
    jitter_jump_threshold=JITTER_JUMP_THRESHOLD,
):
    """
    Compare flytracker_trx.mat vs fixed_trx.mat directly by fly index.
    """
    ux, uy, uth = load_trx(flytracker_path)
    fx, fy, fth = load_trx(fixed_path)

    if ux.shape != fx.shape:
        raise ValueError(
            f"Shape mismatch between flytracker and fixed arrays: "
            f"{ux.shape} vs {fx.shape}"
        )

    angle_threshold_rad = np.radians(angle_threshold_deg)
    n_frames, n_flies = fx.shape

    diffs = []

    for f in range(n_frames):
        for i in range(n_flies):
            base = classify_frame_fly(
                f,
                i,
                ux[f, i],
                uy[f, i],
                uth[f, i],
                fx[f, i],
                fy[f, i],
                fth[f, i],
                dist_threshold,
                angle_threshold_rad,
                id_switch_dist_threshold,
                jitter_jump_threshold,
            )

            if base is not None:
                diffs.append(base)
                continue

            # Optional second-pass jitter classification
            if not (
                np.isnan(ux[f, i]) or np.isnan(uy[f, i]) or np.isnan(uth[f, i]) or
                np.isnan(fx[f, i]) or np.isnan(fy[f, i]) or np.isnan(fth[f, i])
            ):
                dist = float(np.hypot(ux[f, i] - fx[f, i], uy[f, i] - fy[f, i]))
                angle_deg = float(np.degrees(angle_diff(uth[f, i], fth[f, i])))

                jitter = detect_jitter(
                    f,
                    i,
                    ux,
                    uy,
                    fx,
                    fy,
                    dist,
                    angle_deg,
                    jitter_jump_threshold,
                )
                if jitter is not None:
                    diffs.append(jitter)

    return diffs


def print_report(diffs, dist_threshold, angle_threshold_deg, id_switch_dist_threshold, jitter_jump_threshold):
    categories = [
        "missing_in_flytracker",
        "missing_in_fixed",
        "possible_id_switch",
        "position_diff",
        "orientation_diff",
        "possible_jitter",
    ]

    grouped = {k: [d for d in diffs if d["type"] == k] for k in categories}

    print(
        f"Thresholds: dist={dist_threshold:.1f}px, "
        f"angle={angle_threshold_deg:.1f}deg, "
        f"id_switch={id_switch_dist_threshold:.1f}px, "
        f"jitter_jump={jitter_jump_threshold:.1f}px"
    )
    print(f"Total differing fly-frames: {len(diffs)}")
    for k in categories:
        print(f"  {k:22s}: {len(grouped[k])}")
    print()

    if grouped["missing_in_flytracker"]:
        print("=== missing_in_flytracker ===")
        for d in grouped["missing_in_flytracker"][:200]:
            print(f"  frame {d['frame']:6d}  fly={d['fly']:2d}  x={d['x']:.1f} y={d['y']:.1f}")
        print()

    if grouped["missing_in_fixed"]:
        print("=== missing_in_fixed ===")
        for d in grouped["missing_in_fixed"][:200]:
            print(f"  frame {d['frame']:6d}  fly={d['fly']:2d}  x={d['x']:.1f} y={d['y']:.1f}")
        print()

    if grouped["possible_id_switch"]:
        print("=== possible_id_switch (sorted by dist desc) ===")
        for d in sorted(grouped["possible_id_switch"], key=lambda x: -x["dist"])[:200]:
            print(
                f"  frame {d['frame']:6d}  fly={d['fly']:2d}  "
                f"dist={d['dist']:7.2f}px  angle={d['angle_deg']:6.1f}deg"
            )
        print()

    if grouped["position_diff"]:
        print("=== position_diff (sorted by dist desc) ===")
        for d in sorted(grouped["position_diff"], key=lambda x: -x["dist"])[:200]:
            print(
                f"  frame {d['frame']:6d}  fly={d['fly']:2d}  "
                f"dist={d['dist']:7.2f}px  angle={d['angle_deg']:6.1f}deg"
            )
        print()

    if grouped["orientation_diff"]:
        print("=== orientation_diff (sorted by angle desc) ===")
        for d in sorted(grouped["orientation_diff"], key=lambda x: -x["angle_deg"])[:200]:
            print(
                f"  frame {d['frame']:6d}  fly={d['fly']:2d}  "
                f"angle={d['angle_deg']:6.1f}deg  dist={d['dist']:7.2f}px"
            )
        print()

    if grouped["possible_jitter"]:
        print("=== possible_jitter ===")
        for d in grouped["possible_jitter"][:200]:
            print(
                f"  frame {d['frame']:6d}  fly={d['fly']:2d}  "
                f"dist={d['dist']:7.2f}px  angle={d['angle_deg']:6.1f}deg  "
                f"raw_jump={d['raw_jump']:7.2f}px  fixed_jump={d['fixed_jump']:7.2f}px"
            )
        print()


def write_output(diffs, path):
    """
    Write a TSV with columns:
      frame, fly, type, x, y, dist, angle_deg
    """
    with open(path, "w") as f:
        f.write("frame\tfly\ttype\tx\ty\tdist\tangle_deg\n")
        for d in sorted(diffs, key=lambda x: (x["frame"], x["fly"], x["type"])):
            x = d["x"] if d.get("x") is not None else "nan"
            y = d["y"] if d.get("y") is not None else "nan"
            dist = d["dist"] if d.get("dist") is not None else "nan"
            angle_deg = d["angle_deg"] if d.get("angle_deg") is not None else "nan"
            f.write(f"{d['frame']}\t{d['fly']}\t{d['type']}\t{x}\t{y}\t{dist}\t{angle_deg}\n")
    print(f"Output written to {path}")


def main():
    _load_inputs_env()

    parser = argparse.ArgumentParser(
        description="Diff flytracker_trx.mat vs fixed_trx.mat with classified error types"
    )
    parser.add_argument("flytracker", nargs="?", default=os.environ.get("FLYTRACKER"),
                        help="Path to flytracker_trx.mat. Default: $FLYTRACKER from inputs.env")
    parser.add_argument("fixed", nargs="?", default=os.environ.get("FIXED"),
                        help="Path to fixed_trx.mat. Default: $FIXED from inputs.env")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DIST_THRESHOLD,
        help=f"Pixel distance threshold for position_diff (default: {DIST_THRESHOLD})",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=ANGLE_THRESHOLD,
        help=f"Angle threshold in degrees for orientation_diff (default: {ANGLE_THRESHOLD})",
    )
    parser.add_argument(
        "--id-switch-threshold",
        type=float,
        default=ID_SWITCH_DIST_THRESHOLD,
        help=f"Pixel distance threshold for possible_id_switch (default: {ID_SWITCH_DIST_THRESHOLD})",
    )
    parser.add_argument(
        "--jitter-threshold",
        type=float,
        default=JITTER_JUMP_THRESHOLD,
        help=f"Jump threshold in pixels for possible_jitter (default: {JITTER_JUMP_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write error fly-frames as TSV to this file",
    )
    args = parser.parse_args()

    if not args.flytracker or not args.fixed:
        parser.error("flytracker and fixed are required (or set FLYTRACKER and FIXED in inputs.env)")

    diffs = diff_tracking(
        args.flytracker,
        args.fixed,
        dist_threshold=args.threshold,
        angle_threshold_deg=args.angle_threshold,
        id_switch_dist_threshold=args.id_switch_threshold,
        jitter_jump_threshold=args.jitter_threshold,
    )

    print_report(
        diffs,
        args.threshold,
        args.angle_threshold,
        args.id_switch_threshold,
        args.jitter_threshold,
    )

    if args.output:
        write_output(diffs, args.output)


if __name__ == "__main__":
    main()