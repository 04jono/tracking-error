"""
diff_body_tracking.py

Compare unfixed (ctrax_results.mat) vs human-fixed (fixed_trx.mat) body tracking
on a per-frame basis.

Differences flagged:
  - wrong_count     : frame has != 10 fly detections in ctrax
  - position_diff   : after optimal (Hungarian) assignment, at least one fly is
                      further than DIST_THRESHOLD pixels from its matched fixed position
  - orientation_diff: after optimal assignment, at least one fly's heading angle
                      differs by more than ANGLE_THRESHOLD degrees
"""

import argparse
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment


DIST_THRESHOLD  = 10.0   # pixels
ANGLE_THRESHOLD = 20.0   # degrees


def angle_diff(a, b):
    """Absolute angular difference in radians, wrapped to [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def load_ctrax(path):
    """Return per-frame list of (identity, x, y, theta) tuples and ntargets array."""
    d = scipy.io.loadmat(path)
    x     = d["x_pos"].flatten()
    y     = d["y_pos"].flatten()
    theta = d["angle"].flatten()
    ident    = d["identity"].flatten().astype(int)
    ntargets = d["ntargets"].flatten().astype(int)

    frames = []
    idx = 0
    for nt in ntargets:
        frames.append(
            list(zip(ident[idx : idx + nt].tolist(),
                     x[idx : idx + nt].tolist(),
                     y[idx : idx + nt].tolist(),
                     theta[idx : idx + nt].tolist()))
        )
        idx += nt
    return frames, ntargets


def load_fixed(path):
    """Return fixed position/orientation arrays of shape (n_frames, n_flies)."""
    d = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    trx = d["trx"]
    n_flies = len(trx)
    fixed_x     = np.stack([trx[i].x     for i in range(n_flies)], axis=1)
    fixed_y     = np.stack([trx[i].y     for i in range(n_flies)], axis=1)
    fixed_theta = np.stack([trx[i].theta for i in range(n_flies)], axis=1)
    return fixed_x, fixed_y, fixed_theta


def diff_tracking(ctrax_path, fixed_path,
                  dist_threshold=DIST_THRESHOLD,
                  angle_threshold_deg=ANGLE_THRESHOLD):
    frames_ctrax, ntargets = load_ctrax(ctrax_path)
    fixed_x, fixed_y, fixed_theta = load_fixed(fixed_path)

    angle_threshold_rad = np.radians(angle_threshold_deg)
    n_frames = len(ntargets)
    n_flies  = fixed_x.shape[1]

    diffs = []

    for f in range(n_frames):
        frame_1idx = f + 1
        nt = ntargets[f]

        if nt != n_flies:
            diffs.append({
                "frame":         frame_1idx,
                "type":          "wrong_count",
                "ctrax_count":   nt,
                "fixed_count":   n_flies,
                "max_dist":      None,
                "max_angle_deg": None,
                "worst_fly_idx": None,
                "x":             None,
                "y":             None,
            })
            continue

        cx     = np.array([e[1] for e in frames_ctrax[f]])
        cy     = np.array([e[2] for e in frames_ctrax[f]])
        ctheta = np.array([e[3] for e in frames_ctrax[f]])

        cost = np.sqrt(
            (cx[:, None] - fixed_x[f][None, :]) ** 2
            + (cy[:, None] - fixed_y[f][None, :]) ** 2
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_dists = cost[row_ind, col_ind]

        # Orientation differences using the same matching
        matched_angle_diffs = angle_diff(ctheta[row_ind], fixed_theta[f][col_ind])

        max_dist      = matched_dists.max()
        max_angle_rad = matched_angle_diffs.max()

        if max_dist > dist_threshold:
            worst_col = col_ind[matched_dists.argmax()]
            diffs.append({
                "frame":         frame_1idx,
                "type":          "position_diff",
                "ctrax_count":   nt,
                "fixed_count":   n_flies,
                "max_dist":      float(max_dist),
                "max_angle_deg": float(np.degrees(max_angle_rad)),
                "worst_fly_idx": int(worst_col),
                "x":             float(fixed_x[f, worst_col]),
                "y":             float(fixed_y[f, worst_col]),
            })
        elif max_angle_rad > angle_threshold_rad:
            worst_col = col_ind[matched_angle_diffs.argmax()]
            diffs.append({
                "frame":         frame_1idx,
                "type":          "orientation_diff",
                "ctrax_count":   nt,
                "fixed_count":   n_flies,
                "max_dist":      float(max_dist),
                "max_angle_deg": float(np.degrees(max_angle_rad)),
                "worst_fly_idx": int(worst_col),
                "x":             float(fixed_x[f, worst_col]),
                "y":             float(fixed_y[f, worst_col]),
            })

    return diffs


def print_report(diffs, dist_threshold, angle_threshold_deg):
    wrong_count   = [d for d in diffs if d["type"] == "wrong_count"]
    pos_diff      = [d for d in diffs if d["type"] == "position_diff"]
    orient_diff   = [d for d in diffs if d["type"] == "orientation_diff"]

    print(f"Thresholds: {dist_threshold:.1f} px, {angle_threshold_deg:.1f} deg")
    print(f"Total differing frames  : {len(diffs)}")
    print(f"  wrong_count           : {len(wrong_count)}")
    print(f"  position_diff         : {len(pos_diff)}")
    print(f"  orientation_diff      : {len(orient_diff)}")
    print()

    if wrong_count:
        print("=== wrong_count frames ===")
        for d in wrong_count:
            print(f"  frame {d['frame']:6d}  ctrax={d['ctrax_count']}  fixed={d['fixed_count']}")
        print()

    if pos_diff:
        print("=== position_diff frames (sorted by max_dist desc) ===")
        for d in sorted(pos_diff, key=lambda x: -x["max_dist"]):
            print(f"  frame {d['frame']:6d}  max_dist={d['max_dist']:7.2f}px  angle={d['max_angle_deg']:6.1f}deg  worst_fly={d['worst_fly_idx']}")
        print()

    if orient_diff:
        print("=== orientation_diff frames (sorted by max_angle desc) ===")
        for d in sorted(orient_diff, key=lambda x: -x["max_angle_deg"]):
            print(f"  frame {d['frame']:6d}  angle={d['max_angle_deg']:6.1f}deg  max_dist={d['max_dist']:7.2f}px  worst_fly={d['worst_fly_idx']}")


def write_output(diffs, path):
    """Write a TSV with columns: frame, type, x, y (NaN for wrong_count frames)."""
    with open(path, "w") as f:
        f.write("frame\ttype\tx\ty\n")
        for d in sorted(diffs, key=lambda x: x["frame"]):
            x = d["x"] if d["x"] is not None else "nan"
            y = d["y"] if d["y"] is not None else "nan"
            f.write(f"{d['frame']}\t{d['type']}\t{x}\t{y}\n")
    print(f"Output written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Diff ctrax vs fixed body tracking")
    parser.add_argument("ctrax", help="Path to ctrax_results.mat (unfixed)")
    parser.add_argument("fixed", help="Path to fixed_trx.mat (human-fixed)")
    parser.add_argument(
        "--threshold", type=float, default=DIST_THRESHOLD,
        help=f"Pixel distance threshold for position_diff (default: {DIST_THRESHOLD})"
    )
    parser.add_argument(
        "--angle-threshold", type=float, default=ANGLE_THRESHOLD,
        help=f"Angle threshold in degrees for orientation_diff (default: {ANGLE_THRESHOLD})"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write error frames as TSV (frame, type, x, y) to this file"
    )
    args = parser.parse_args()

    diffs = diff_tracking(args.ctrax, args.fixed, args.threshold, args.angle_threshold)
    print_report(diffs, args.threshold, args.angle_threshold)

    if args.output:
        write_output(diffs, args.output)


if __name__ == "__main__":
    main()
