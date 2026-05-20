#!/usr/bin/env python3
"""
Convert video files to AVI and optionally overlay APT tracking data.

Can be used as:
1. Video converter: Convert UFMF/other formats to AVI
2. Tracking visualizer: Overlay APT keypoint tracking on video

Usage:
    # Convert UFMF to AVI without tracking
    python converter.py --video movie.ufmf --output output.avi

    # Visualize all flies with keypoint tracking
    python converter.py --video movie.ufmf --trk fixed_apt.trk --output output.avi

    # Visualize all flies with body tracking (ctrax_trx.mat, flytracker_trx.mat, fixed_trx.mat, or ctrax_results.mat)
    python converter.py --video movie.ufmf --mat flytracker_trx.mat --output output.avi

    # Visualize specific flies only
    python converter.py --video movie.ufmf --trk fixed_apt.trk --output output.avi --targets 0,1,2

    # Export specific frame range
    python converter.py --video movie.ufmf --trk fixed_apt.trk --output output.avi --start 1000 --num-frames 500
"""

import cv2
import numpy as np
import argparse
import h5py
import scipy.io
import os
from collections import deque
from dotenv import load_dotenv
from movies import Movie

load_dotenv()
FPS = float(os.getenv("FPS", 150.0))
TRAIL_LENGTH = 30


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
        arr = np.asarray(f[ref])
        out.append(arr.reshape(-1))
    return out


def _read_h5_scalar_ref_array(f, ds):
    """
    Read a MATLAB v7.3 field stored as references to scalar datasets.
    Returns a list of Python ints/floats.
    """
    refs = np.asarray(ds).reshape(-1)
    out = []
    for ref in refs:
        val = np.asarray(f[ref]).reshape(-1)
        if val.size != 1:
            raise ValueError(f"Expected scalar dataset, got shape {val.shape}")
        out.append(val[0].item())
    return out


def _load_body_tracking_data_h5(mat_path):
    """
    Load body tracking from MATLAB v7.3 HDF5 trx format.
    Supports both ctrax_trx.mat and flytracker_trx.mat.
    """
    with h5py.File(mat_path, "r") as f:
        if "trx" not in f:
            raise ValueError(f"No 'trx' group found in {mat_path}. Keys: {list(f.keys())}")

        trx = f["trx"]
        if not isinstance(trx, h5py.Group):
            raise ValueError(f"Expected 'trx' to be an HDF5 group, got {type(trx)}")

        required_fields = ["x", "y", "theta", "a", "b", "firstframe", "endframe"]
        missing = [k for k in required_fields if k not in trx]
        if missing:
            raise ValueError(f"Missing required trx fields in {mat_path}: {missing}")

        xs = _read_h5_ref_array(f, trx["x"])
        ys = _read_h5_ref_array(f, trx["y"])
        thetas = _read_h5_ref_array(f, trx["theta"])
        aas = _read_h5_ref_array(f, trx["a"])
        bbs = _read_h5_ref_array(f, trx["b"])
        firstframes = [int(v) for v in _read_h5_scalar_ref_array(f, trx["firstframe"])]
        endframes = [int(v) for v in _read_h5_scalar_ref_array(f, trx["endframe"])]

        n_flies = len(xs)
        n_frames = max(endframes)

        body_x = np.full((n_frames, n_flies), np.nan)
        body_y = np.full((n_frames, n_flies), np.nan)
        body_theta = np.full((n_frames, n_flies), np.nan)
        body_a = np.full((n_frames, n_flies), np.nan)
        body_b = np.full((n_frames, n_flies), np.nan)

        for i in range(n_flies):
            f0 = firstframes[i] - 1   # MATLAB frame indices are 1-based
            f1 = endframes[i]

            x = xs[i]
            y = ys[i]
            theta = thetas[i]
            a = aas[i]
            b = bbs[i]

            expected_len = f1 - f0
            if not (len(x) == len(y) == len(theta) == len(a) == len(b) == expected_len):
                raise ValueError(
                    f"Fly {i}: inconsistent trajectory lengths. "
                    f"Expected {expected_len}, got "
                    f"x={len(x)}, y={len(y)}, theta={len(theta)}, a={len(a)}, b={len(b)}"
                )

            body_x[f0:f1, i] = x
            body_y[f0:f1, i] = y
            body_theta[f0:f1, i] = theta
            body_a[f0:f1, i] = a
            body_b[f0:f1, i] = b

        print(f"Loaded HDF5 trx body tracking: {n_flies} flies, {n_frames} frames")
        return body_x, body_y, body_theta, body_a, body_b


def _load_body_tracking_data_legacy(mat_path):
    """
    Load body tracking from legacy non-v7.3 MATLAB files.
    Supports:
      - fixed_trx.mat with struct array 'trx'
      - ctrax_results.mat with packed arrays
    """
    d = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if "trx" in d:
        trx = d["trx"]
        if np.ndim(trx) == 0:
            trx = [trx]

        n_flies = len(trx)
        n_frames = max(int(fly.endframe) for fly in trx)

        body_x = np.full((n_frames, n_flies), np.nan)
        body_y = np.full((n_frames, n_flies), np.nan)
        body_theta = np.full((n_frames, n_flies), np.nan)
        body_a = np.full((n_frames, n_flies), np.nan)
        body_b = np.full((n_frames, n_flies), np.nan)

        for i, fly in enumerate(trx):
            f0 = int(fly.firstframe) - 1
            f1 = int(fly.endframe)

            body_x[f0:f1, i] = _flatten_numeric(fly.x)
            body_y[f0:f1, i] = _flatten_numeric(fly.y)
            body_theta[f0:f1, i] = _flatten_numeric(fly.theta)
            body_a[f0:f1, i] = _flatten_numeric(fly.a)
            body_b[f0:f1, i] = _flatten_numeric(fly.b)

        print(f"Loaded legacy trx body tracking: {n_flies} flies, {n_frames} frames")
        return body_x, body_y, body_theta, body_a, body_b

    elif "x_pos" in d:
        x = np.asarray(d["x_pos"]).flatten()
        y = np.asarray(d["y_pos"]).flatten()
        theta = np.asarray(d["angle"]).flatten()
        a = np.asarray(d["maj_ax"]).flatten()
        b = np.asarray(d["min_ax"]).flatten()
        ident = np.asarray(d["identity"]).flatten().astype(int)
        ntargets = np.asarray(d["ntargets"]).flatten().astype(int)

        n_frames = len(ntargets)
        n_flies = int(ident.max()) + 1

        body_x = np.full((n_frames, n_flies), np.nan)
        body_y = np.full((n_frames, n_flies), np.nan)
        body_theta = np.full((n_frames, n_flies), np.nan)
        body_a = np.full((n_frames, n_flies), np.nan)
        body_b = np.full((n_frames, n_flies), np.nan)

        idx = 0
        for fidx, nt in enumerate(ntargets):
            for _ in range(nt):
                fid = int(ident[idx])
                body_x[fidx, fid] = x[idx]
                body_y[fidx, fid] = y[idx]
                body_theta[fidx, fid] = theta[idx]
                body_a[fidx, fid] = a[idx]
                body_b[fidx, fid] = b[idx]
                idx += 1

        print(f"Loaded ctrax_results body tracking: {n_flies} identities, {n_frames} frames")
        return body_x, body_y, body_theta, body_a, body_b

    else:
        raise ValueError(f"Unrecognized legacy MATLAB tracking format in {mat_path}")


def load_body_tracking_data(mat_path):
    """
    Unified loader for body tracking.

    Supports:
      - MATLAB v7.3 HDF5 trx files:
          * ctrax_trx.mat
          * flytracker_trx.mat
      - legacy scipy-readable files:
          * fixed_trx.mat
          * ctrax_results.mat
    """
    try:
        return _load_body_tracking_data_legacy(mat_path)
    except NotImplementedError:
        print("Detected MATLAB v7.3 file, using h5py loader...")
        return _load_body_tracking_data_h5(mat_path)


def load_apt_tracking_data(trk_path):
    """Load tracking data from APT .trk file (MATLAB v7.3 / HDF5).

    The .trk file contains separate tracking arrays for each fly.
    Each array has shape (n_frames, 2, n_keypoints) where:
        - Dimension 0: frame number
        - Dimension 1: [x-coordinates, y-coordinates]
        - Dimension 2: keypoint index

    Args:
        trk_path: Path to .trk file

    Returns:
        list of numpy arrays: Each array has shape (n_frames, 2, n_keypoints)
                             Index by: data[frame_num, 0, :] for x-coords
                                      data[frame_num, 1, :] for y-coords
    """
    tracking_data = []

    with h5py.File(trk_path, "r") as f:
        n_targets = f["pTrk"].shape[0]

        for i in range(n_targets):
            ptrk_ref = f["pTrk"][i, 0]
            data = f[ptrk_ref][:]
            tracking_data.append(data)

    print(f"Loaded tracking: {len(tracking_data)} targets/flies")
    if len(tracking_data) > 0:
        n_frames, _, n_keypoints = tracking_data[0].shape
        print(f"  {n_frames} frames, {n_keypoints} keypoints per fly")

    return tracking_data


def draw_body(frame, x, y, theta, a, b, color):
    """Draw a single fly as an ellipse with an orientation line."""
    if np.isnan(x) or np.isnan(y) or np.isnan(theta):
        return
    cx, cy = int(round(x)), int(round(y))
    # a/b are quarter-axes from the MAT file; cv2.ellipse wants semi-axes (half-axes)
    ax_a = max(1, int(round(a * 2)))
    ax_b = max(1, int(round(b * 2)))
    angle_deg = np.degrees(theta)
    cv2.ellipse(frame, (cx, cy), (ax_a, ax_b), angle_deg, 0, 360, color, 2)
    # heading line along the semi-major axis
    ex = int(round(cx + a * 2 * np.cos(theta)))
    ey = int(round(cy + a * 2 * np.sin(theta)))
    cv2.line(frame, (cx, cy), (ex, ey), color, 2)



def draw_trails(frame, history, color, alpha=0.25):
    """Draw a translucent trail of past heading lines from center."""
    if not history:
        return
    overlay = frame.copy()
    for x, y, theta, a, b in history:
        if np.isnan(x) or np.isnan(y) or np.isnan(theta):
            continue
        cx, cy = int(round(x)), int(round(y))
        ex = int(round(cx + a * 2 * np.cos(theta)))
        ey = int(round(cy + a * 2 * np.sin(theta)))
        cv2.line(overlay, (cx, cy), (ex, ey), color, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_keypoints(frame, x_coords, y_coords, color):
    """Draw keypoints on frame.

    Args:
        frame: OpenCV image (BGR format)
        x_coords: Array of x-coordinates for all keypoints
        y_coords: Array of y-coordinates for all keypoints
        color: BGR color tuple (e.g., (0, 255, 0) for green)
    """
    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]

        # Skip NaN values (missing/occluded keypoints)
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def export_video(video_path, trk_path=None, body_trk_path=None, output_path="output.avi",
                 start_frame=0, num_frames=None, target_ids=None, black_bg_output=None, raw_output=None):
    """Export video with optional tracking overlay.

    Args:
        video_path: Path to input video (UFMF, AVI, etc.)
        trk_path: Path to APT .trk file (optional, None = no tracking overlay)
        output_path: Path for output AVI file
        start_frame: Starting frame number (0-indexed)
        num_frames: Number of frames to export (None = all frames)
        target_ids: Comma-separated string of target IDs (e.g., "0,1,2") or None for all
    """
    # Load keypoint tracking data if provided
    if trk_path is not None:
        with h5py.File(trk_path, "r") as f:
            startframes = f["startframes"][:].flatten()
            endframes = f["endframes"][:].flatten()

        tracking_data = load_apt_tracking_data(trk_path)
        n_targets = len(tracking_data)

        if n_targets == 0:
            print("No tracking data found")
            return

        n_tracking_frames = endframes.max()
    else:
        tracking_data = None
        startframes = None
        endframes = None
        n_targets = 0
        n_tracking_frames = float("inf")

    # Load body tracking data if provided
    if body_trk_path is not None:
        body_x, body_y, body_theta, body_a, body_b = load_body_tracking_data(body_trk_path)
    else:
        body_x = body_y = body_theta = body_a = body_b = None

    # Load video using Movie class (handles UFMF and standard formats)
    mov = Movie(video_path)
    n_video_frames = mov.get_n_frames()
    width = mov.get_width()
    height = mov.get_height()

    print(f"Video: {n_video_frames} frames, {FPS:.2f} fps, {width}x{height}")

    # Determine frame range to export
    if num_frames is None:
        end_frame = min(n_video_frames, n_tracking_frames)
    else:
        end_frame = min(start_frame + num_frames, n_video_frames, n_tracking_frames)

    # Parse which targets to visualize
    if target_ids is not None:
        targets_to_draw = [int(t) for t in target_ids.split(",")]
    elif tracking_data is not None:
        targets_to_draw = list(range(n_targets))
    elif body_x is not None:
        targets_to_draw = list(range(body_x.shape[1]))
    else:
        targets_to_draw = []

    print(f"Exporting frames {start_frame} to {end_frame}")
    if targets_to_draw:
        print(f"Drawing targets: {targets_to_draw}")

    # Setup video writer (XVID codec for AVI)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    out_black = cv2.VideoWriter(black_bg_output, fourcc, FPS, (width, height)) if black_bg_output else None
    out_raw   = cv2.VideoWriter(raw_output,      fourcc, FPS, (width, height)) if raw_output   else None

    # Distinct colors for each fly (BGR) — saturated, high-contrast against black and white
    colors = [
        (0,   0,   255),  # Red
        (0,   255, 0),    # Lime
        (0,   220, 255),  # Yellow
        (255, 0,   200),  # Hot pink
        (255, 140, 0),    # Deep sky blue
        (0,   128, 255),  # Orange
        (255, 0,   80),   # Violet-red
        (0,   255, 200),  # Spring green
        (200, 0,   255),  # Magenta-purple
        (0,   200, 255),  # Gold
        (255, 60,  60),   # Dodger blue
        (60,  255, 60),   # Bright green
        (60,  60,  255),  # Coral red
        (0,   255, 255),  # Bright yellow
        (255, 0,   160),  # Deep violet
        (0,   160, 255),  # Dark orange
        (180, 255, 0),    # Neon cyan-green
        (255, 0,   0),    # Pure blue
        (0,   80,  255),  # Scarlet
        (80,  255, 255),  # Neon yellow
    ]

    # Trail history: one deque per fly (body tracking only)
    trail_history = (
        [deque(maxlen=TRAIL_LENGTH) for _ in range(body_x.shape[1])]
        if body_x is not None else []
    )

    # Process frames
    for frame_num in range(start_frame, end_frame):
        frame, _ = mov.get_frame(frame_num)

        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if out_raw is not None:
            out_raw.write(frame)

        black = np.zeros_like(frame) if out_black else None

        # Draw keypoint tracking for selected targets
        if tracking_data is not None:
            for target_id in targets_to_draw:
                if target_id < n_targets:
                    data = tracking_data[target_id]
                    video_frame_1indexed = frame_num + 1
                    if video_frame_1indexed < startframes[target_id] or video_frame_1indexed > endframes[target_id]:
                        continue
                    data_idx = video_frame_1indexed - startframes[target_id]
                    x_coords = data[data_idx, 0, :]
                    y_coords = data[data_idx, 1, :]
                    color = colors[target_id % len(colors)]
                    draw_keypoints(frame, x_coords, y_coords, color)
                    if black is not None:
                        draw_keypoints(black, x_coords, y_coords, color)

        # Draw body tracking for selected targets
        if body_x is not None and frame_num < body_x.shape[0]:
            for target_id in range(body_x.shape[1]):
                color = colors[target_id % len(colors)]
                # Draw translucent trail of past positions
                draw_trails(frame, trail_history[target_id], color)
                if black is not None:
                    draw_trails(black, trail_history[target_id], color)
                # Draw current ellipse
                draw_body(frame,
                          body_x[frame_num, target_id],
                          body_y[frame_num, target_id],
                          body_theta[frame_num, target_id],
                          body_a[frame_num, target_id],
                          body_b[frame_num, target_id],
                          color)
                if black is not None:
                    draw_body(black,
                              body_x[frame_num, target_id],
                              body_y[frame_num, target_id],
                              body_theta[frame_num, target_id],
                              body_a[frame_num, target_id],
                              body_b[frame_num, target_id],
                              color)
                # Append current position to trail for next frame
                trail_history[target_id].append((
                    body_x[frame_num, target_id],
                    body_y[frame_num, target_id],
                    body_theta[frame_num, target_id],
                    body_a[frame_num, target_id],
                    body_b[frame_num, target_id],
                ))

        out.write(frame)
        if out_black is not None:
            out_black.write(black)

        # Progress update every 500 frames
        if (frame_num - start_frame) % 500 == 0:
            print(f"  Progress: {frame_num - start_frame}/{end_frame - start_frame}")

    out.release()
    if out_black is not None:
        out_black.release()
        print(f"Saved black-bg to {black_bg_output}")
    if out_raw is not None:
        out_raw.release()
        print(f"Saved raw to {raw_output}")
    mov.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to AVI and optionally overlay APT tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert UFMF to AVI (no tracking)
  %(prog)s --video movie.ufmf --output movie.avi

  # Visualize all flies with tracking
  %(prog)s --video movie.ufmf --trk fixed_apt.trk --output out.avi

  # Visualize all flies with body tracking
  %(prog)s --video movie.ufmf --mat flytracker_trx.mat --output out.avi

  # Visualize specific flies only
  %(prog)s --video movie.ufmf --trk fixed_apt.trk --output out.avi --targets 0,1,2

  # Export specific frame range
  %(prog)s --video movie.ufmf --trk fixed_apt.trk --output out.avi --start 1000 --num-frames 500
        """
    )

    parser.add_argument("--video", required=True,
                        help="Video file (UFMF, AVI, MP4, etc)")
    parser.add_argument("--trk",
                        help="APT .trk file (HDF5 format) for keypoint tracking overlay")
    parser.add_argument("--mat",
                        help="Body tracking .mat file (ctrax_trx.mat, flytracker_trx.mat, fixed_trx.mat, or ctrax_results.mat)")
    parser.add_argument("--output", default="output.avi",
                        help="Output AVI file (default: output.avi)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame (default: 0)")
    parser.add_argument("--num-frames", type=int,
                        help="Number of frames to export (default: all)")
    parser.add_argument("--targets",
                        help='Comma-separated target IDs to draw, e.g. "0,1,2" (default: all)')
    parser.add_argument("--black-bg-output",
                        help="Also write a second video with tracking on a black background")

    args = parser.parse_args()

    export_video(args.video, args.trk, args.mat, args.output, args.start, args.num_frames, args.targets, args.black_bg_output)


if __name__ == "__main__":
    main()