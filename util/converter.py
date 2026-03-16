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

    # Visualize all flies with body tracking (fixed_trx.mat or ctrax_results.mat)
    python converter.py --video movie.ufmf --mat fixed_trx.mat --output output.avi

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
import sys
import os
from dotenv import load_dotenv
from movies import Movie

load_dotenv()
FPS = float(os.getenv("FPS", 150.0))


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

    with h5py.File(trk_path, 'r') as f:
        # pTrk contains HDF5 references to separate tracking arrays (one per fly)
        n_targets = f['pTrk'].shape[0]

        for i in range(n_targets):
            # Dereference to get actual tracking data
            ptrk_ref = f['pTrk'][i, 0]
            data = f[ptrk_ref][:]  # Shape: (n_frames, 2, n_keypoints)
            tracking_data.append(data)

    print(f"Loaded tracking: {len(tracking_data)} targets/flies")
    if len(tracking_data) > 0:
        n_frames, _, n_keypoints = tracking_data[0].shape
        print(f"  {n_frames} frames, {n_keypoints} keypoints per fly")

    return tracking_data


def load_body_tracking_data(mat_path):
    """Load body tracking from either fixed_trx.mat or ctrax_results.mat.

    Returns:
        body_x:     (n_frames, n_flies) float array, NaN where untracked
        body_y:     (n_frames, n_flies) float array
        body_theta: (n_frames, n_flies) float array
        body_a:     (n_frames, n_flies) semi-major axis in pixels
        body_b:     (n_frames, n_flies) semi-minor axis in pixels
    """
    d = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if "trx" in d:
        # fixed_trx.mat: struct array, one entry per fly, variable start/end frames
        trx = d["trx"]
        n_flies = len(trx)
        n_frames = max(int(fly.endframe) for fly in trx)
        body_x     = np.full((n_frames, n_flies), np.nan)
        body_y     = np.full((n_frames, n_flies), np.nan)
        body_theta = np.full((n_frames, n_flies), np.nan)
        body_a     = np.full((n_frames, n_flies), np.nan)
        body_b     = np.full((n_frames, n_flies), np.nan)
        for i, fly in enumerate(trx):
            f0 = int(fly.firstframe) - 1
            f1 = int(fly.endframe)
            body_x[f0:f1, i]     = fly.x
            body_y[f0:f1, i]     = fly.y
            body_theta[f0:f1, i] = fly.theta
            body_a[f0:f1, i]     = fly.a
            body_b[f0:f1, i]     = fly.b
        print(f"Loaded fixed_trx body tracking: {n_flies} flies, {n_frames} frames")

    elif "x_pos" in d:
        # ctrax_results.mat: packed rows with identity column
        x     = d["x_pos"].flatten()
        y     = d["y_pos"].flatten()
        theta = d["angle"].flatten()
        a     = d["maj_ax"].flatten()
        b     = d["min_ax"].flatten()
        ident    = d["identity"].flatten().astype(int)
        ntargets = d["ntargets"].flatten().astype(int)
        n_frames = len(ntargets)
        n_flies  = int(ident.max()) + 1
        body_x     = np.full((n_frames, n_flies), np.nan)
        body_y     = np.full((n_frames, n_flies), np.nan)
        body_theta = np.full((n_frames, n_flies), np.nan)
        body_a     = np.full((n_frames, n_flies), np.nan)
        body_b     = np.full((n_frames, n_flies), np.nan)
        idx = 0
        for f, nt in enumerate(ntargets):
            for j in range(nt):
                fid = int(ident[idx])
                body_x[f, fid]     = x[idx]
                body_y[f, fid]     = y[idx]
                body_theta[f, fid] = theta[idx]
                body_a[f, fid]     = a[idx]
                body_b[f, fid]     = b[idx]
                idx += 1
        print(f"Loaded ctrax body tracking: {n_flies} identities, {n_frames} frames")

    else:
        raise ValueError(f"Unrecognised body tracking format in {mat_path}")

    return body_x, body_y, body_theta, body_a, body_b


def draw_body(frame, x, y, theta, a, b, color):
    """Draw a single fly as an ellipse with an orientation line."""
    if np.isnan(x) or np.isnan(y) or np.isnan(theta):
        return
    cx, cy = int(round(x)), int(round(y))
    # a/b are semi-axes; cv2.ellipse wants integer half-axes
    ax_a = max(1, int(round(a)))
    ax_b = max(1, int(round(b)))
    angle_deg = np.degrees(theta)
    cv2.ellipse(frame, (cx, cy), (ax_a, ax_b), angle_deg, 0, 360, color, 1)
    # heading line along the major axis
    ex = int(round(cx + a * np.cos(theta)))
    ey = int(round(cy + a * np.sin(theta)))
    cv2.line(frame, (cx, cy), (ex, ey), color, 1)


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


def export_video(video_path, trk_path=None, body_trk_path=None, output_path='output.avi',
                 start_frame=0, num_frames=None, target_ids=None):
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
        with h5py.File(trk_path, 'r') as f:
            startframes = f['startframes'][:].flatten()
            endframes = f['endframes'][:].flatten()

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
        n_tracking_frames = float('inf')

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
        targets_to_draw = [int(t) for t in target_ids.split(',')]
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    # Distinct colors for each fly
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Orange
        (255, 128, 0),  # Sky blue
        (128, 255, 0),  # Spring green
    ]

    # Process frames
    for frame_num in range(start_frame, end_frame):
        frame, _ = mov.get_frame(frame_num)

        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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

        # Draw body tracking for selected targets
        if body_x is not None and frame_num < body_x.shape[0]:
            for target_id in range(body_x.shape[1]):
                if target_id < body_x.shape[1]:
                    color = colors[target_id % len(colors)]
                    draw_body(frame,
                              body_x[frame_num, target_id],
                              body_y[frame_num, target_id],
                              body_theta[frame_num, target_id],
                              body_a[frame_num, target_id],
                              body_b[frame_num, target_id],
                              color)

        out.write(frame)

        # Progress update every 500 frames
        if (frame_num - start_frame) % 500 == 0:
            print(f"  Progress: {frame_num - start_frame}/{end_frame - start_frame}")

    out.release()
    mov.close()
    print(f"Saved to {output_path}")





def main():
    parser = argparse.ArgumentParser(
        description='Convert video to AVI and optionally overlay APT tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert UFMF to AVI (no tracking)
  %(prog)s --video movie.ufmf --output movie.avi

  # Visualize all flies with tracking
  %(prog)s --video movie.ufmf --trk fixed_apt.trk --output out.avi

  # Visualize specific flies only
  %(prog)s --video movie.ufmf --trk fixed_apt.trk --output out.avi --targets 0,1,2

  # Export specific frame range
  %(prog)s --video movie.ufmf --trk fixed_apt.trk --output out.avi --start 1000 --num-frames 500
        """
    )

    parser.add_argument('--video', required=True,
                       help='Video file (UFMF, AVI, MP4, etc)')
    parser.add_argument('--trk',
                       help='APT .trk file (HDF5 format) for keypoint tracking overlay')
    parser.add_argument('--mat',
                       help='Body tracking .mat file (fixed_trx.mat or ctrax_results.mat)')
    parser.add_argument('--output', default='output.avi',
                       help='Output AVI file (default: output.avi)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start frame (default: 0)')
    parser.add_argument('--num-frames', type=int,
                       help='Number of frames to export (default: all)')
    parser.add_argument('--targets',
                       help='Comma-separated target IDs to draw, e.g. "0,1,2" (default: all)')

    args = parser.parse_args()

    export_video(args.video, args.trk, args.mat, args.output, args.start, args.num_frames, args.targets)


if __name__ == '__main__':
    main()
