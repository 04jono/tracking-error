#!/usr/bin/env python3
"""
Convert video files to AVI and optionally overlay APT tracking data.

Can be used as:
1. Video converter: Convert UFMF/other formats to AVI
2. Tracking visualizer: Overlay APT keypoint tracking on video

Usage:
    # Convert UFMF to AVI without tracking
    python converter.py --video movie.ufmf --output output.avi

    # Visualize all flies with tracking
    python converter.py --video movie.ufmf --trk fixed_apt.trk --output output.avi

    # Visualize specific flies only
    python converter.py --video movie.ufmf --trk fixed_apt.trk --output output.avi --targets 0,1,2

    # Export specific frame range
    python converter.py --video movie.ufmf --trk fixed_apt.trk --output output.avi --start 1000 --num-frames 500
"""

import cv2
import numpy as np
import argparse
import h5py
import sys
import os
from movies import Movie


FPS = 30.0


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


def export_video(video_path, trk_path=None, output_path='output.avi',
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
    # Load tracking data if provided
    if trk_path is not None:
        with h5py.File(trk_path, 'r') as f:
            # Load start and end frames for each target
            startframes = f['startframes'][:].flatten()
            endframes = f['endframes'][:].flatten()

        tracking_data = load_apt_tracking_data(trk_path)
        n_targets = len(tracking_data)

        if n_targets == 0:
            print("No tracking data found")
            return

        n_tracking_frames = endframes.max()
    else:
        # No tracking - just convert video
        tracking_data = None
        startframes = None
        endframes = None
        n_targets = 0
        n_tracking_frames = float('inf')  # No limit from tracking

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
    if tracking_data is not None:
        if target_ids is None:
            targets_to_draw = list(range(n_targets))
        else:
            targets_to_draw = [int(t) for t in target_ids.split(',')]

        print(f"Exporting frames {start_frame} to {end_frame}")
        print(f"Drawing targets: {targets_to_draw}")
    else:
        targets_to_draw = []
        print(f"Converting {end_frame - start_frame} frames (no tracking overlay)")

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

        # Draw tracking for selected targets
        for target_id in targets_to_draw:
            if target_id < n_targets:
                data = tracking_data[target_id]

                # Check if current video frame is within this target's tracking range
                # Note: frame numbers in .trk are 1-indexed (MATLAB convention)
                # but frame_num here is 0-indexed
                video_frame_1indexed = frame_num + 1

                if video_frame_1indexed < startframes[target_id] or video_frame_1indexed > endframes[target_id]:
                    continue  # This target isn't tracked at this frame

                # Calculate index into tracking data array
                # data[0] corresponds to startframes[target_id]
                data_idx = video_frame_1indexed - startframes[target_id]

                # Extract x and y coordinates for all keypoints at this frame
                # data[frame, 0, :] = all x-coordinates
                # data[frame, 1, :] = all y-coordinates
                x_coords = data[data_idx, 0, :]
                y_coords = data[data_idx, 1, :]

                color = colors[target_id % len(colors)]
                draw_keypoints(frame, x_coords, y_coords, color)

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
                       help='APT .trk file (HDF5 format) - optional, omit for video conversion only')
    parser.add_argument('--output', default='output.avi',
                       help='Output AVI file (default: output.avi)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start frame (default: 0)')
    parser.add_argument('--num-frames', type=int,
                       help='Number of frames to export (default: all)')
    parser.add_argument('--targets',
                       help='Comma-separated target IDs to draw, e.g. "0,1,2" (default: all)')

    args = parser.parse_args()

    export_video(args.video, args.trk, args.output, args.start, args.num_frames, args.targets)


if __name__ == '__main__':
    main()
