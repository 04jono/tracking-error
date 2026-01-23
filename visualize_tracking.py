#!/usr/bin/env python3
"""
Visualize fly tracking data overlaid on video frames.
"""

import cv2
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_tracking_data(mat_path):
    """Load tracking data from .mat file."""
    mat = sio.loadmat(mat_path)
    trk = mat['trk']
    data = trk['data'][0, 0]
    names = [n[0] for n in trk['names'][0, 0][0]]
    flag_frames = trk['flag_frames'][0, 0].flatten()

    # Create index mapping for easier access
    idx = {name: i for i, name in enumerate(names)}

    return data, names, idx, flag_frames


def draw_fly(frame, fly_data, idx, color):
    """Draw fly body and wings on frame."""
    # Body position
    pos_x = fly_data[idx['pos x']]
    pos_y = fly_data[idx['pos y']]

    if np.isnan(pos_x) or np.isnan(pos_y):
        return

    center = (int(pos_x), int(pos_y))

    # Orientation and body size
    ori = fly_data[idx['ori']]
    major = fly_data[idx['major axis len']]
    minor = fly_data[idx['minor axis len']]

    # Draw body ellipse
    if not np.isnan(major) and not np.isnan(minor):
        axes_len = (int(major / 2), int(minor / 2))
        # FlyTracker uses standard math convention (CCW from +x, y-up)
        # OpenCV uses image coordinates (y-down), so negate the angle
        angle = -np.degrees(ori) if not np.isnan(ori) else 0
        cv2.ellipse(frame, center, axes_len, angle, 0, 360, color, 2)

    # Draw body center
    cv2.circle(frame, center, 3, color, -1)

    # Wing positions
    wing_l_x = fly_data[idx['wing l x']]
    wing_l_y = fly_data[idx['wing l y']]
    wing_r_x = fly_data[idx['wing r x']]
    wing_r_y = fly_data[idx['wing r y']]

    # Draw left wing
    if not np.isnan(wing_l_x) and not np.isnan(wing_l_y):
        wing_l = (int(wing_l_x), int(wing_l_y))
        cv2.circle(frame, wing_l, 4, color, -1)
        cv2.line(frame, center, wing_l, color, 1)

    # Draw right wing
    if not np.isnan(wing_r_x) and not np.isnan(wing_r_y):
        wing_r = (int(wing_r_x), int(wing_r_y))
        cv2.circle(frame, wing_r, 4, color, -1)
        cv2.line(frame, center, wing_r, color, 1)


def visualize_frames(video_path, mat_path, output_path='tracking_overlay.png',
                     sample_frames=None):
    """Create visualization of tracking data on sample frames."""
    # Load tracking data
    data, names, idx, flag_frames = load_tracking_data(mat_path)
    n_flies, n_frames, n_features = data.shape

    print(f"Loaded tracking data: {n_flies} flies, {n_frames} frames, {n_features} features")

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frames: {total_video_frames}")

    # Default sample frames
    if sample_frames is None:
        sample_frames = [0, 1000, 5000, 10000, 20000, 50000]

    # Filter out frames beyond video length
    sample_frames = [f for f in sample_frames if f < min(total_video_frames, n_frames)]

    # Colors for each fly (BGR format for OpenCV)
    colors = [(0, 0, 255), (0, 255, 0)]  # Red and Green

    # Create subplot grid
    n_samples = len(sample_frames)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, frame_num in enumerate(sample_frames):
        ax = axes[ax_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            ax.set_title(f"Frame {frame_num} - Could not read")
            ax.axis('off')
            continue

        # Draw tracking data for each fly
        for fly_id in range(n_flies):
            fly_data = data[fly_id, frame_num, :]
            draw_fly(frame, fly_data, idx, colors[fly_id])

        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {frame_num}")
        ax.axis('off')

    # Hide unused axes
    for ax_idx in range(len(sample_frames), len(axes)):
        axes[ax_idx].axis('off')

    cap.release()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    plt.close()


def export_video_clip(video_path, mat_path, output_path='tracking_clip.mp4',
                      start_frame=0, num_frames=1000):
    """Export a video clip with tracking overlay."""
    # Load tracking data
    data, names, idx, flag_frames = load_tracking_data(mat_path)
    n_flies, n_tracking_frames, n_features = data.shape

    print(f"Loaded tracking data: {n_flies} flies, {n_tracking_frames} frames")

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {total_video_frames} frames, {fps} fps, {width}x{height}")

    # Validate frame range
    end_frame = min(start_frame + num_frames, total_video_frames, n_tracking_frames)
    actual_frames = end_frame - start_frame

    print(f"Exporting frames {start_frame} to {end_frame} ({actual_frames} frames)")

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Colors for each fly (BGR format)
    colors = [(0, 0, 255), (0, 255, 0)]  # Red and Green

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_num}")
            break

        # Draw tracking data for each fly
        for fly_id in range(n_flies):
            fly_data = data[fly_id, frame_num, :]
            draw_fly(frame, fly_data, idx, colors[fly_id])

        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

        # Progress update
        if (frame_num - start_frame) % 100 == 0:
            progress = (frame_num - start_frame) / actual_frames * 100
            print(f"Progress: {progress:.1f}%")

    cap.release()
    out.release()
    print(f"Saved video clip to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize fly tracking data on video')
    parser.add_argument('--video', type=str, default='movie1.avi',
                        help='Path to video file')
    parser.add_argument('--mat', type=str,
                        default='BoyMeetsBoy_wo_movies/Boy_meets_boy light/movie1/movie1_track.mat',
                        help='Path to tracking .mat file')
    parser.add_argument('--output', type=str, default='tracking_overlay.png',
                        help='Output path (image or video)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Specific frame numbers to visualize (image mode)')

    # Video clip options
    parser.add_argument('--clip', action='store_true',
                        help='Export video clip instead of image')
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame for video clip')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='Number of frames to export in clip')

    args = parser.parse_args()

    if args.clip:
        output = args.output if args.output.endswith('.mp4') else 'tracking_clip.mp4'
        export_video_clip(args.video, args.mat, output, args.start, args.num_frames)
    else:
        visualize_frames(args.video, args.mat, args.output, args.frames)


if __name__ == '__main__':
    main()
