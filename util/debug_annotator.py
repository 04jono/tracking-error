#!/usr/bin/env python3
"""
Annotate a video with error frames from a TSV produced by diff_body_tracking.py.

Each error frame gets a red border, an ERROR label, and a circle at the error
location. wrong_count frames have no coordinates so only the border/label is drawn.

Usage:
    python annotator.py --video body_tracking.avi --errors errors.tsv --output annotated.avi
    python annotator.py --video body_tracking.avi --errors errors.tsv --output annotated.avi --start 1000 --num-frames 500
"""

import csv
import cv2
import math
import numpy as np
import argparse
import os
from dotenv import load_dotenv
from movies import Movie

load_dotenv()
FPS              = float(os.getenv("FPS", 150.0))
BORDER_THICKNESS = 6
BORDER_COLOR     = (0, 0, 255)   # red (BGR)
LABEL_COLOR      = (0, 0, 255)
CIRCLE_COLOR     = (0, 0, 255)
CIRCLE_RADIUS    = 30
CIRCLE_THICKNESS = 3
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 1.2
FONT_THICKNESS   = 2


def load_errors(path):
    """Return {frame_1idx: (x, y) or None} from a TSV produced by diff_body_tracking.py."""
    frames = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            frame = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            coords = None if math.isnan(x) or math.isnan(y) else (x, y)
            frames[frame] = coords
    print(f"Loaded {len(frames)} error frames from {path}")
    return frames


def annotate_frame(frame, frame_1idx, coords=None):
    """Draw red border, ERROR label, and optional circle at error coordinates."""
    h, w = frame.shape[:2]
    t = BORDER_THICKNESS

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), BORDER_COLOR, t)

    label = f"ERROR  frame {frame_1idx}"
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 4
    x0, y0 = t + pad, t + pad
    cv2.rectangle(frame, (x0, y0), (x0 + tw + pad, y0 + th + baseline + pad), (0, 0, 0), -1)
    cv2.putText(frame, label, (x0 + pad, y0 + th + pad // 2),
                FONT, FONT_SCALE, LABEL_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    if coords is not None:
        cx, cy = int(round(coords[0])), int(round(coords[1]))
        cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)


def annotate_video(video_path, errors_path, output_path="annotated.avi",
                   start_frame=0, num_frames=None, padding=5):
    error_frames = load_errors(errors_path)

    # Build a map from every frame_1idx that should be annotated -> (origin_frame_1idx, coords)
    # padding frames before/after each error use the error's coordinates
    padded = {}
    for f, coords in error_frames.items():
        for offset in range(-padding, padding + 1):
            pf = f + offset
            if pf not in padded:
                padded[pf] = (f, coords)

    mov = Movie(video_path)
    n_video_frames = mov.get_n_frames()
    width  = mov.get_width()
    height = mov.get_height()
    print(f"Video: {n_video_frames} frames, {FPS:.2f} fps, {width}x{height}")

    end_frame = min(start_frame + num_frames, n_video_frames) if num_frames else n_video_frames

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    n_annotated = 0
    for frame_num in range(start_frame, end_frame):
        frame, _ = mov.get_frame(frame_num)

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_1idx = frame_num + 1
        if frame_1idx in padded:
            origin, coords = padded[frame_1idx]
            annotate_frame(frame, origin, coords=coords)
            n_annotated += 1

        out.write(frame)

        if (frame_num - start_frame) % 500 == 0:
            print(f"  Progress: {frame_num - start_frame}/{end_frame - start_frame}")

    out.release()
    mov.close()
    print(f"Annotated {n_annotated} error frames")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate video with error highlights from diff_body_tracking.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video body_tracking.avi --errors errors.tsv --output annotated.avi
  %(prog)s --video body_tracking.avi --errors errors.tsv --output annotated.avi --start 1000 --num-frames 500
        """
    )
    parser.add_argument("--video",      required=True, help="Input video file")
    parser.add_argument("--errors",     required=True, help="TSV from diff_body_tracking.py --output")
    parser.add_argument("--output",     default="annotated.avi", help="Output AVI (default: annotated.avi)")
    parser.add_argument("--start",      type=int, help="Start frame, 0-indexed (default: 0)")
    parser.add_argument("--num-frames", type=int, help="Number of frames to process (default: all)")
    parser.add_argument("--padding",    type=int, default=5, help="Frames to annotate before/after each error (default: 5)")

    args = parser.parse_args()
    annotate_video(args.video, args.errors, args.output,
                   start_frame=args.start or 0,
                   num_frames=args.num_frames,
                   padding=args.padding)


if __name__ == "__main__":
    main()
