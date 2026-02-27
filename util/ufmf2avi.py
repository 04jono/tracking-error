#!/usr/bin/env python
"""Convert a UFMF file to AVI format."""

import argparse
import sys

import cv2
from movies import Movie


def convert_ufmf_to_avi(input_path, output_path, fps=None):
    """Convert a UFMF file to AVI.

    Args:
        input_path: Path to input UFMF file
        output_path: Path to output AVI file
        fps: Frames per second (estimated from timestamps if not provided)
    """
    mov = Movie(input_path)

    n_frames = mov.get_n_frames()
    width = mov.get_width()
    height = mov.get_height()

    # Estimate fps from timestamps if not provided
    if fps is None:
        timestamps = mov.get_some_timestamps(0, min(100, n_frames))
        if len(timestamps) > 1:
            fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        else:
            fps = 30.0
        print(f"Estimated fps: {fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    if not out.isOpened():
        print(f"Error: Could not open output file {output_path}")
        mov.close()
        sys.exit(1)

    print(f"Converting {n_frames} frames...")
    for i in range(n_frames):
        frame, timestamp = mov.get_frame(i)
        out.write(frame)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_frames}")

    out.release()
    mov.close()
    print(f"Done. Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert UFMF to AVI")
    parser.add_argument("input", help="Input UFMF file")
    parser.add_argument("output", help="Output AVI file")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second (estimated from timestamps if not provided)")
    args = parser.parse_args()

    convert_ufmf_to_avi(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
