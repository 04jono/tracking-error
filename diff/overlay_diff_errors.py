#!/usr/bin/env python3
"""
overlay_diff_errors.py

Read a diff TSV and an overlaid video, then draw a red circle on each frame
where an error was detected.

Expected TSV columns:
    frame    fly    type    x    y    ...

Notes:
- frame is assumed to be 1-indexed
- x, y are assumed to be image coordinates in pixels
- rows with NaN x/y are skipped
"""

import argparse
import math
import csv
from collections import defaultdict

import cv2


def load_diff_points(tsv_path):
    """
    Returns:
        errors_by_frame: dict[int, list[dict]]
            keyed by 0-indexed frame number
    """
    errors_by_frame = defaultdict(list)

    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"frame", "type", "x", "y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required TSV columns: {sorted(missing)}")

        for row in reader:
            frame_str = row["frame"].strip()
            x_str = row["x"].strip()
            y_str = row["y"].strip()

            if not frame_str:
                continue

            frame_1idx = int(frame_str)
            frame_0idx = frame_1idx - 1

            try:
                x = float(x_str)
                y = float(y_str)
            except ValueError:
                continue

            if math.isnan(x) or math.isnan(y):
                continue

            errors_by_frame[frame_0idx].append(
                {
                    "type": row.get("type", "").strip(),
                    "fly": row.get("fly", "").strip(),
                    "x": x,
                    "y": y,
                }
            )

    return errors_by_frame


def overlay_errors_on_video(
    input_video_path,
    diff_tsv_path,
    output_video_path,
    radius=18,
    thickness=2,
    draw_label=False,
):
    errors_by_frame = load_diff_points(diff_tsv_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {output_video_path}")

    frame_idx = 0
    red = (0, 0, 255)  # BGR
    label_color = (0, 0, 255)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in errors_by_frame:
            for err in errors_by_frame[frame_idx]:
                x = int(round(err["x"]))
                y = int(round(err["y"]))

                # Circle
                cv2.circle(frame, (x, y), radius, red, thickness)

                # Optional center dot
                cv2.circle(frame, (x, y), 2, red, -1)

                if draw_label:
                    label = err["type"]
                    if err["fly"] != "":
                        label += f" fly={err['fly']}"

                    cv2.putText(
                        frame,
                        label,
                        (x + radius + 4, y - radius - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        label_color,
                        1,
                        cv2.LINE_AA,
                    )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"Processed {frame_idx} / {n_frames} frames")
    print(f"Saved output to {output_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay red circles on a video using error locations from a diff TSV"
    )
    parser.add_argument("video", help="Path to overlaid input video")
    parser.add_argument("diff", help="Path to diff TSV")
    parser.add_argument("output", help="Path to output annotated video")
    parser.add_argument("--radius", type=int, default=18, help="Circle radius in pixels")
    parser.add_argument("--thickness", type=int, default=2, help="Circle thickness")
    parser.add_argument(
        "--draw-label",
        action="store_true",
        help="Draw error type text next to each circle",
    )
    args = parser.parse_args()

    overlay_errors_on_video(
        input_video_path=args.video,
        diff_tsv_path=args.diff,
        output_video_path=args.output,
        radius=args.radius,
        thickness=args.thickness,
        draw_label=args.draw_label,
    )


if __name__ == "__main__":
    main()