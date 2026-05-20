#!/usr/bin/env python3
"""
Convert a movie + tracking file into 3x3 grid subclips with tracking overlay.

Pipeline:
  1. Render tracking overlay onto the video (via util/converter.py)
  2. Slice the result into 3-second clips, each cropped into a 3x3 grid with 30px overlap

Usage:
    python make_subclips.py --video movie.ufmf --trk fixed_apt.trk
    python make_subclips.py --video movie.ufmf --mat fixed_trx.mat --output-dir out --n 20
    python make_subclips.py --video movie.ufmf --trk fixed_apt.trk --black-bg
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "util"))
from converter import export_video, FPS
from trim_video import slice_and_crop_multi, get_video_duration, compute_tiles, get_video_dimensions


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


def main():
    _load_inputs_env()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=os.environ.get("VIDEO"),
                        help="Input video file. Default: $VIDEO from inputs.env")
    parser.add_argument("--trk", default=os.environ.get("TRK"),
                        help="APT .trk keypoint tracking file. Default: $TRK from inputs.env")
    parser.add_argument("--mat", default=os.environ.get("MAT"),
                        help="Body tracking .mat file. Default: $MAT from inputs.env")
    parser.add_argument("--output-dir", default=os.environ.get("CLIPS_DIR", "subclips"),
                        help="Output directory. Default: subclips or $CLIPS_DIR from inputs.env")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of 3-second clips to produce (default: entire video)")
    parser.add_argument("--start", type=int, default=0, help="Start frame for converter (default: 0)")
    parser.add_argument("--targets", help="Comma-separated target IDs to draw (default: all)")
    parser.add_argument("--black-bg", action="store_true",
                        help="Also produce subclips from a black-background tracking-only render")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel ffmpeg jobs for cropping (default: 4)")
    args = parser.parse_args()

    if not args.video:
        parser.error("--video is required (or set VIDEO in inputs.env)")

    os.makedirs(args.output_dir, exist_ok=True)

    overlaid = os.path.join(args.output_dir, "_overlaid.avi")
    black_overlaid = os.path.join(args.output_dir, "_overlaid_black.avi") if args.black_bg else None
    raw_overlaid = os.path.join(args.output_dir, "_raw.avi")

    already_rendered = (os.path.exists(overlaid) and os.path.getsize(overlaid) > 0 and
                        os.path.exists(raw_overlaid) and os.path.getsize(raw_overlaid) > 0)
    if args.black_bg:
        already_rendered = already_rendered and os.path.exists(black_overlaid) and os.path.getsize(black_overlaid) > 0

    if already_rendered:
        print("=== Skipping render (overlaid video already exists) ===")
    else:
        print("=== Rendering tracking overlay ===")
        export_video(
            video_path=args.video,
            trk_path=args.trk,
            body_trk_path=args.mat,
            output_path=overlaid,
            start_frame=args.start,
            target_ids=args.targets,
            black_bg_output=black_overlaid,
            raw_output=raw_overlaid,
        )

    if not os.path.exists(overlaid) or os.path.getsize(overlaid) == 0:
        print(f"ERROR: overlaid video was not created at {overlaid}")
        sys.exit(1)

    # Determine clip count and tile layout from the rendered video
    if args.n is None:
        n_clips = int(get_video_duration(overlaid) // 3)
    else:
        n_clips = args.n
    W, H = get_video_dimensions(overlaid)
    tiles = compute_tiles(W, H)

    print(f"\n=== Cropping clips ===")
    inputs = [
        (raw_overlaid, os.path.join(args.output_dir, "raw")),
        (overlaid,     os.path.join(args.output_dir, "overlay")),
    ]
    if black_overlaid:
        inputs.append((black_overlaid, os.path.join(args.output_dir, "black_bg")))
    slice_and_crop_multi(inputs, n_clips, args.workers)

    # Build frame manifest: maps each output file to its frame range in the original video
    frames_per_clip = int(3 * FPS)
    manifest = {}
    for i in range(n_clips):
        orig_start = args.start + i * frames_per_clip
        orig_end   = orig_start + frames_per_clip - 1

        base = {"start_frame": orig_start, "end_frame": orig_end,
                "start_time_s": orig_start / FPS, "end_time_s": (orig_end + 1) / FPS}

        for row, col, x, y, cw, ch in tiles:
            tile_meta = {**base, "tile": {"row": row, "col": col},
                         "crop": {"x": x, "y": y, "w": cw, "h": ch}}
            manifest[f"overlay/clip_{i}_tile_{row}_{col}.mp4"]   = tile_meta
            manifest[f"raw/clip_{i}_tile_{row}_{col}.mp4"] = tile_meta.copy()
            if black_overlaid:
                manifest[f"black_bg/clip_{i}_tile_{row}_{col}.mp4"] = tile_meta.copy()

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
