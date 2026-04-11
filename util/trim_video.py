#!/usr/bin/env python3
"""Slice a video into 3-second clips, each cropped into a 3x3 grid with 30px overlap."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_video_dimensions(path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path
    ], capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    return video["width"], video["height"]


def get_video_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_entries", "format=duration", path
    ], capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def compute_tiles(W, H, overlap=30):
    tile_w = W // 2
    tile_h = H // 2
    tiles = []
    for row in range(2):
        for col in range(2):
            base_x = col * tile_w
            base_y = row * tile_h
            tile_right = W if col == 1 else base_x + tile_w
            tile_bottom = H if row == 1 else base_y + tile_h
            x = max(0, base_x - overlap)
            y = max(0, base_y - overlap)
            crop_w = min(W, tile_right + overlap) - x
            crop_h = min(H, tile_bottom + overlap) - y
            tiles.append((row, col, x, y, crop_w, crop_h))
    return tiles


def process_clip(i, input_video, output_dir, tiles):
    start_time = i * 3
    prefix = os.path.join(output_dir, f"clip_{i}")

    n = len(tiles)
    split = f"[0:v]split={n}" + "".join(f"[s{j}]" for j in range(n)) + ";"
    crops = ";".join(
        f"[s{j}]crop={cw}:{ch}:{x}:{y}[t{j}]"
        for j, (_, _, x, y, cw, ch) in enumerate(tiles)
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time), "-t", "3", "-i", input_video,
        "-filter_complex", split + crops,
    ]
    for j, (row, col, *_) in enumerate(tiles):
        cmd += ["-map", f"[t{j}]", "-c:v", "mpeg4", "-q:v", "3",
                f"{prefix}_tile_{row}_{col}.mp4"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR clip_{i}:\n{result.stderr}")
    return i, start_time


def slice_clip(i, input_video, output_dir):
    start_time = i * 3
    out = os.path.join(output_dir, f"clip_{i}.mp4")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_time), "-t", "3", "-i", input_video,
        "-c", "copy", out
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR clip_{i}:\n{result.stderr}")
    return i, start_time


def slice_only(input_video, output_dir, n_clips=None, workers=4):
    os.makedirs(output_dir, exist_ok=True)
    if n_clips is None:
        n_clips = int(get_video_duration(input_video) // 3)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(slice_clip, i, input_video, output_dir): i
                   for i in range(n_clips)}
        for future in as_completed(futures):
            i, start_time = future.result()
            print(f"  Done clip_{i} (seconds {start_time}-{start_time+2})")


def slice_and_crop(input_video, output_dir, n_clips=None, workers=4):
    os.makedirs(output_dir, exist_ok=True)
    W, H = get_video_dimensions(input_video)
    tiles = compute_tiles(W, H)

    if n_clips is None:
        n_clips = int(get_video_duration(input_video) // 3)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_clip, i, input_video, output_dir, tiles): i
                   for i in range(n_clips)}
        for future in as_completed(futures):
            i, start_time = future.result()
            print(f"  Done clip_{i} (seconds {start_time}-{start_time+2})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input video file")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of clips to create (default: entire video)")
    parser.add_argument("--output-dir", default="clips", help="Output directory (default: clips)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel ffmpeg jobs (default: 4)")
    args = parser.parse_args()

    slice_and_crop(args.input_file, args.output_dir, args.n, args.workers)


if __name__ == "__main__":
    main()
