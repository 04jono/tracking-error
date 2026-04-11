import argparse
import asyncio
import cv2
import json
import time
from pathlib import Path
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).parent
PROMPTS_DIR = SCRIPT_DIR / "assets" / "prompts"

DEFAULT_MODEL = "/share/j_sun/jqc3/Qwen3.5-27B-FP8"

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (PROMPTS_DIR / "body_tracking.txt").read_text().strip()
}


def get_clip_groups(clips_dir):
    """Return sorted list of (clip_idx, tile_row, tile_col, uncropped, overlay, black_bg) tuples."""
    clips_dir = Path(clips_dir)
    overlay_dir  = clips_dir / "overlay"
    uncropped_dir = clips_dir / "uncropped"
    black_bg_dir  = clips_dir / "black_bg"

    groups = []
    for tile_path in sorted(overlay_dir.glob("clip_*_tile_*.mp4")):
        parts = tile_path.stem.split("_")  # clip, N, tile, R, C
        clip_idx, row, col = int(parts[1]), int(parts[3]), int(parts[4])

        uncropped = uncropped_dir / f"clip_{clip_idx}.mp4"
        black_bg  = black_bg_dir  / f"clip_{clip_idx}_tile_{row}_{col}.mp4"

        if not uncropped.exists() or not black_bg.exists():
            continue

        groups.append((clip_idx, row, col, str(uncropped), str(tile_path), str(black_bg)))

    return sorted(groups)


def get_clip_duration_seconds(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps if fps > 0 else 0.0


def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def video_url(path):
    return {"type": "video_url", "video_url": {"url": f"file://{path}"}}


async def process_clip(client, clip_idx, row, col, uncropped, overlay, black_bg, model):
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                video_url(uncropped),
                video_url(overlay),
                video_url(black_bg),
                {
                    "type": "text",
                    "text": (
                        "You are given three views of the same video clip:\n"
                        "1. The full uncropped video with tracking overlay\n"
                        "2. A cropped sub-region with tracking overlay\n"
                        "3. The tracking data alone on a black background for that same sub-region\n\n"
                        "Focus on the cropped sub-region (videos 2 and 3). "
                        "Do the keypoint annotations correctly track the animal's keypoints throughout this clip? "
                        "You may use the full video (1) for context. "
                        "Identify any errors in the cropped region specifically."
                    )
                }
            ]
        }
    ]

    print(f"Processing clip_{clip_idx}_tile_{row}_{col}")

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return clip_idx, row, col, response.choices[0].message.content


async def main():
    parser = argparse.ArgumentParser(description="Run keypoint annotation quality control on video clips.")
    parser.add_argument("--port",      type=int, required=True, help="vLLM server port")
    parser.add_argument("--model",     type=str, default=DEFAULT_MODEL, help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--clips-dir", type=str, default="subclips", help="Subclips directory from make_subclips.py (default: subclips)")
    parser.add_argument("--output",    type=str, default="results.out", help="Output file for results (default: results.out)")
    args = parser.parse_args()

    groups = get_clip_groups(args.clips_dir)
    if not groups:
        print(f"No clip groups found in {args.clips_dir}")
        return

    clip_duration = get_clip_duration_seconds(groups[0][3])  # duration of first uncropped clip
    print(f"model: {args.model} | port: {args.port}")
    print(f"Found {len(groups)} tile-clips, {clip_duration:.2f}s each")

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key=""
    )

    manifest_path = Path(args.clips_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    t0 = time.monotonic()
    tasks = [process_clip(client, *g, args.model) for g in groups]
    results = await asyncio.gather(*tasks)

    results = sorted(results, key=lambda r: (r[0], r[1], r[2]))

    lines = []
    for clip_idx, row, col, result in results:
        manifest_key = f"overlay/clip_{clip_idx}_tile_{row}_{col}.mp4"
        meta = manifest.get(manifest_key, {})

        start_frame = meta.get("start_frame", clip_idx * int(clip_duration * 150))
        end_frame   = meta.get("end_frame",   start_frame + int(clip_duration * 150) - 1)
        t_start     = meta.get("start_time_s", clip_idx * clip_duration)
        t_end       = meta.get("end_time_s",   t_start + clip_duration)

        first_word = result.strip().split()[0].rstrip(":.").upper() if result.strip() else ""
        verdict = first_word if first_word in ("CORRECT", "ERROR") else "UNKNOWN"

        header = (
            f"{verdict} | "
            f"clip_{clip_idx}_tile_{row}_{col} | "
            f"frames {start_frame}-{end_frame} | "
            f"{format_timestamp(t_start)}-{format_timestamp(t_end)}"
        )
        line = f"[{header}]\n{result}\n"
        print(line)
        lines.append(line)

    Path(args.output).write_text("\n".join(lines))
    print(f"Results written to {args.output}")
    print(f"Total time: {time.monotonic() - t0:.1f}s")


asyncio.run(main())
