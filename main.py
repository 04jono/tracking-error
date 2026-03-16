import argparse
import asyncio
import cv2
from pathlib import Path
from openai import AsyncOpenAI

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets" / "videos"
PROMPTS_DIR = SCRIPT_DIR / "assets" / "prompts"

DEFAULT_MODEL = "/share/j_sun/jqc3/Qwen3.5-27B-FP8"

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (PROMPTS_DIR / "body_tracking.txt").read_text().strip()
}


def get_clip_paths(clips_dir):
    """Return clip paths from clips_dir sorted by numeric index."""
    clips_dir = Path(clips_dir)
    paths = sorted(clips_dir.glob("clip_*.mp4"), key=lambda p: int(p.stem.split("_")[1]))
    return [str(p) for p in paths]


def get_clip_duration_seconds(path):
    """Return duration of a video in seconds using OpenCV."""
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


async def process_video(client, video_path, model):
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path}"},
                },
                {
                    "type": "text",
                    "text": "Do the keypoint annotations correctly track the animal's keypoints throughout this video clip?"
                }
            ]
        }
    ]

    print(f"Processing {video_path}")

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
    return video_path, response.choices[0].message.content


async def main():
    parser = argparse.ArgumentParser(description="Run keypoint annotation quality control on video clips.")
    parser.add_argument("--port",       type=int, required=True, help="vLLM server port")
    parser.add_argument("--model",      type=str, default=DEFAULT_MODEL, help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--clips-dir",  type=str, default="clips", help="Directory containing clip_N.mp4 files (default: clips)")
    parser.add_argument("--output",     type=str, default="results.out", help="Output file for results (default: results.out)")
    args = parser.parse_args()

    video_paths = get_clip_paths(args.clips_dir)
    if not video_paths:
        print(f"No clips found in {args.clips_dir}")
        return

    clip_duration = get_clip_duration_seconds(video_paths[0])
    print(f"model: {args.model} | port: {args.port}")
    print(f"Found {len(video_paths)} clips, {clip_duration:.2f}s each")

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key=""
    )

    tasks = [process_video(client, path, args.model) for path in video_paths]
    results = await asyncio.gather(*tasks)

    # Sort results by clip index to ensure chronological output
    results = sorted(results, key=lambda r: int(Path(r[0]).stem.split("_")[1]))

    lines = []
    for i, (video_path, result) in enumerate(results):
        t_start = i * clip_duration
        t_end   = t_start + clip_duration
        ts = f"{format_timestamp(t_start)}-{format_timestamp(t_end)}"
        line = f"[{ts}] {Path(video_path).name}: {result}"
        print(line)
        lines.append(line)

    out_path = Path(args.output)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {out_path}")


asyncio.run(main())
