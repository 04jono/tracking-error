import argparse
import asyncio
import cv2
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI


def _load_inputs_env(filename="inputs.env"):
    p = Path.cwd()
    for _ in range(4):
        candidate = p / filename
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return
        p = p.parent


class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._file = open(log_path, "w", buffering=1)
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

SCRIPT_DIR = Path(__file__).parent
PROMPTS_DIR = SCRIPT_DIR / "assets" / "prompts"
EXAMPLES_DIR = SCRIPT_DIR / "assets" / "examples"


def make_messages(thinking: bool) -> tuple[dict, dict]:
    suffix = "_thinking" if thinking else ""
    system_msg = {
        "role": "system",
        "content": (PROMPTS_DIR / f"body_tracking{suffix}.txt").read_text().strip()
    }
    prompt_msg = {
        "type": "text",
        "text": (PROMPTS_DIR / f"clip_prompt{suffix}.txt").read_text().strip()
    }
    return system_msg, prompt_msg

COMPLETION_KWARGS = dict(
    temperature=0.7, top_p=0.8, presence_penalty=1.5,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
)


def make_completion_kwargs(thinking: bool) -> dict:
    kwargs = COMPLETION_KWARGS.copy()
    kwargs["extra_body"] = {**kwargs["extra_body"], "chat_template_kwargs": {"enable_thinking": thinking}}
    kwargs["max_tokens"] = 8192 if thinking else 512
    return kwargs


def get_clip_groups(clips_dir):
    """Return sorted list of (clip_idx, tile_row, tile_col, raw, overlay, black_bg) tuples."""
    clips_dir = Path(clips_dir)
    overlay_dir = clips_dir / "overlay"
    raw_dir     = clips_dir / "raw"
    black_bg_dir = clips_dir / "black_bg"

    groups = []
    for tile_path in sorted(overlay_dir.glob("clip_*_tile_*.mp4")):
        parts = tile_path.stem.split("_")  # clip, N, tile, R, C
        clip_idx, row, col = int(parts[1]), int(parts[3]), int(parts[4])

        raw      = raw_dir     / f"clip_{clip_idx}_tile_{row}_{col}.mp4"
        black_bg = black_bg_dir / f"clip_{clip_idx}_tile_{row}_{col}.mp4"

        if not raw.exists() or not black_bg.exists():
            continue

        groups.append((clip_idx, row, col, str(raw), str(tile_path), str(black_bg)))

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
    abs_path = Path(path).resolve()
    return f"file://{abs_path}"


def load_few_shot(thinking):
    """Load few-shot examples from assets/examples/.

    Returns list of (urls, response_text) in order: error_pos, error_orient, correct.
    Returns None if any required file is missing.
    """
    if not EXAMPLES_DIR.exists():
        return None
    suffix = "_thinking" if thinking else ""
    examples = []
    for name in ("error_pos", "error_orient", "correct"):
        d = EXAMPLES_DIR / name
        resp_file = d / f"response{suffix}.txt"
        if not d.exists() or not resp_file.exists():
            print(f"Warning: few-shot example missing: {d}")
            return None
        urls = [
            video_url(str(d / "raw.mp4")),
            video_url(str(d / "overlay.mp4")),
            video_url(str(d / "black_bg.mp4")),
        ]
        examples.append((urls, resp_file.read_text().strip()))
    return examples


def _make_user_turn(urls, prompt_msg):
    u, o, b = urls
    return {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": u}},
            {"type": "video_url", "video_url": {"url": o}},
            {"type": "video_url", "video_url": {"url": b}},
            prompt_msg,
        ],
    }


async def process_clip(client, clip_idx, row, col, raw, overlay, black_bg, model, sem, completion_kwargs, system_msg, prompt_msg, few_shot=None):
    async with sem:
        raw_url      = video_url(raw)
        overlay_url  = video_url(overlay)
        black_bg_url = video_url(black_bg)

        print(f"Processing overlay clip: {overlay_url}")

        few_shot_turns = []
        if few_shot:
            for urls, resp in few_shot:
                few_shot_turns.append(_make_user_turn(urls, prompt_msg))
                few_shot_turns.append({"role": "assistant", "content": resp})

        messages = [
            system_msg,
            *few_shot_turns,
            _make_user_turn([raw_url, overlay_url, black_bg_url], prompt_msg),
        ]

        r = await client.chat.completions.create(
            model=model,
            messages=messages,
            **completion_kwargs,
        )

        print(r.choices[0].message.content)
        return clip_idx, row, col, r.choices[0].message.content


async def main():
    _load_inputs_env()

    parser = argparse.ArgumentParser(description="Run keypoint annotation quality control on video clips.")
    parser.add_argument("--port",        type=int, required=True,       help="vLLM server port")
    parser.add_argument("--model",       type=str, default=os.environ.get("MODEL"), help="Model path (default: $MODEL from inputs.env)")
    parser.add_argument("--clips-dir",   type=str, default=os.environ.get("CLIPS_DIR", "subclips"), help="Subclips directory (default: subclips or $CLIPS_DIR)")
    parser.add_argument("--output",      type=str, default=os.environ.get("VLM_RESULTS", "results.out"), help="Output file for results (default: results.out or $VLM_RESULTS)")
    parser.add_argument("--concurrency", type=int, default=4,           help="Max concurrent requests (default: 2)")
    parser.add_argument("--log",         type=str, default=None,        help="Optional log file (stdout is also mirrored here)")
    parser.add_argument("--thinking",   action="store_true", help="Enable Gemma thinking mode")
    parser.add_argument("--few-shot",   action="store_true", help="Prepend in-context examples from assets/examples/")
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set MODEL in inputs.env)")

    tee = _Tee(args.log) if args.log else None

    groups = get_clip_groups(args.clips_dir)
    if not groups:
        print(f"No clip groups found in {args.clips_dir}")
        return

    print("Getting clip duration...")
    clip_duration = get_clip_duration_seconds(groups[0][3])
    print(f"Clip duration: {clip_duration:.2f}s")

    print(f"model: {args.model} | port: {args.port} | concurrency: {args.concurrency}")
    print(f"Found {len(groups)} tile-clips, {clip_duration:.2f}s each")

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"
    )

    manifest_path = Path(args.clips_dir) / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.monotonic()
    print("Submitting tasks...")
    completion_kwargs = make_completion_kwargs(args.thinking)
    system_msg, prompt_msg = make_messages(args.thinking)
    few_shot = load_few_shot(args.thinking) if args.few_shot else None
    if few_shot:
        print(f"Few-shot: {len(few_shot)} examples loaded from {EXAMPLES_DIR}")
    tasks = [process_clip(client, *g, args.model, sem, completion_kwargs, system_msg, prompt_msg, few_shot) for g in groups]
    results = await asyncio.gather(*tasks)
    print("All tasks complete.")

    results = sorted(results, key=lambda r: (r[0], r[1], r[2]))

    lines = []
    for clip_idx, row, col, result in results:
        manifest_key = f"overlay/clip_{clip_idx}_tile_{row}_{col}.mp4"
        meta = manifest.get(manifest_key, {})

        start_frame = meta.get("start_frame", clip_idx * int(clip_duration * 150))
        end_frame   = meta.get("end_frame",   start_frame + int(clip_duration * 150) - 1)
        t_start     = meta.get("start_time_s", clip_idx * clip_duration)
        t_end       = meta.get("end_time_s",   t_start + clip_duration)

        words = [w.rstrip(":.").upper() for w in result.strip().split()]
        verdict_words = [w for w in words if w in ("CORRECT", "ERROR")]
        verdict = verdict_words[-1] if args.thinking else (verdict_words[0] if verdict_words else "UNKNOWN")
        if not verdict_words:
            verdict = "UNKNOWN"

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
    if tee:
        tee.close()


asyncio.run(main())