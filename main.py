import argparse
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets" / "videos"

DEFAULT_MODEL = "/share/j_sun/jqc3/Qwen3.5-27B-FP8"
VIDEO_PATHS = [
    "/home/jqc3/tracking-error/clips/clip_0.mp4",
    "/home/jqc3/tracking-error/clips/clip_1.mp4",
    "/home/jqc3/tracking-error/clips/clip_2.mp4",
    "/home/jqc3/tracking-error/clips/clip_3.mp4",
    "/home/jqc3/tracking-error/clips/clip_4.mp4",
    "/home/jqc3/tracking-error/clips/clip_5.mp4",
    "/home/jqc3/tracking-error/clips/clip_6.mp4",
    "/home/jqc3/tracking-error/clips/clip_7.mp4",
    "/home/jqc3/tracking-error/clips/clip_8.mp4",
    "/home/jqc3/tracking-error/clips/clip_9.mp4",
]

# ICL example videos
ICL_CORRECT_VIDEO = str((ASSETS_DIR / "cx_fixed_30fps_0000-0005.mp4").absolute())
ICL_ERROR_VIDEO = str((ASSETS_DIR / "cx_fixed_30fps_2040-2045.mp4").absolute())

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an expert in animal pose estimation and keypoint annotation quality control. "
        "You will be shown videos of animals with keypoint annotations overlaid. "
        "Your job is to determine whether the annotations are correct or contain errors. "
        "Errors include: keypoints placed on wrong body parts, swapped left/right sides, "
        "keypoints drifting off the animal, missing keypoints, or annotations that don't "
        "track the animal's joints consistently across frames. "
        "Consider each video snippet independent of each other, there may be errors in one but not errors in the other."
        "Suspicious frames and flies are any for which:"
        "* Dropped tracks, a trajectory suddenly begins or ends (switches colors) but the fly is still on screen."
        "* The keypoints and orientation make sudden non-smooth changes between frames, i.e. the fly makes large jumps. There could be jumps due to the frame rate, if the annotation makes jumps check for false positives based on if the annotation still overlays the fly."
        "* The keypoints or orientation are not properly overlaid over the fly."
        "IMPORTANT: Always respond with CORRECT or ERROR at the beginning of your response, followed by a brief reason."
    )
}


def build_icl_messages(use_icl: bool) -> list:
    messages = [SYSTEM_MESSAGE]
    if not use_icl:
        return messages

    if Path(ICL_CORRECT_VIDEO).exists():
        messages.extend([
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"file://{ICL_CORRECT_VIDEO}"},
                    },
                    {
                        "type": "text",
                        "text": "Do the keypoint annotations correctly track the animal's keypoints throughout this video clip?"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": (
                    "CORRECT. All keypoints of all flies are accurately placed on the corresponding joints and consistently "
                    "track the animal's body parts throughout the video with no drift, swap, or misplacement."
                )
            },
        ])

    if Path(ICL_ERROR_VIDEO).exists():
        messages.extend([
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"file://{ICL_ERROR_VIDEO}"},
                    },
                    {
                        "type": "text",
                        "text": "Do the keypoint annotations correctly track the animal's keypoints throughout this video clip?"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": (
                    "ERROR. The fly with green keypoints in the bottom left corner has annotations that jump around the body and do not track the ends of one of its legs"
                )
            },
        ])

    return messages


async def process_video(client, video_path, icl_messages, model):
    messages = icl_messages + [
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
    parser.add_argument("--port", type=int, required=True, help="vLLM server port")
    parser.add_argument("--no-icl", action="store_true", help="Disable in-context learning examples")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model path (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key=""
    )

    use_icl = not args.no_icl
    icl_messages = build_icl_messages(use_icl)

    print(f"ICL: {'enabled' if use_icl else 'disabled'} | model: {args.model} | port: {args.port}")

    tasks = [process_video(client, path, icl_messages, args.model) for path in VIDEO_PATHS]
    results = await asyncio.gather(*tasks)
    for video_path, result in results:
        print(f"\n=== {video_path} ===")
        print(result)


asyncio.run(main())
