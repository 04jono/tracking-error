import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:38343/v1",
    api_key=""
)

video_paths = [
    "/share/path/to/video1.mp4",
    "/share/path/to/video2.mp4",
    "/share/path/to/video3.mp4",
    "/share/path/to/video4.mp4",
    "/share/path/to/video5.mp4",
    "/share/path/to/video6.mp4",
    "/share/path/to/video7.mp4",
    "/share/path/to/video8.mp4",
    "/share/path/to/video9.mp4",
    "/share/path/to/video10.mp4",
]

ICL_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an expert in animal pose estimation and keypoint annotation quality control. "
            "You will be shown videos of animals with keypoint annotations overlaid. "
            "Your job is to determine whether the annotations are correct or contain errors. "
            "Errors include: keypoints placed on wrong body parts, swapped left/right sides, "
            "keypoints drifting off the animal, missing keypoints, or annotations that don't "
            "track the animal's joints consistently across frames. "
            "Consider each video snippet independent of each other, there may be errors in one but not errors in the other."
            "Always respond with CORRECT or ERROR at the beginning of your response, followed by a brief reason."
        )
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "file:///share/path/to/correct_example.mp4"},
            },
            {
                "type": "text",
                "text": "Do the keypoint annotations correctly track the animal's joints throughout this video?"
            }
        ]
    },
    {
        "role": "assistant",
        "content": (
            "CORRECT. All keypoints are accurately placed on the corresponding joints and consistently "
            "track the animal's body parts throughout the video with no drift or misplacement."
        )
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "file:///share/path/to/error_example.mp4"},
            },
            {
                "type": "text",
                "text": "Do the keypoint annotations correctly track the animal's joints throughout this video?"
            }
        ]
    },
    {
        "role": "assistant",
        "content": (
            "ERROR. The left foreleg keypoint is placed on the right foreleg, indicating a "
            "left/right swap in the annotation."
        )
    },
]

async def process_video(video_path):
    messages = ICL_MESSAGES + [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path}"},
                },
                {
                    "type": "text",
                    "text": "Do the keypoint annotations correctly track the animal's joints throughout this video?"
                }
            ]
        }
    ]

    response = await client.chat.completions.create(
        model="/share/j_sun/jqc3/Qwen3.5-27B-FP8",
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
    tasks = [process_video(path) for path in video_paths]
    results = await asyncio.gather(*tasks)
    for video_path, result in results:
        print(f"\n=== {video_path} ===")
        print(result)

asyncio.run(main())