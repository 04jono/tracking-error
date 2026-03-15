import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM

# -----------------------
# Paths
# -----------------------
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets" / "videos"

videos = [
    str((ASSETS_DIR / "cx_fixed_30fps_0000-0005.avi").absolute()),
    str((ASSETS_DIR / "cx_fixed_30fps_2040-2045.avi").absolute()),
]

MODEL_PATH = "/share/j_sun/jqc3/Qwen3.5-27B-FP8"

# -----------------------
# Load model + processor
# -----------------------
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

SYSTEM_PROMPT = (
    "You are an expert in animal pose estimation and keypoint annotation quality control. "
    "You will be shown videos of animals with keypoint annotations overlaid. "
    "Your job is to determine whether the annotations are correct or contain errors. "
    "Errors include: keypoints placed on wrong body parts, swapped left/right sides, "
    "keypoints drifting off the animal, missing keypoints, or inconsistent tracking. "
    "IMPORTANT: Always respond with CORRECT or ERROR at the beginning of your response, "
    "followed by a brief reason."
)

QUESTION = "Do the keypoint annotations correctly track the animal's keypoints throughout this video clip?"

# -----------------------
# Serial processing
# -----------------------
def process_video(video_path):
    print(f"\nProcessing {video_path}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": QUESTION},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # deterministic
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(generated, skip_special_tokens=True)

    print(response)


def main():
    for video in videos:  # SERIAL
        process_video(video)


if __name__ == "__main__":
    main()