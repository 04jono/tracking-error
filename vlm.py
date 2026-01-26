from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-30B-A3B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": '''You are an experienced data annotator. The dataset is a top down video of flies. The keypoints and orientation of the flies have already been annotated and overlaid on top of the original video. The true flies can be seen underneath the annotations. You have been tasked with evaluating whether there is any error in the annotation within the video snippet you are given.
            
            Suspicious frames and flies are any for which:
            * Dropped tracks, a trajectory suddenly begins or ends but the fly is still on screen.
            * The keypoints and orientation make sudden non-smooth changes between frames, i.e. the fly makes large jumps.
            * The keypoints or orientation are not properly overlaid over the fly.
            
            '''},
            {
                "type": "video",
                "video": "tracking_overlay_10s.mp4",
            },
            {"type": "text", "text": "The video is an example of a perfectly annotated video snippet. Notice how the movements are smooth and do not jump, and the annotation is directly over the true fly."},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "tracking_overlay_10s.mp4",
            },
            {"type": "text", "text": "Are there any errors in the annotation? If yes, reply YES and the potential error. If no, reply NO."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
