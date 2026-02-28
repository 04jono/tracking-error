from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
import torch

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")

video_list = [
    "clips/clip_0.mp4",
    "clips/clip_1.mp4",
    "clips/clip_2.mp4",
    "clips/clip_3.mp4",
    "clips/clip_4.mp4",
    "clips/clip_5.mp4",
    "clips/clip_6.mp4",
    "clips/clip_7.mp4",
    "clips/clip_8.mp4",
    "clips/clip_9.mp4",
]

# System message (same for all videos)
system_message = {
    "role": "system",
    "content": [
        {"type": "text", "text": '''You are an experienced data annotator. 
         The dataset is a top down video of flies. 
         The keypoints and orientation of the flies have already been annotated and overlaid on top of the original video. 
         The true flies can be seen underneath the annotations. You have been tasked with evaluating whether there is any error in the annotation within the video snippet you are given.
         The fly is always the ground truth and always correct, only consider if the red keypoint mask matches the ground truth.
        
        Suspicious frames and flies are any for which:
        * Dropped tracks, a trajectory suddenly begins or ends but the fly is still on screen.
        * The keypoints and orientation make sudden non-smooth changes between frames, i.e. the fly makes large jumps. There could be jumps due to the frame rate, if the annotation makes jumps check for false positives based on if the annotation still overlays the fly.
        * The keypoints or orientation are not properly overlaid over the fly.
        
        Consider each video snippet independent of each other, there may be errors in one but not errors in the other.
        
        '''},
    ],
}

# Process each video
for video_path in video_list:
    print(f"\nProcessing video: {video_path}")
    print("-" * 50)
    
    messages = [
        system_message,
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
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
    
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print(f"Result: {output_text[0]}")