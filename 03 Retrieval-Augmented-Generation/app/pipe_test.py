import sys
from PIL import Image

from pathlib import Path
current_dir = Path.cwd()
target_dir_vlm = current_dir.parent / "models" / "qwen2.5-vl-3b"
target_dir_embedding = current_dir.parent / "models" / "qwen3-embedding-0.6b"
target_dir_test = current_dir.parent / "test"
candy_image_path = target_dir_test / "candy.JPG"
print(f"target_dir_vlm: {target_dir_vlm}")
print(f"target_dir_embedding: {target_dir_embedding}")
print(f"candy_image_path: {candy_image_path}")
print(f"candy_image_path.exists(): {candy_image_path.exists()}")
print("=" * 10)

# with Image.open(candy_image_path) as img:
#     print(img.format, img.size, img.mode)

from transformers import pipeline, Qwen2_5_VLForConditionalGeneration

pipe = pipeline("image-text-to-text", model=str(target_dir_vlm))
model = pipe.model

print(type(model))
print(type(model) is Qwen2_5_VLForConditionalGeneration)
print("=" * 10)

with Image.open(candy_image_path) as img:
    image = img.convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
result = pipe(text=messages)
print(f"result: {result}")
response = result[0]["generated_text"][-1]["content"]
print(f"response: {response}")