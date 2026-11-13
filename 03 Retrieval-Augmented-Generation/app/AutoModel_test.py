# pip install accelerate
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

from transformers import AutoProcessor, AutoModelForMultimodalLM, Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained(str(target_dir_vlm))
model = AutoModelForMultimodalLM.from_pretrained(str(target_dir_vlm), device_map="auto")

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
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))