import json
import socket

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import HOST, MODEL_DIR, PORT, resolve_input_path

PROMPT = (
    "Transcribe only the registration number or sign text visible in this image. "
    "Return only the characters in reading order, without explanations or labels."
)


def load_model():
    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"Model configuration not found: {MODEL_DIR}")

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    model.eval()
    return processor, model


def read_image(image_path, processor, model):
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64)

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]

    return processor.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def handle_connection(connection, processor, model):
    with connection:
        with connection.makefile("rwb") as stream:
            try:
                line = stream.readline()
                if not line:
                    raise ValueError("Empty worker request.")

                request = json.loads(line.decode("utf-8"))
                image_path = resolve_input_path(request["image_path"])

                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                text = read_image(image_path, processor, model)
                response = {"text": text}

            except Exception as exc:
                response = {"error": str(exc)}

            stream.write(
                (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
            )
            stream.flush()


def main():
    print(f"Loading model from {MODEL_DIR}", flush=True)
    processor, model = load_model()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        print(f"OCR worker ready at {HOST}:{PORT}", flush=True)

        while True:
            connection, _ = server.accept()
            handle_connection(connection, processor, model)


if __name__ == "__main__":
    main()