import argparse
import json
import socket
import time

from config import HOST, OUTPUT_DIR, PORT, resolve_input_path

WORKER_CONNECT_TIMEOUT = 600


def request_ocr(image_path):
    deadline = time.monotonic() + WORKER_CONNECT_TIMEOUT
    last_error = None

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=2) as client:
                client.settimeout(35)

                request = {"image_path": str(image_path)}
                client.sendall(
                    (json.dumps(request) + "\n").encode("utf-8")
                )

                with client.makefile("rb") as stream:
                    line = stream.readline()

                if not line:
                    raise RuntimeError("Worker closed the connection without a response.")

                response = json.loads(line.decode("utf-8"))

                if "error" in response:
                    raise RuntimeError(response["error"])

                return response["text"]

        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)

    raise RuntimeError(
        f"OCR worker at {HOST}:{PORT} did not become ready: {last_error}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", required=True)
    args = parser.parse_args()

    image_path = resolve_input_path(args.input_image)

    if not image_path.is_file():
        parser.error(f"Input image not found: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_output.json"

    text = " ".join(request_ocr(image_path).split())

    output_path.write_text(
        json.dumps({"text": text}, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Result {text} written to {output_path}", flush=True)


if __name__ == "__main__":
    main()