from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-vl-3b"
OUTPUT_DIR = APP_DIR / "output"

HOST = "127.0.0.1"
PORT = 45831


def resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()
