from pathlib import Path

current_dir = Path.cwd()
target_dir_test = current_dir.parent / "test"
test_dir = target_dir_test / "encoding_tests"
test_dir.mkdir(exist_ok=True)

sample_text = "Hello World! | Café | €"

# Create a Windows-1252 encoded file
path = test_dir / "windows1252.txt"
path.write_text(sample_text, encoding="cp1252")

print(f"Created: {path}")
