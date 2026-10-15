# AMD OCR Challenge

An image-to-text application built for the [AMD AI League](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge).

Challenge Details: Refer to the [AMD AI League – Match 1 & 2 Challenge Brief](./AMD%20AI%20League%20–%20Match%201%20&%202%20Challenge%20Brief.pdf)

The application uses Qwen2.5-VL-3B-Instruct, a vision-language model, to recognize text in images such as license plates and road signs. It runs on AMD ROCm and returns the recognized text as JSON.

## Demo

**Video demonstration:**


https://github.com/user-attachments/assets/8dd42872-8bbf-4565-905e-6a9983bb8d0e



## Sample Results

The repository includes ten sample images in `test/` and their corresponding JSON results in `app/output/`.

| Sample image | Expected recognized text |
|---|---|
| `image_01.png` | `7ABC123` |
| `image_02.png` | `京 A·12345` |
| `image_06.png` | `STOP` |
| `image_08.png` | `SPEED LIMIT 65` |
| `image_09.png` | `ROAD WORK AHEAD` |

Check `test/` for the input images and `app/output/` for the actual generated results.

## Run Locally

**Prerequisites:** Python 3, pip, and a compatible PyTorch installation for your hardware (AMD ROCm or NVIDIA CUDA for GPU inference). Ensure the required GPU drivers and software are installed before proceeding.

**1. Install dependencies**

Install the project dependencies using:

```bash
python3 -m pip install -r app/requirements.txt
python3 -m pip install huggingface_hub
```

**2. Download the model**

From the project root, run:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct', local_dir='models/qwen2.5-vl-3b')"
```

**3. Start the OCR worker**

In the first terminal:

```bash
python3 app/worker.py
```

Wait until the worker reports that it is ready.

**4. Process an image**

In a second terminal:

```bash
python3 app/app.py --input-image ./test/image_01.png
```

The recognized text is saved as JSON in `app/output/image_01_output.json`. Change the input path to process another image.

## Docker Image

For instructions to download the model automatically, build the Docker image, and publish it to Docker Hub, see [BUILD_AND_PUSH.md](BUILD_AND_PUSH.md).
