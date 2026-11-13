# AMD Retrieval-Augmented Generation Challenge

Challenge Details: Refer to the [AMD AI League – Match 3 Challenge Brief](./AMD%20AI%20League%20%E2%80%93%20Match%203%20Challenge%20Brief%20-%20Copy.pdf)

This project implements a Retrieval-Augmented Generation (RAG) application for the [AMD AI League](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge) challenge. It scans a corpus of mixed content types, indexes the data with an embedding model, and answers questions using a local Qwen-based pipeline. The application is designed for AMD ROCm-based GPU inference and is packaged for Docker-based deployment.

You can also refer to `app/output/` to view the generated output or testing results.

## Demo

**Video demonstration:**

https://github.com/user-attachments/assets/bcd6ab1d-2d0f-4a24-bd48-8e78bad0bcf6

## What this project does

The app:

- walks a corpus directory containing PDFs, Word documents, spreadsheets, CSV/text files, logs, source code, and images
- extracts text and image-based content from those files
- chunks and embeds the content using the Qwen3 embedding model
- stores the indexed data in a local Chroma vector database
- uses the Qwen2.5-VL-3B-Instruct model for image-aware extraction when needed
- answers user queries and writes the result as structured JSON with the final answer, source citations, and confidence score

## Project structure

- `app/app.py` — main challenge entry point used to index a corpus and answer queries
- `app/worker.py` — background worker that loads the models and serves indexing/query requests
- `app/config.py` — local host and port configuration for the worker process
- `models/` — downloaded model weights for the VLM and embedding model
- `app/output/` — generated outputs for example questions
- `Dockerfile` — container image definition for AMD GPU inference
- `build-and-push.ps1` — script to verify Docker requirements, download models, build the image, and push it to Docker Hub

## Prerequisites

Before running locally or building the container, make sure you have:

- Python 3 installed
- pip available
- a compatible GPU setup (AMD ROCm or NVIDIA CUDA depending on your environment)
- Docker Desktop running if you plan to build or publish the image
- a Docker Hub account and repository name if you want to push the image

## Run locally

### 1. Install dependencies

From the project root:

```powershell
cd "D:\Projects\AMD AI League\03 Retrieval-Augmented-Generation"
python -m pip install -r app/requirements.txt
python -m pip install huggingface_hub
```

### 2. Download the model files

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct', local_dir='models/qwen2.5-vl-3b')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-Embedding-0.6B', local_dir='models/qwen3-embedding-0.6b')"
```

### 3. Start the worker

In one terminal:

```powershell
python app/worker.py
```

### 4. Build the index for your corpus

Place your corpus in a folder such as `corpus/` and run:

```powershell
python app/app.py --index .\corpus
```

This step prepares the vector database and indexing artifacts required for retrieval.

### 5. Ask a question

```powershell
python app/app.py --corpus .\corpus --query-id query_01 --query "What is the maximum junction temperature?"
```

The output is written to:

```text
app/output/query_01_output.json
```

Expected output structure:

```json
{
  "answer": "94",
  "citations": ["specs/tq40_datasheet_r2.pdf"],
  "confidence": 0.9
}
```

## Docker build and push

The repository includes a PowerShell script to automate the Docker build and publishing flow.

Open PowerShell and run:

```powershell
cd "D:\Projects\AMD AI League\03 Retrieval-Augmented-Generation"
.\build-and-push.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build-and-push.ps1
```

The script will:

1. ask for your Docker Hub repository name and image tag
2. validate Docker and required project files
3. sign in to Docker Hub
4. create a local downloader environment if needed
5. download or verify the Qwen models
6. build the Docker image using the project `Dockerfile`
7. push the image to Docker Hub

Example inputs:

```text
yourusername/amd-rag
v1
```

The resulting image will be published as:

```text
yourusername/amd-rag:v1
```

For a shorter build-and-push guide, see `BUILD_AND_PUSH.md`.

## Notes

- The challenge harness expects a structured JSON result with `answer`, `citations`, and `confidence`.
- The app is designed to continue processing even when some files in the corpus are unreadable or malformed.
- Make sure the required model directories exist before running the worker locally.
