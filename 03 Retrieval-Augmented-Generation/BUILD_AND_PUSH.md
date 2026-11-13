# Build and Push Docker Image

**Prerequisites:** Docker Desktop installed and running, a Docker Hub account, and the Docker Hub repository you want to publish to.

Open PowerShell and run:

```powershell
cd "D:\Projects\AMD AI League\03 Retrieval-Augmented-Generation"
```

```powershell
.\build-and-push.ps1
```

If the script does not run because PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build-and-push.ps1
```

The script will:

1. Ask for your Docker Hub repository name and image tag.
2. Verify Docker and the required project files.
3. Sign in to Docker Hub.
4. Create an isolated downloader environment if needed.
5. Download or verify **Qwen2.5-VL-3B-Instruct**.
6. Download or verify **Qwen3-Embedding-0.6B**.
7. Build the Docker image using the project's Dockerfile.
8. Push the image to Docker Hub.

Example inputs:

```text
yourusername/amd-rag
v1
```

Replace these example values with your own Docker Hub repository and desired tag.

The resulting image will be published as:

```text
yourusername/amd-rag:v1
```
