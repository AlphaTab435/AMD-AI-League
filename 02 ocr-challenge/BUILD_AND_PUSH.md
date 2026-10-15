# Build and Push Docker Image

**Prerequisites:** Docker Desktop installed and running, a Docker Hub account, and the Docker Hub repository you want to publish to.

Open PowerShell and run:

```powershell
cd "D:\Projects\AMD AI League\02 ocr-challenge"
```
```powershell
.\build-and-push.ps1
```

if above does not work run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build-and-push.ps1
```

The script will:

1. Download Qwen2.5-VL-3B-Instruct if it has not already been downloaded.
2. Ask for your Docker Hub repository name and image tag.
3. Sign in to Docker Hub.
4. Build the Docker image using the project's Dockerfile.
5. Push the image to Docker Hub.

Example inputs:

```text
yourusername/amd-ocr
v1
```

Replace these example values with your own Docker Hub repository and desired tag.

The resulting image will be published as `yourusername/amd-ocr:v1`.