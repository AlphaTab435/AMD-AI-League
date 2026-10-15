$ErrorActionPreference = "Stop"

# Always work from the challenge directory.
Set-Location -LiteralPath $PSScriptRoot

$modelDir = Join-Path $PSScriptRoot "models\qwen2.5-vl-3b"
$configPath = Join-Path $modelDir "config.json"

# Keep the downloader environment inside this project.
$venvDir = Join-Path $PSScriptRoot ".venv-downloader"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

# Collect and validate inputs before doing any work.
$repository = Read-Host "Docker Hub repository (username/repository)"
$tag = Read-Host "Image tag (e.g. v1)"

if ($repository -cnotmatch '^[a-z0-9]+(?:[._-][a-z0-9]+)*/[a-z0-9]+(?:[._-][a-z0-9]+)*$') {
    throw "Invalid repository. Use lowercase username/repository."
}

if ($tag -cnotmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$') {
    throw "Invalid Docker image tag."
}

$image = "${repository}:${tag}"

# Verify required tools and project files.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI not found. Install Docker Desktop first."
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found."
}

if (-not (Test-Path (Join-Path $PSScriptRoot "Dockerfile"))) {
    throw "Dockerfile missing from $PSScriptRoot"
}

if (-not (Test-Path (Join-Path $PSScriptRoot ".dockerignore"))) {
    throw ".dockerignore is missing."
}

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not running. Start Docker Desktop."
}

# Authenticate before downloading the model.
docker login
if ($LASTEXITCODE -ne 0) {
    throw "Docker Hub login failed."
}

# Create an isolated downloader environment if needed.
if (-not (Test-Path $venvPython)) {
    Write-Host "`nCreating project-local downloader environment..."

    py -m venv $venvDir

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        throw "Could not create the downloader virtual environment."
    }
}

# Install the downloader only in its dedicated environment.
Write-Host "`nPreparing Hugging Face downloader..."

& $venvPython -m pip install --upgrade "huggingface_hub<2.0"

if ($LASTEXITCODE -ne 0) {
    throw "Could not install huggingface_hub."
}

# Download or resume the model snapshot.
New-Item -ItemType Directory -Force -Path $modelDir |
    Out-Null

Write-Host "`nDownloading/verifying Qwen2.5-VL-3B-Instruct..."
Write-Host "Destination: $modelDir"

$env:MODEL_DIR = $modelDir

try {
    & $venvPython -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct', local_dir=os.environ['MODEL_DIR'])"

    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed."
    }
}
finally {
    Remove-Item Env:MODEL_DIR -ErrorAction SilentlyContinue
}

# Verify essential model files exist.
if (-not (Test-Path $configPath)) {
    throw "Model config.json is missing."
}

$weights = @(
    Get-ChildItem -Path $modelDir -Filter "*.safetensors" -File
)

if ($weights.Count -eq 0) {
    throw "No model weight files were found."
}

# Build for the Linux AMD64 platform used by the target environment.
Write-Host "`nBuilding $image for linux/amd64..."

docker build --platform linux/amd64 -t $image .

if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed."
}

# Publish the image to Docker Hub.
Write-Host "`nPushing $image..."

docker push $image

if ($LASTEXITCODE -ne 0) {
    throw "Docker image push failed."
}

Write-Host "`nBuild and push completed: $image"