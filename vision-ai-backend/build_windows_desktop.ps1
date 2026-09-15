[CmdletBinding()]
param(
  [ValidateSet('cpu', 'cuda', 'auto')]
  [string]$Acceleration = 'auto',
  [switch]$SkipPyInstaller
)

# Produces a self-contained 64-bit Windows NSIS installer. Run this on a
# 64-bit Windows machine: PyInstaller and PyTorch must be built natively.
$ErrorActionPreference = 'Stop'
$BackendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path (Split-Path -Parent $BackendDir) 'vision-ai-frontend'
$ReleaseVenv = if (Test-Path (Join-Path $BackendDir '.venv')) {
  Join-Path $BackendDir '.venv'
} else {
  Join-Path $BackendDir '.venv-release'
}
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
  $PythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $PythonCmd) { throw 'Python was not found in PATH. Please install Python 3.10+ and add it to PATH.' }
$Python = $PythonCmd.Source

if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows host is required.' }
if (-not (Test-Path (Join-Path $FrontendDir 'package.json'))) { throw 'vision-ai-frontend must be beside vision-ai-backend.' }

if (-not (Test-Path $ReleaseVenv)) {
  & $Python -m venv --system-site-packages $ReleaseVenv
}
$VenvPython = Join-Path $ReleaseVenv 'Scripts\python.exe'

# Auto-detect acceleration if 'auto' is specified
if ($Acceleration -eq 'auto') {
  $NvidiaGpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" }
  if ($null -ne $NvidiaGpus -and @($NvidiaGpus).Count -gt 0) {
    $Acceleration = 'cuda'
    Write-Host "Auto-detected NVIDIA GPU: $(($NvidiaGpus | Select-Object -First 1).Name); targeting CUDA acceleration."
  } else {
    $Acceleration = 'cpu'
    Write-Host "No NVIDIA GPU detected on build host; targeting CPU acceleration."
  }
}

# Verify CUDA PyTorch and TorchVision
$HasCuda = & $VenvPython -c "import torch; print(torch.cuda.is_available() or 'cu' in torch.__version__)" 2>$null
$HasTorchVision = & $VenvPython -c "import torchvision; import importlib.metadata; _ = importlib.metadata.version('torchvision'); print('True')" 2>$null
Write-Host "CUDA PyTorch status in environment: $HasCuda | TorchVision metadata status: $HasTorchVision"
if (($Acceleration -eq 'cuda' -and $HasCuda -ne 'True') -or ($HasTorchVision -ne 'True' -and $Acceleration -eq 'cuda')) {
  $CudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { 'https://download.pytorch.org/whl/cu121' }
  Write-Host "Installing CUDA PyTorch & TorchVision from $CudaIndex..."
  & $VenvPython -m pip install --index-url $CudaIndex torch torchvision
  if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA PyTorch from $CudaIndex." }
} elseif (($Acceleration -eq 'cpu' -and $HasCuda -eq 'True') -or ($HasTorchVision -ne 'True' -and $Acceleration -eq 'cpu')) {
  Write-Host 'Installing CPU-only PyTorch & TorchVision for CPU bundle...'
  & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  if ($LASTEXITCODE -ne 0) { throw "Could not install CPU PyTorch from PyTorch CPU index." }
}

$DuckAnalyzerWheel = Get-ChildItem -Path (Join-Path $BackendDir 'app\ml') -Filter 'duck_analyzer-*.whl' -Recurse |
  Sort-Object Name -Descending | Select-Object -First 1
if (-not $DuckAnalyzerWheel) { throw 'The bundled duck_analyzer wheel is missing.' }
& $VenvPython -m pip install $DuckAnalyzerWheel.FullName
if (-not $SkipPyInstaller -or -not (Test-Path (Join-Path $BackendDir 'dist\backend\backend.exe'))) {
  Push-Location $BackendDir
  try {
    Remove-Item -LiteralPath (Join-Path $BackendDir 'build') -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $BackendDir 'dist') -Recurse -Force -ErrorAction SilentlyContinue

    $PyInstallerArgs = @(
      '--noconfirm', '--clean', '--onedir', '--name', 'backend', 'run.py',
      '--add-data', 'app/ml/model;app/ml/model',
      '--add-data', 'app/ml/config;app/ml/config',
      '--add-data', 'app/ml/torch_hub;app/ml/torch_hub',
      '--add-data', 'alembic;alembic'
    )
    $PyInstallerArgs += @(
      '--copy-metadata', 'torchvision',
      '--copy-metadata', 'ultralytics',
      '--collect-all', 'app', '--collect-all', 'fastapi', '--collect-all', 'starlette', '--collect-all', 'uvicorn',
      '--collect-all', 'sqlalchemy', '--collect-all', 'cv2', '--collect-all', 'torch', '--collect-all', 'torchvision',
      '--collect-all', 'ultralytics', '--collect-all', 'segmentation_models_pytorch', '--collect-all', 'depthai',
      '--collect-all', 'av', '--collect-all', 'duck_analyzer', '--collect-all', 'mediapipe', '--collect-all', 'matplotlib',
      '--collect-all', 'scipy', '--collect-all', 'lap', '--collect-all', 'imageio_ffmpeg'
    )
    & $VenvPython -m PyInstaller @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
  } finally { Pop-Location }
} else {
  Write-Host 'Reusing existing dist\backend...'
}

$IconIco = Join-Path $FrontendDir 'public\icon.ico'
if (-not (Test-Path $IconIco)) {
  & $VenvPython (Join-Path $FrontendDir 'public\generate_icons.py')
}

$BackendDist = Join-Path $BackendDir 'dist\backend'
if (Test-Path $BackendDist) {
  Write-Host 'Optimizing dist\backend by removing non-runtime development files...'
  Get-ChildItem -Path $BackendDist -Recurse -Include *.lib, *.pdb, *.exp, *.a -File | Remove-Item -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\torch\include') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\torch\testing') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\torch\test') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\torch\bin') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\jaxlib') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\psycopg2_binary.libs') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\hf_xet') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\_polars_runtime_32') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\app\ml\output') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $BackendDist '_internal\app\unused') -Recurse -Force -ErrorAction SilentlyContinue
}

Push-Location $FrontendDir
try {
  Remove-Item -LiteralPath (Join-Path $FrontendDir 'dist_app\win-unpacked') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -Path (Join-Path $FrontendDir 'dist_app\*.nsis.7z*') -Force -ErrorAction SilentlyContinue
  Remove-Item -Path (Join-Path $FrontendDir 'dist_app\Vision-Monitor*') -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -Path (Join-Path $FrontendDir 'dist_app\Vision-AI*') -Recurse -Force -ErrorAction SilentlyContinue

  if (-not (Test-Path (Join-Path $FrontendDir 'node_modules'))) {
    & npm.cmd ci --include=optional
    if ($LASTEXITCODE -ne 0) {
      Write-Host 'npm ci failed, falling back to npm install...'
      & npm.cmd install --include=optional
    }
  }
  & npm.cmd run package:win:x64
  if ($LASTEXITCODE -ne 0) { throw 'electron-builder NSIS packaging failed.' }
} finally { Pop-Location }

Write-Host "Installer ready in $FrontendDir\dist_app"
