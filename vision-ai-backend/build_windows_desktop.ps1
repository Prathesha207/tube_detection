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
  & $Python -m venv $ReleaseVenv
}
$VenvPython = Join-Path $ReleaseVenv 'Scripts\python.exe'

# Auto-detect acceleration if 'auto' is specified
if ($Acceleration -eq 'auto') {
  $HasNvidia = $false
  $ComputeCap = 0.0

  # 1. Primary detection: nvidia-smi (reliable for Optimus laptops and all NVIDIA drivers)
  $SmiOutput = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
  if ($LASTEXITCODE -eq 0 -and $null -ne $SmiOutput) {
    $HasNvidia = $true
    $ComputeCaps = $SmiOutput | ForEach-Object {
        $parts = $_ -split ','
        if ($parts.Count -ge 2) { [float]$parts[1].Trim() } else { 0.0 }
    }
    if ($ComputeCaps) { $ComputeCap = ($ComputeCaps | Measure-Object -Maximum).Maximum }
    Write-Host "nvidia-smi detected NVIDIA GPU(s) with max Compute Capability: $ComputeCap"
  } else {
    # 2. Fallback detection: WMI
    $NvidiaGpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "NVIDIA" }
    if ($null -ne $NvidiaGpus -and @($NvidiaGpus).Count -gt 0) {
      $HasNvidia = $true
      Write-Host "WMI detected NVIDIA GPU: $(($NvidiaGpus | Select-Object -First 1).Name)"
    }
  }

  if ($HasNvidia) {
    $Acceleration = 'cuda'
    
    # 3. Target appropriate CUDA wheels based on Compute Capability
    # sm_120 (12.0) = Blackwell (RTX 50-series) -> cu126+
    # sm_89 (8.9) = Ada Lovelace (RTX 40-series) -> cu124
    if ($ComputeCap -ge 12.0) {
      $TargetCuda = "cu126"
      $CudaIndex = "https://download.pytorch.org/whl/cu126"
      Write-Host "Targeting CUDA 12.6+ (cu126) for Blackwell (RTX 50-series) support."
    } elseif ($ComputeCap -ge 8.9) {
      $TargetCuda = "cu124"
      $CudaIndex = "https://download.pytorch.org/whl/cu124"
      Write-Host "Targeting CUDA 12.4 (cu124) for Ada/Ampere optimization."
    } else {
      $TargetCuda = "cu121"
      $CudaIndex = "https://download.pytorch.org/whl/cu121"
      Write-Host "Targeting CUDA 12.1 (cu121) for broad compatibility."
    }
  } else {
    $Acceleration = 'cpu'
    $TargetCuda = "cpu"
    Write-Host "No NVIDIA GPU detected on build host; targeting CPU acceleration."
  }
} else {
  $TargetCuda = if ($Acceleration -eq 'cuda') { "cu121" } else { "cpu" }
  $CudaIndex = if ($Acceleration -eq 'cuda') { "https://download.pytorch.org/whl/cu121" } else { "https://download.pytorch.org/whl/cpu" }
}

if ($env:PYTORCH_CUDA_INDEX) { $CudaIndex = $env:PYTORCH_CUDA_INDEX }

# Verify PyTorch and TorchVision status safely without triggering NativeCommandError on stderr
$CurrentTorchVer = try {
  & $VenvPython -c "try: import torch; print(torch.__version__)`nexcept Exception: pass" 2>$null
} catch { $null }

$HasTorchVision = try {
  & $VenvPython -c "try:`n    import torchvision, importlib.metadata`n    _ = importlib.metadata.version('torchvision')`n    print('True')`nexcept Exception:`n    pass" 2>$null
} catch { $null }

Write-Host "Installed PyTorch version: $CurrentTorchVer | TorchVision status: $HasTorchVision"

if ($Acceleration -eq 'cuda') {
  # Force reinstall if the installed PyTorch doesn't match our targeted CUDA wheel (or is missing)
  if ($CurrentTorchVer -notmatch $TargetCuda -or $HasTorchVision -ne 'True') {
    Write-Host "Installing/Updating CUDA PyTorch & TorchVision from $CudaIndex..."
    & $VenvPython -m pip install --index-url $CudaIndex torch torchvision --upgrade --force-reinstall
    if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA PyTorch from $CudaIndex." }
  } else {
    Write-Host "Correct CUDA PyTorch ($TargetCuda) is already installed."
  }
} elseif ($Acceleration -eq 'cpu') {
  if ($CurrentTorchVer -match 'cu' -or -not $CurrentTorchVer -or $HasTorchVision -ne 'True') {
    Write-Host 'Installing CPU-only PyTorch & TorchVision for CPU bundle...'
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade --force-reinstall
    if ($LASTEXITCODE -ne 0) { throw "Could not install CPU PyTorch from PyTorch CPU index." }
  } else {
    Write-Host "Correct CPU PyTorch is already installed."
  }
}

$DuckAnalyzerWheel = Get-ChildItem -Path (Join-Path $BackendDir 'app\ml') -Filter 'duck_analyzer-*.whl' -Recurse |
  Sort-Object Name -Descending | Select-Object -First 1
if (-not $DuckAnalyzerWheel) { throw 'The bundled duck_analyzer wheel is missing.' }
$PipExtraArgs = if ($Acceleration -eq 'cuda' -and $CudaIndex) { @('--prefer-binary', '--extra-index-url', $CudaIndex) } else { @('--prefer-binary') }
& $VenvPython -m pip install @PipExtraArgs $DuckAnalyzerWheel.FullName
if ($LASTEXITCODE -ne 0) { throw 'Failed to install duck_analyzer wheel.' }
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
      '--collect-all', 'av', '--collect-all', 'mediapipe',
      '--collect-all', 'scipy', '--collect-all', 'lap', '--collect-all', 'imageio_ffmpeg', '--collect-all', 'duck_analyzer'
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
