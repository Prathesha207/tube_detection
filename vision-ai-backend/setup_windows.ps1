[CmdletBinding()]
param(
    [switch]$Clean
)

# ==============================================================================
# Vision Monitor - Windows Environment Setup Script
# Configures Python virtual environment, dependencies, CUDA PyTorch,
# tube inference dependencies, and Node.js frontend packages without errors.
# ==============================================================================
$ErrorActionPreference = 'Stop'

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "       Vision Monitor - Automated Setup (Windows)      " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Determine directories
$BackendDir = if ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    Get-Location
}
$ProjectRoot = Split-Path -Parent $BackendDir
$FrontendDir = Join-Path $ProjectRoot 'vision-ai-frontend'

Write-Host "[SETUP] Backend Directory : $BackendDir"
Write-Host "[SETUP] Frontend Directory: $FrontendDir"
Write-Host ""

# ------------------------------------------------------------------------------
# 1. Locate and Validate Python (>= 3.10)
# ------------------------------------------------------------------------------
Write-Host "[1/7] Validating Python installation..." -ForegroundColor Yellow

function Find-ValidPython {
    $candidates = @('python', 'py', 'python3')
    foreach ($cmdName in $candidates) {
        $cmd = Get-Command $cmdName -ErrorAction SilentlyContinue
        if ($cmd) {
            $verOut = & $cmd.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $verOut -match '^(\d+)\.(\d+)') {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                if ($major -eq 3 -and $minor -ge 10) {
                    return $cmd.Source
                }
            }
        }
    }

    # Check common system install locations
    $defaultPaths = @(
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:LocalAppData\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe"
    )
    foreach ($p in $defaultPaths) {
        if (Test-Path $p) {
            $verOut = & $p -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $verOut -match '^(\d+)\.(\d+)') {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                if ($major -eq 3 -and $minor -ge 10) {
                    return $p
                }
            }
        }
    }
    return $null
}

$Python = Find-ValidPython
if (-not $Python) {
    Write-Host "[ERROR] Python 3.10+ was not found on your system." -ForegroundColor Red
    Write-Host "Please download and install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Be sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
    throw "Python 3.10+ not found."
}

$PyVersion = & $Python --version 2>&1
Write-Host "[OK] Using Python: $PyVersion ($Python)" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 2. Locate and Validate Node.js / npm
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/7] Validating Node.js and npm..." -ForegroundColor Yellow

$NpmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
if (-not $NpmCmd) {
    $NpmCmd = (Get-Command npm -ErrorAction SilentlyContinue)
}
if (-not $NpmCmd) {
    $nodePaths = @(
        "$env:ProgramFiles\nodejs\npm.cmd",
        "${env:ProgramFiles(x86)}\nodejs\npm.cmd"
    )
    foreach ($np in $nodePaths) {
        if (Test-Path $np) {
            $NpmCmd = Get-Item $np
            break
        }
    }
}

if (-not $NpmCmd) {
    Write-Host "[ERROR] Node.js / npm was not found on your system." -ForegroundColor Red
    Write-Host "Please download and install Node.js LTS (v18+) from https://nodejs.org/" -ForegroundColor Red
    throw "Node.js / npm not found."
}

$NpmExecutable = if ($NpmCmd.Source) { $NpmCmd.Source } else { $NpmCmd.FullName }
$NpmVersion = & $NpmExecutable --version 2>&1
Write-Host "[OK] Using npm: v$NpmVersion ($NpmExecutable)" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Configure Python Virtual Environment (.venv) & Backend Packages
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/7] Setting up Python virtual environment..." -ForegroundColor Yellow

$VenvDir = Join-Path $BackendDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

if ($Clean -and (Test-Path $VenvDir)) {
    Write-Host "[CLEAN] Clean setup requested. Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating fresh virtual environment in $VenvDir..."
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        throw "Failed to create Python virtual environment."
    }
    Write-Host "Upgrading pip, setuptools, and wheel..."
    & $VenvPython -m pip install --quiet --upgrade pip setuptools wheel
} else {
    Write-Host "Reusing existing virtual environment at $VenvDir."
}

# Fast check: are backend packages already installed?
$BackendCheck = & $VenvPython -c "import fastapi, uvicorn, ultralytics, cv2, skimage, yaml, torchvision; print('INSTALLED')" 2>$null
if ($BackendCheck -eq 'INSTALLED') {
    Write-Host "[OK] Backend dependencies (including torchvision) are already installed. (Skipping requirements reinstall)." -ForegroundColor Green
} else {
    Write-Host "Installing backend requirements..."
    $ReqFile = Join-Path $BackendDir 'requirements.txt'
    if (Test-Path $ReqFile) {
        $CudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { 'https://download.pytorch.org/whl/cu124' }
        & $VenvPython -m pip install --prefer-binary --extra-index-url $CudaIndex -r $ReqFile
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Python requirements from $ReqFile."
        }
    }
    $MlReqFile = Join-Path $BackendDir 'app\ml\tube\requirements.txt'
    if (Test-Path $MlReqFile) {
        Write-Host "Installing ML requirements from $MlReqFile..."
        & $VenvPython -m pip install --prefer-binary -r $MlReqFile
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install ML requirements from $MlReqFile."
        }
    }
}

# ------------------------------------------------------------------------------
# 4. Check PyTorch Acceleration (GPU / Older Driver / CPU) & Torchvision
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/7] Verifying PyTorch, torchvision, and hardware acceleration..." -ForegroundColor Yellow

# 1. Comprehensive NVIDIA hardware and driver detection
$HasNvidia = $false
$ComputeCap = 0.0

# Try nvidia-smi first (extracts compute capability for exact CUDA version matching)
$SmiPath = "nvidia-smi"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $SmiCandidates = @(
        "$env:SystemRoot\System32\nvidia-smi.exe",
        "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "${env:ProgramFiles(x86)}\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )
    foreach ($cand in $SmiCandidates) {
        if (Test-Path $cand) { $SmiPath = $cand; break }
    }
}
if ($SmiPath -ne "nvidia-smi" -or (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $SmiOutput = & $SmiPath --query-gpu=name,compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $null -ne $SmiOutput) {
        $HasNvidia = $true
        $ComputeCaps = $SmiOutput | ForEach-Object {
            $parts = $_ -split ','
            if ($parts.Count -ge 2) { [float]$parts[1].Trim() } else { 0.0 }
        }
        if ($ComputeCaps) { $ComputeCap = ($ComputeCaps | Measure-Object -Maximum).Maximum }
        Write-Host "[GPU INFO] nvidia-smi detected NVIDIA GPU(s) with max Compute Capability: $ComputeCap" -ForegroundColor Cyan
    }
}

# Fallback detection if nvidia-smi fails or is missing
if (-not $HasNvidia) {
    $AllVideoControllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    $NvidiaGpus = @($AllVideoControllers) | Where-Object {
        ($_.Name -match 'NVIDIA|GeForce|Quadro|Tesla|RTX') -or
        ($_.Caption -match 'NVIDIA|GeForce|Quadro|Tesla|RTX') -or
        ($_.Description -match 'NVIDIA|GeForce|Quadro|Tesla|RTX') -or
        ($_.VideoProcessor -match 'NVIDIA|GeForce|RTX') -or
        ($_.AdapterCompatibility -match 'NVIDIA') -or
        ($_.PNPDeviceID -match 'VEN_10DE')
    }
    if ($null -ne $NvidiaGpus -and @($NvidiaGpus).Count -gt 0) {
        $HasNvidia = $true
    } else {
        $NvidiaDriverFiles = @(
            "$env:SystemRoot\System32\nvcuda.dll",
            "$env:SystemRoot\System32\nvapi64.dll"
        )
        foreach ($ndf in $NvidiaDriverFiles) {
            if (Test-Path $ndf) { $HasNvidia = $true; break }
        }
    }
}

# Test if an existing PyTorch and torchvision are already installed and whether CUDA is operational safely
$TorchCheck = try {
    & $VenvPython -c "
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.init()
        t = torch.zeros((1, 1), device='cuda:0')
        _ = t + 1.0
        del t
        print('CUDA_OPERATIONAL:' + torch.cuda.get_device_name(0))
    else:
        print('CPU_OPERATIONAL')
except Exception as e:
    print('CUDA_ERROR:' + str(e))
" 2>$null
} catch { $null }

$TorchVisionCheck = try {
    & $VenvPython -c "
try:
    import torchvision, importlib.metadata
    _ = importlib.metadata.version('torchvision')
    print('TV_OK')
except Exception:
    pass
" 2>$null
} catch { $null }

$GpuName = if ($NvidiaGpus) { ($NvidiaGpus | Select-Object -First 1).Name } else { 'NVIDIA GPU' }

if ($TorchCheck -like 'CUDA_OPERATIONAL*' -and $TorchVisionCheck -eq 'TV_OK') {
    Write-Host "[OK] PyTorch and torchvision are already installed and verified operational on $GpuName (GPU Accelerated)." -ForegroundColor Green
} elseif ($HasNvidia) {
    Write-Host "[INFO] NVIDIA GPU detected ($GpuName). Installing CUDA-accelerated PyTorch and torchvision..." -ForegroundColor Cyan
    if ($ComputeCap -ge 12.0) {
        $TargetCuda = 'cu126'
        $CudaIndex = 'https://download.pytorch.org/whl/cu126'
    } elseif ($ComputeCap -ge 8.9 -or $ComputeCap -eq 0.0) {
        $TargetCuda = 'cu124'
        $CudaIndex = 'https://download.pytorch.org/whl/cu124'
    } else {
        $TargetCuda = 'cu121'
        $CudaIndex = 'https://download.pytorch.org/whl/cu121'
    }
    if ($env:PYTORCH_CUDA_INDEX) { $CudaIndex = $env:PYTORCH_CUDA_INDEX }
    Write-Host "[GPU INFO] Targeting CUDA $TargetCuda for maximum compatibility/performance." -ForegroundColor Cyan
    Write-Host "Fetching CUDA wheels from $CudaIndex..."
    & $VenvPython -m pip install --upgrade --force-reinstall --index-url $CudaIndex torch torchvision
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] $TargetCuda install returned non-zero; retrying with cu121 fallback..." -ForegroundColor Yellow
        & $VenvPython -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install CUDA PyTorch and torchvision."
        }
    }
} elseif ($TorchCheck -eq 'CPU_OPERATIONAL' -and $TorchVisionCheck -eq 'TV_OK') {
    Write-Host "[OK] CPU PyTorch and torchvision are already installed and operational for CPU inference." -ForegroundColor Green
} else {
    Write-Host "[INFO] No NVIDIA GPU hardware detected on this PC. Installing CPU-only PyTorch and torchvision..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade torch torchvision
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install CPU PyTorch and torchvision."
    }
}

# Strict verification: verify that torchvision metadata is functional right now
$TorchVisionVerify = & $VenvPython -c "import torch, torchvision, importlib.metadata; _ = importlib.metadata.version('torchvision'); print('VERIFIED')" 2>$null
if ($TorchVisionVerify -ne 'VERIFIED') {
    throw "PyTorch / torchvision verification failed: 'torchvision' package metadata is missing or corrupted."
}

# ------------------------------------------------------------------------------
# 5. Verify tube inference dependencies
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/7] Verifying tube inference packages..." -ForegroundColor Yellow
$TubeCheck = & $VenvPython -c "from ultralytics import YOLO; import cv2, numpy, torch, torchvision, skimage; print('INSTALLED')" 2>$null
if ($TubeCheck -ne 'INSTALLED') {
    $TubeReqFile = Join-Path $BackendDir 'app\ml\tube\requirements.txt'
    & $VenvPython -m pip install --prefer-binary -r $TubeReqFile
    if ($LASTEXITCODE -ne 0) { throw "Failed to install tube requirements from $TubeReqFile." }
}

# ------------------------------------------------------------------------------
# 6. Install Frontend Dependencies & Build Verification
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[6/7] Verifying frontend dependencies..." -ForegroundColor Yellow

$NodeModulesDir = Join-Path $FrontendDir 'node_modules'
$DistHtml = Join-Path $FrontendDir 'dist\index.html'

if ((Test-Path $NodeModulesDir) -and (Test-Path $DistHtml)) {
    Write-Host "[OK] Frontend packages and production build are already up-to-date." -ForegroundColor Green
} else {
    Push-Location $FrontendDir
    try {
        Write-Host "Installing frontend dependencies..."
        & $NpmExecutable install --include=optional
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed in $FrontendDir."
        }

        Write-Host "Building frontend assets..."
        & $NpmExecutable run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build verification failed."
        }
        Write-Host "[OK] Frontend build verified successfully." -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------------------
# 7. End-to-End Pre-Flight Smoke Test (Model & Inference Readiness)
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[7/7] Verifying inference engine and model components..." -ForegroundColor Yellow

$SmokeTestCode = @"
import sys
sys.path.insert(0, r'$BackendDir')
try:
    import torch
    import torchvision
    import importlib.metadata
    tv = importlib.metadata.version('torchvision')
    from ultralytics import YOLO
    from app.ml.tube.inference.tube_analyzer import TubeAnalyzer
    print(f'PASS|{torch.__version__}|{tv}')
except Exception as e:
    print(f'FAIL|{e}')
    sys.exit(1)
"@

$SmokeTestResult = & $VenvPython -c $SmokeTestCode 2>&1

if ($LASTEXITCODE -ne 0 -or -not ($SmokeTestResult -match 'PASS\|')) {
    throw "Pre-flight inference verification failed: $SmokeTestResult. Inference would fail at runtime."
}
Write-Host "[OK] Pre-flight inference smoke test PASSED! (tube inference and torchvision ready)." -ForegroundColor Green

# ------------------------------------------------------------------------------
# Completion Summary
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "       SETUP COMPLETED SUCCESSFULLY WITH NO ERRORS!    " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""

& $VenvPython -c "
import torch
import torchvision
print(f'  • PyTorch Version     : {torch.__version__}')
print(f'  • TorchVision Version : {torchvision.__version__}')
print(f'  • CUDA Enabled        : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  • GPU Device          : {torch.cuda.get_device_name(0)}')
"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Run the application in development: Double-click run.bat" -ForegroundColor White
Write-Host "  2. Package the Windows desktop app   : Double-click build.bat" -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor Green
