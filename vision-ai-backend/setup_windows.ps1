[CmdletBinding()]
param(
    [switch]$Clean
)

# ==============================================================================
# Vision Monitor - Windows Environment Setup Script
# Configures Python virtual environment, dependencies, CUDA PyTorch,
# DuckAnalyzer wheel, and Node.js frontend packages without errors.
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
$BackendCheck = & $VenvPython -c "import fastapi, uvicorn, ultralytics, cv2, mediapipe, yaml, torchvision; print('INSTALLED')" 2>$null
if ($BackendCheck -eq 'INSTALLED') {
    Write-Host "[OK] Backend dependencies (including torchvision) are already installed. (Skipping requirements reinstall)." -ForegroundColor Green
} else {
    Write-Host "Installing backend requirements..."
    $ReqFile = Join-Path $BackendDir 'requirements.txt'
    if (Test-Path $ReqFile) {
        $CudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { 'https://download.pytorch.org/whl/cu121' }
        & $VenvPython -m pip install --prefer-binary --extra-index-url $CudaIndex -r $ReqFile
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install Python requirements from $ReqFile."
        }
    }
    $MlReqFile = Join-Path $BackendDir 'app\ml\requirements.txt'
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

$NvidiaGpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" }
$HasNvidia = ($null -ne $NvidiaGpus -and @($NvidiaGpus).Count -gt 0)

if (-not $HasNvidia) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $HasNvidia = $true
    }
}

# Test if an existing PyTorch and torchvision are already installed and whether CUDA is operational
$TorchCheck = & $VenvPython -c "
import torch
try:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        t = torch.zeros((1, 1), device='cuda:0')
        _ = t + 1.0
        del t
        print('CUDA_OPERATIONAL')
    else:
        print('CPU_OPERATIONAL')
except Exception as e:
    print('DRIVER_TOO_OLD')
" 2>$null

$TorchVisionCheck = & $VenvPython -c "import torchvision; import importlib.metadata; _ = importlib.metadata.version('torchvision'); print('TV_OK')" 2>$null

if ($TorchCheck -eq 'CUDA_OPERATIONAL' -and $TorchVisionCheck -eq 'TV_OK') {
    $GpuName = if ($NvidiaGpus) { ($NvidiaGpus | Select-Object -First 1).Name } else { 'NVIDIA GPU' }
    Write-Host "[OK] PyTorch and torchvision are already installed and verified operational on $GpuName." -ForegroundColor Green
} elseif ($TorchCheck -eq 'CPU_OPERATIONAL' -and $TorchVisionCheck -eq 'TV_OK' -and -not $HasNvidia) {
    Write-Host "[OK] CPU PyTorch and torchvision are already installed and operational for CPU inference." -ForegroundColor Green
} elseif ($HasNvidia) {
    Write-Host "[INFO] NVIDIA GPU detected. Ensuring CUDA PyTorch and torchvision are installed..." -ForegroundColor Cyan
    $CudaIndex = if ($env:PYTORCH_CUDA_INDEX) { $env:PYTORCH_CUDA_INDEX } else { 'https://download.pytorch.org/whl/cu121' }
    & $VenvPython -m pip install --upgrade --index-url $CudaIndex torch torchvision
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install CUDA PyTorch and torchvision from $CudaIndex."
    }
} else {
    Write-Host "[INFO] CPU mode detected. Ensuring PyTorch and torchvision are installed..." -ForegroundColor Cyan
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
# 5. Install Bundled DuckAnalyzer Wheel
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/7] Verifying DuckAnalyzer package..." -ForegroundColor Yellow

$DuckAnalyzerInstalled = & $VenvPython -c "import duck_analyzer; print('INSTALLED')" 2>$null
if ($DuckAnalyzerInstalled -eq 'INSTALLED') {
    Write-Host "[OK] DuckAnalyzer package is already installed." -ForegroundColor Green
} else {
    # Sort wheels by semantic version (e.g. 1.0.15 > 1.0.9 instead of string alphabetical sort)
    $DuckAnalyzerWheel = Get-ChildItem -Path (Join-Path $BackendDir 'app\ml') -Filter 'duck_analyzer-*.whl' -Recurse -ErrorAction SilentlyContinue |
        Sort-Object {
            if ($_.Name -match 'duck_analyzer-([0-9]+(\.[0-9]+)*)') {
                try { [System.Version]$matches[1] } catch { [System.Version]'0.0.0' }
            } else {
                [System.Version]'0.0.0'
            }
        } -Descending |
        Select-Object -First 1

    if (-not $DuckAnalyzerWheel) {
        throw "The bundled duck_analyzer wheel is missing from app\ml."
    }

    Write-Host "Installing $($DuckAnalyzerWheel.Name)..."
    & $VenvPython -m pip install --prefer-binary --no-deps --force-reinstall $DuckAnalyzerWheel.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $($DuckAnalyzerWheel.Name)."
    }
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

$SmokeTestResult = & $VenvPython -c "
import sys
try:
    import torch
    import torchvision
    import importlib.metadata
    tv = importlib.metadata.version('torchvision')
    from ultralytics import YOLO
    import duck_analyzer
    print(f'PASS|{torch.__version__}|{tv}')
except Exception as e:
    print(f'FAIL|{e}', file=sys.stderr)
    sys.exit(1)
" 2>&1

if ($LASTEXITCODE -ne 0 -or -not ($SmokeTestResult -match '^PASS\|')) {
    throw "Pre-flight inference verification failed: $SmokeTestResult. Inference would fail at runtime."
}
Write-Host "[OK] Pre-flight inference smoke test PASSED! (YOLO, DuckAnalyzer, and torchvision ready)." -ForegroundColor Green

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
