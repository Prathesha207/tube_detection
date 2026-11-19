# Portable desktop releases

Each installer includes Electron, the Python backend, model weights, Python
libraries, and native video/ML dependencies. An end user does **not** need to
install Python, Node.js, PyTorch, or the project source code.

Build each target on its own native OS and CPU architecture. PyInstaller and
PyTorch cannot safely cross-compile these artifacts.

| Target | Build command | Output |
| --- | --- | --- |
| Windows x64 CPU | `.\build_windows_desktop.ps1 -Acceleration cpu` | NSIS `.exe` |
| Windows x64 NVIDIA | `.\build_windows_desktop.ps1 -Acceleration cuda` | NSIS `.exe` |
| Linux x64 | `./build_linux_desktop.sh` on x86_64 Linux | AppImage and `.deb` |
| Linux ARM64 | `./build_linux_desktop.sh` on aarch64 Linux | AppImage and `.deb` |

Run the Windows command from the repository's `backend` directory in
PowerShell. Run the Linux command from `backend` after
`chmod +x build_linux_desktop.sh`. Artifacts are written to `frontend/dist_app`.

The GitHub workflows build the same artifacts: `windows-desktop.yml` produces
the x64 NSIS installer and `linux-desktop.yml` produces x64 and ARM64 Linux
packages. The ARM64 workflow needs a GitHub ARM64 runner; if the repository
does not have access to `ubuntu-24.04-arm`, change that matrix runner to an
ARM64 self-hosted runner.

## NVIDIA behavior

Use the CPU installer as the default broadly compatible release. It runs on
every supported machine. The CUDA installer embeds CUDA-enabled PyTorch and
automatically uses NVIDIA acceleration when the target computer has a
compatible NVIDIA driver and GPU. If CUDA is unavailable, inference already
falls back to CPU; no separate Python/PyTorch installation is required.

The CUDA installer cannot bundle the NVIDIA kernel display driver. Users who
want acceleration must install the current NVIDIA driver for their GPU. Linux
ARM64 CUDA (for example Jetson) is vendor-specific: build on that target using
its vendor PyTorch/CUDA image; the script retains that PyTorch installation
instead of replacing it.

## Smoke test before sharing

1. Install the generated artifact on a clean machine of the matching platform
   and architecture.
2. Open Vision AI and wait for the splash screen to close.
3. Upload a short video or connect a camera, then confirm analysis completes.
4. On an NVIDIA target, inspect the backend log or model status and confirm
   CUDA is selected; repeat once without an NVIDIA GPU to confirm CPU fallback.

Release artifacts are intentionally not committed. They are in
`vision-ai-frontend/dist_app` and can be uploaded by the CI workflow.
