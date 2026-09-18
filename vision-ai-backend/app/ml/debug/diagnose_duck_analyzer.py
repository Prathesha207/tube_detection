#!/usr/bin/env python3

"""
Diagnose which duck_analyzer module is imported, AND manually run
inference against the real DuckAnalyzer -- no FastAPI server, no frontend.

This script does two things:

1. DIAGNOSIS
   Confirms exactly which duck_analyzer module/wheel this Python
   environment is importing.

2. MANUAL INFERENCE
   Loads config.yaml, constructs ONE DuckAnalyzer instance and
   reuses it for every frame.

For video inference, this script saves:

    output_dir/
    ├── raw_frames/
    │   ├── frame_000001.jpg
    │   ├── frame_000002.jpg
    │   └── ...
    │
    ├── annotated_frames/
    │   ├── frame_000001.jpg
    │   ├── frame_000002.jpg
    │   └── ...
    │
    ├── anomaly_frames/
    │   ├── anomaly_frame_000018.jpg
    │   ├── anomaly_frame_000019.jpg
    │   └── ...
    │
    └── annotated_video.mp4

raw_frames
----------
Original input frames exactly as read from OpenCV.

annotated_frames
----------------
Every frame returned by DuckAnalyzer's "annotated_frame".

anomaly_frames
--------------
Only frames identified as anomalous.

annotated_video.mp4
-------------------
Optional video containing every annotated frame.

Usage
-----

Diagnosis only:

    python diagnose_duck_analyzer.py

Video:

    python diagnose_duck_analyzer.py ^
        --video "D:\\videos\\test.mp4" ^
        --expected 18

Video with explicit output directory:

    python diagnose_duck_analyzer.py ^
        --video "D:\\videos\\test.mp4" ^
        --expected 18 ^
        --output-dir "D:\\duck_results"

Video with annotated MP4:

    python diagnose_duck_analyzer.py ^
        --video "D:\\videos\\test.mp4" ^
        --expected 18 ^
        --output-dir "D:\\duck_results" ^
        --save-annotated "D:\\duck_results\\annotated_video.mp4"

Limit processing:

    python diagnose_duck_analyzer.py ^
        --video "D:\\videos\\test.mp4" ^
        --expected 18 ^
        --output-dir "D:\\duck_results" ^
        --max-frames 200

Single image:

    python diagnose_duck_analyzer.py ^
        --image "D:\\frames\\frame.jpg" ^
        --expected 18 ^
        --output-dir "D:\\duck_results"

Specific config:

    python diagnose_duck_analyzer.py ^
        --video "D:\\videos\\test.mp4" ^
        --expected 18 ^
        --config "D:\\project\\backend\\ml\\config.yaml" ^
        --output-dir "D:\\duck_results"
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def get_duck_analyzer_class():
    """
    Load DuckAnalyzer from debug/duck_analyzer.py first if present,
    otherwise from vision-ai-backend/app/ml/duck_analyzer/analyzer.py.
    """
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

    # 1. First priority: analyzer.py right in debug folder
    _debug_analyzer_file = _script_dir / "analyzer.py"
    if _debug_analyzer_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("analyzer", str(_debug_analyzer_file))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["analyzer"] = module
            spec.loader.exec_module(module)
            duck_cls = getattr(module, "DuckAnalyzer", None)
            if duck_cls is not None:
                return duck_cls, str(_debug_analyzer_file), "debug_folder"

    # 2. Local source file: duck_analyzer/analyzer.py
    _local_analyzer_file = _script_dir.parent / "duck_analyzer" / "analyzer.py"
    if _local_analyzer_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_duck_analyzer", str(_local_analyzer_file))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["local_duck_analyzer"] = module
            spec.loader.exec_module(module)
            duck_cls = getattr(module, "DuckAnalyzer", None)
            if duck_cls is not None:
                return duck_cls, str(_local_analyzer_file), "local_source"

    # Fallback only if local file is missing
    from duck_analyzer import DuckAnalyzer
    import duck_analyzer
    return DuckAnalyzer, getattr(duck_analyzer, "__file__", "installed_package"), "site_packages"




# ======================================================================
# PART 1
# DIAGNOSIS
# ======================================================================


def _candidate_paths() -> List[str]:
    """
    Return every duck_analyzer-like file or directory currently visible
    on sys.path.
    """

    paths: List[str] = []
    seen = set()

    for entry in sys.path:
        if not entry:
            continue

        base = Path(entry)

        if not base.exists() or not base.is_dir():
            continue

        try:
            children = sorted(base.iterdir())
        except (PermissionError, OSError):
            continue

        for child in children:

            name = child.name

            if name.startswith("duck_analyzer"):

                try:
                    resolved = str(child.resolve())
                except OSError:
                    resolved = str(child)

                if resolved not in seen:

                    seen.add(resolved)
                    paths.append(resolved)

    return paths


def _safe_version() -> Optional[str]:

    try:
        return md.version("duck_analyzer")

    except md.PackageNotFoundError:
        return None

    except Exception:
        return None


def _show_pip_info() -> dict:

    try:

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "show",
                "duck_analyzer",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        return {
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

    except Exception as exc:

        return {
            "rc": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def run_diagnosis() -> None:
    _script_dir = str(Path(__file__).resolve().parent)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    elif sys.path[0] != _script_dir:
        sys.path.remove(_script_dir)
        sys.path.insert(0, _script_dir)

    print()
    print("duck_analyzer diagnosis")
    print("=" * 60)

    print(f"python executable: {sys.executable}")

    print()
    print("sys.path:")
    for index, entry in enumerate(sys.path[:10]):
        print(f"  [{index}] {entry}")

    # --------------------------------------------------------------
    # pip information
    # --------------------------------------------------------------

    pip_info = _show_pip_info()

    print()
    print("[ pip show duck_analyzer ]")
    print("-" * 60)

    print(f"return code: {pip_info['rc']}")

    if pip_info["stdout"]:
        print(pip_info["stdout"])

    if pip_info["stderr"]:
        print()
        print("STDERR:")
        print(pip_info["stderr"])

    # --------------------------------------------------------------
    # actual import
    # --------------------------------------------------------------

    print()
    print("[ import duck_analyzer ]")
    print("-" * 60)

    try:

        mod = importlib.import_module(
            "duck_analyzer"
        )

        file_path = getattr(
            mod,
            "__file__",
            None,
        )

        print(
            f"imported module: {mod!r}"
        )

        print(
            f"__file__: {file_path}"
        )

        if file_path and os.path.exists(file_path):

            print(
                f"mtime: {os.path.getmtime(file_path)}"
            )

            print(
                f"size: {os.path.getsize(file_path)} bytes"
            )

        else:

            print(
                "__file__ does not exist on disk"
            )

        version = _safe_version()

        print(
            f"installed dist version: {version}"
        )

        has_analyzer = hasattr(
            mod,
            "DuckAnalyzer",
        )

        print(
            f"has DuckAnalyzer: {has_analyzer}"
        )

        if has_analyzer:

            print(
                f"DuckAnalyzer class: "
                f"{mod.DuckAnalyzer}"
            )

    except Exception as exc:

        print(
            f"IMPORT ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------------
    # local source check
    # --------------------------------------------------------------

    print()
    print("[ local duck_analyzer source file ]")
    print("-" * 60)
    # NOTE: production (video_inference_service.py / camera_inference_service.py) imports
    # analyzer.py, not the legacy analyzer.py -- check the file that is
    # actually running, or this diagnosis silently reports on dead code.
    _local_analyzer = Path(__file__).resolve().parent.parent / "duck_analyzer" / "analyzer.py"
    if _local_analyzer.exists():
        print(f"FOUND local source file: {_local_analyzer}")
        print(f"mtime: {os.path.getmtime(_local_analyzer)}")
        print(f"size: {os.path.getsize(_local_analyzer)} bytes")
    else:
        print(f"Local source file NOT found at: {_local_analyzer}")

    # --------------------------------------------------------------
    # candidates
    # --------------------------------------------------------------

    print()
    print("[ duck_analyzer candidates on sys.path ]")
    print("-" * 60)

    candidates = _candidate_paths()

    if not candidates:

        print(
            "No duck_analyzer-like files/directories "
            "found on sys.path"
        )

    else:

        for candidate in candidates:

            path = Path(candidate)

            info = [
                candidate
            ]

            try:

                info.append(
                    f"exists={path.exists()}"
                )

                info.append(
                    f"is_dir={path.is_dir()}"
                )

                if path.exists():

                    info.append(
                        f"mtime={path.stat().st_mtime}"
                    )

            except OSError:
                pass

            print(
                " - " + " | ".join(info)
            )

    # --------------------------------------------------------------
    # filesystem search
    # --------------------------------------------------------------

    print()
    print(
        "[ file system search: "
        "duck_analyzer*.whl and duck_analyzer.py ]"
    )
    print("-" * 60)

    seen = set()

    search_dirs = [
        Path.cwd(),
        Path(__file__).resolve().parents[3],
        Path.home() / "Downloads",
    ]

    for root in search_dirs:

        if not root or not root.exists():
            continue

        try:

            for match in root.rglob(
                "duck_analyzer*.whl"
            ):

                if "__pycache__" in str(match):
                    continue

                key = str(match.resolve())

                if key not in seen:

                    seen.add(key)
                    print(match)

            for match in root.rglob(
                "duck_analyzer.py"
            ):

                if "__pycache__" in str(match):
                    continue

                key = str(match.resolve())

                if key not in seen:

                    seen.add(key)
                    print(match)

        except (
            PermissionError,
            OSError,
        ):

            continue


# ======================================================================
# PART 2
# PATH RESOLUTION
# ======================================================================


def _resolve_model_path(
    configured_path: Optional[str],
    ml_dir: str,
) -> str:
    """
    Find best.pt prioritizing debug folder and configured path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    if configured_path:
        if os.path.isabs(configured_path) and os.path.exists(configured_path):
            return os.path.abspath(configured_path)
        candidates.extend([
            os.path.join(ml_dir, configured_path),
            os.path.join(script_dir, configured_path),
            os.path.abspath(configured_path),
        ])

    candidates.extend([
        os.path.join(script_dir, "best.pt"),
        os.path.join(ml_dir, "best.pt"),
        os.path.join(ml_dir, "models", "best.pt"),
        os.path.join(script_dir, "..", "models", "best.pt"),
        os.path.join(os.getcwd(), "models", "best.pt"),
        os.path.join(os.getcwd(), "best.pt"),
    ])

    checked = []
    for cand in candidates:
        if not cand:
            continue
        checked.append(cand)
        if os.path.exists(cand):
            return os.path.abspath(cand)

    raise FileNotFoundError(
        "Could not find best.pt on this machine.\n\n"
        "Checked:\n  "
        + "\n  ".join(checked)
    )


def _resolve_config_path(
    configured_path: Optional[str] = None,
) -> str:
    """
    Find config.yaml robustly prioritizing debug/ directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # --------------------------------------------------------------
    # User supplied config
    # --------------------------------------------------------------
    if configured_path:
        if os.path.isabs(configured_path):
            candidates.append(configured_path)
        else:
            candidates.extend([
                os.path.abspath(configured_path),
                os.path.join(script_dir, configured_path),
                os.path.join(script_dir, "..", configured_path),
                os.path.join(os.getcwd(), configured_path),
            ])

    # --------------------------------------------------------------
    # Standard locations (debug/ first)
    # --------------------------------------------------------------
    candidates.extend([
        os.path.join(script_dir, "config.yaml"),
        os.path.join(script_dir, "..", "config.yaml"),
        os.path.join(os.getcwd(), "app", "ml", "debug", "config.yaml"),
        os.path.join(os.getcwd(), "app", "ml", "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
    ])

    seen = set()

    checked = []

    for cand in candidates:

        if not cand:
            continue

        abs_cand = os.path.abspath(
            cand
        )

        if abs_cand in seen:
            continue

        seen.add(abs_cand)

        checked.append(abs_cand)

        if os.path.exists(abs_cand):

            return abs_cand

    raise FileNotFoundError(
        "config.yaml not found.\n\n"
        "Checked:\n  "
        + "\n  ".join(checked)
    )


# ======================================================================
# PART 3 & 4
# FRAME SUMMARY & ANOMALY (PURE ML RETURN, NO EXTRA LOGIC)
# ======================================================================


def _print_frame_summary(
    frame_no: int,
    result: dict,
    analyzer=None,
) -> None:
    """
    Print the exact fields returned by DuckAnalyzer -- no recalculation.
    """
    status = result.get("status", "UNKNOWN")
    detected = result.get("detected_duck_count", 0)
    expected = result.get("expected_duck_count", 0)
    missing_ids = result.get("missing_ids", [])
    added_ids = result.get("added_ids", [])
    other_ids = result.get("other_ids", [])
    hand = result.get("hand_detected", False)
    anchor_locked = result.get("anchor_locked", False)
    reasons = result.get("reasons", [])

    line = (
        f"frame {frame_no:>6} | "
        f"status={str(status):<10} | "
        f"anchor_locked={str(anchor_locked):<5} | "
        f"ducks={detected}/{expected} | "
        f"other={len(other_ids)} | "
        f"hand={hand}"
    )

    if missing_ids:
        line += f" | missing={missing_ids}"
    if added_ids:
        line += f" | added={added_ids}"
    if reasons:
        line += f" | reasons={reasons}"

    print(line)

    for thumbnail in result.get("thumbnails", []):
        print(
            "    thumbnail event: "
            f"{thumbnail.get('event')} "
            f"id={thumbnail.get('id')} "
            f"species={thumbnail.get('species')}"
        )


def _is_anomaly_frame(
    result: dict,
    analyzer=None,
    expected_duck_count: int = None,
) -> bool:
    """
    Pure ML check: return True only if DuckAnalyzer's status is ANOMALY.
    No extra logic, no overrides.
    """
    return str(result.get("status", "")).upper() == "ANOMALY"


# ======================================================================
# PART 6
# SAFE IMAGE SAVE
# ======================================================================


def _save_jpeg(
    path: str,
    frame,
) -> bool:

    """
    Save one frame as JPEG.

    Returns True when OpenCV reports a successful write.
    """

    try:

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        return bool(
            __import__("cv2").imwrite(
                path,
                frame,
                [
                    __import__("cv2").IMWRITE_JPEG_QUALITY,
                    95,
                ],
            )
        )

    except Exception as exc:

        print(
            f"WARNING: Failed to save frame "
            f"{path}: {type(exc).__name__}: {exc}"
        )

        return False


# ======================================================================
# PART 7
# MANUAL INFERENCE
# ======================================================================


def run_manual_inference(
    config_path: Optional[str],
    expected_duck_count: int,
    video_path: Optional[str],
    image_path: Optional[str],
    save_annotated: Optional[str],
    max_frames: Optional[int],
    output_dir: Optional[str],
    save_all_frames: bool = False,
) -> int:

    import cv2
    import yaml

    # --------------------------------------------------------------
    # Validate input arguments
    # --------------------------------------------------------------

    if video_path and image_path:

        print(
            "ERROR: Use either --video or --image, "
            "not both."
        )

        return 1

    if not video_path and not image_path:

        print(
            "ERROR: Provide --video or --image."
        )

        return 1

    if expected_duck_count < 0:

        print(
            "ERROR: --expected cannot be negative."
        )

        return 1

    if max_frames is not None and max_frames <= 0:

        print(
            "ERROR: --max-frames must be greater than 0."
        )

        return 1

    # --------------------------------------------------------------
    # Import real DuckAnalyzer (prioritizing local app/ml/duck_analyzer/analyzer.py)
    # --------------------------------------------------------------

    try:
        DuckAnalyzer, analyzer_source_path, source_kind = get_duck_analyzer_class()
        print(f"[LOAD] Using DuckAnalyzer from: {analyzer_source_path} ({source_kind})")

    except Exception as exc:

        print(
            f"Could not import DuckAnalyzer: {exc}"
        )

        print(
            "\nRun the diagnosis section first "
            "to determine which module is being imported."
        )

        return 1

    # --------------------------------------------------------------
    # Resolve config
    # --------------------------------------------------------------

    try:

        config_path = _resolve_config_path(
            config_path
        )

    except FileNotFoundError as exc:

        print(str(exc))

        return 1

    print()
    print(
        f"Using config: {config_path}"
    )

    # --------------------------------------------------------------
    # Load YAML
    # --------------------------------------------------------------

    try:

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            cfg = yaml.safe_load(f) or {}

    except Exception as exc:

        print(
            f"Could not load config.yaml: "
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    # --------------------------------------------------------------
    # Resolve model
    # --------------------------------------------------------------

    ml_dir = os.path.dirname(
        config_path
    )

    try:

        cfg["model_path"] = _resolve_model_path(
            cfg.get("model_path"),
            ml_dir,
        )

    except FileNotFoundError as exc:

        print(str(exc))

        return 1

    print(
        f"Using model_path: "
        f"{cfg['model_path']}"
    )

    if cfg.get("roi_path"):
        roi_candidates = [
            os.path.join(ml_dir, cfg["roi_path"]),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["roi_path"]),
            os.path.abspath(cfg["roi_path"]),
        ]
        for rc in roi_candidates:
            if os.path.exists(rc):
                cfg["roi_path"] = os.path.abspath(rc)
                break
        print(f"Using roi_path: {cfg['roi_path']}")

    # --------------------------------------------------------------
    # Disable duplicate internal saving to Documents directory
    # --------------------------------------------------------------
    cfg["save_local"] = False
    cfg["annotated_dir"] = None
    cfg["thumbnail_dir"] = None

    # --------------------------------------------------------------
    # Create temporary session config in system temp dir
    # --------------------------------------------------------------

    import tempfile

    session_config_path = os.path.join(
        tempfile.gettempdir(),
        f"_duck_analyzer_session_{os.getpid()}.yaml",
    )

    try:

        with open(
            session_config_path,
            "w",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
            )

    except Exception as exc:

        print(
            f"Could not create temporary config: "
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    analyzer = None
    cap = None
    out_writer = None

    try:

        # ==========================================================
        # CONSTRUCT ONE ANALYZER
        # ==========================================================

        print()
        print(
            "Constructing DuckAnalyzer ONCE..."
        )

        print(
            "The same analyzer instance will be reused "
            "for every frame."
        )

        t0 = time.time()

        analyzer = DuckAnalyzer(
            session_config_path,
            expected_duck_count=expected_duck_count,
        )

        model_load_time = (
            time.time() - t0
        )

        print(
            f"Model load took "
            f"{model_load_time:.2f}s"
        )

        # ==========================================================
        # SINGLE IMAGE
        # ==========================================================

        if image_path:

            frame = cv2.imread(
                image_path
            )

            if frame is None:

                print(
                    f"Could not read image: "
                    f"{image_path}"
                )

                return 1

            # ------------------------------------------------------
            # Determine output directory
            # ------------------------------------------------------

            if output_dir:

                output_root = os.path.abspath(
                    output_dir
                )

            elif save_annotated:

                output_root = (
                    os.path.splitext(
                        os.path.abspath(
                            save_annotated
                        )
                    )[0]
                    + "_frames"
                )

            else:

                image_stem = Path(
                    image_path
                ).stem

                output_root = os.path.join(
                    os.getcwd(),
                    "manual_inference_output",
                    image_stem,
                )

            raw_dir = os.path.join(
                output_root,
                "raw_frames",
            )

            annotated_dir = os.path.join(
                output_root,
                "annotated_frames",
            )

            anomaly_dir = os.path.join(
                output_root,
                "anomaly_frames",
            )

            os.makedirs(
                raw_dir,
                exist_ok=True,
            )

            os.makedirs(
                annotated_dir,
                exist_ok=True,
            )

            os.makedirs(
                anomaly_dir,
                exist_ok=True,
            )

            # ------------------------------------------------------
            # Save raw image BEFORE analyzer
            # ------------------------------------------------------

            raw_path = os.path.join(
                raw_dir,
                "frame_000001.jpg",
            )

            _save_jpeg(
                raw_path,
                frame,
            )

            # ------------------------------------------------------
            # Run analyzer
            # ------------------------------------------------------

            result = analyzer.process_frame(
                frame
            )

            _print_frame_summary(
                1,
                result,
                analyzer,
            )

            # ------------------------------------------------------
            # Get backend-generated annotated frame
            # ------------------------------------------------------

            annotated_frame = result.get(
                "annotated_frame"
            )

            if annotated_frame is None:

                print(
                    "DuckAnalyzer did not return "
                    "annotated_frame."
                )

                return 1

            # ------------------------------------------------------
            # Save annotated image
            # ------------------------------------------------------

            annotated_path = os.path.join(
                annotated_dir,
                "frame_000001.jpg",
            )

            _save_jpeg(
                annotated_path,
                annotated_frame,
            )

            # ------------------------------------------------------
            # Determine anomaly
            # ------------------------------------------------------

            is_anomaly = _is_anomaly_frame(
                result,
                analyzer,
                expected_duck_count,
            )

            if is_anomaly:

                anomaly_path = os.path.join(
                    anomaly_dir,
                    "anomaly_frame_000001.jpg",
                )

                _save_jpeg(
                    anomaly_path,
                    annotated_frame,
                )

                print(
                    f"Saved anomaly frame: "
                    f"{anomaly_path}"
                )

            # ------------------------------------------------------
            # Optional standalone annotated image
            # ------------------------------------------------------

            if save_annotated:

                os.makedirs(
                    os.path.dirname(
                        os.path.abspath(
                            save_annotated
                        )
                    ),
                    exist_ok=True,
                )

                cv2.imwrite(
                    save_annotated,
                    annotated_frame,
                )

                print(
                    f"Wrote annotated image to "
                    f"{save_annotated}"
                )

            print()
            print(
                "=" * 60
            )
            print(
                "IMAGE INFERENCE COMPLETE"
            )
            print(
                "=" * 60
            )

            print(
                f"Output directory:"
                f"\n  {output_root}"
            )

            print(
                f"\nRaw frame:"
                f"\n  {raw_path}"
            )

            print(
                f"\nAnnotated frame:"
                f"\n  {annotated_path}"
            )

            print(
                f"\nAnomaly:"
                f"\n  {'YES' if is_anomaly else 'NO'}"
            )

            return 0

        # ==========================================================
        # VIDEO
        # ==========================================================

        if not os.path.exists(video_path):
            p = Path(video_path)
            alt = p.parent / "test-videos" / p.name
            if alt.exists():
                print(f"[PATH] Located video in test-videos: {alt}")
                video_path = str(alt)

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                f"Could not open video: "
                f"{video_path}"
            )

            return 1

        fps = (
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 30.0
        )
        if fps <= 0 or fps > 120:
            fps = 30.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        print()
        print(
            f"Video: "
            f"{width}x{height} "
            f"@ {fps:.2f} fps, "
            f"{total} frames"
        )

        # ==========================================================
        # OUTPUT ROOT
        # ==========================================================

        if output_dir:

            output_root = os.path.abspath(
                output_dir
            )

        elif save_annotated:

            output_root = (
                os.path.splitext(
                    os.path.abspath(
                        save_annotated
                    )
                )[0]
                + "_frames"
            )

        else:

            video_stem = Path(
                video_path
            ).stem

            output_root = os.path.join(
                os.getcwd(),
                "manual_inference_output",
                video_stem,
            )

        # ==========================================================
        # REQUIRED OUTPUT FOLDERS
        # ==========================================================

        raw_dir = os.path.join(
            output_root,
            "raw_frames",
        )

        annotated_dir = os.path.join(
            output_root,
            "annotated_frames",
        )

        anomaly_dir = os.path.join(
            output_root,
            "anomaly_frames",
        )

        if save_all_frames:
            os.makedirs(
                raw_dir,
                exist_ok=True,
            )
            os.makedirs(
                annotated_dir,
                exist_ok=True,
            )

        os.makedirs(
            anomaly_dir,
            exist_ok=True,
        )

        print()
        print(
            "Output folders:"
        )

        if save_all_frames:
            print(
                f"  Raw frames:"
                f"\n    {raw_dir}"
            )
            print(
                f"  Annotated frames:"
                f"\n    {annotated_dir}"
            )

        print(
            f"  Anomaly frames:"
            f"\n    {anomaly_dir}"
        )

        # ==========================================================
        # OPTIONAL ANNOTATED VIDEO
        # ==========================================================

        if save_annotated:

            save_annotated = os.path.abspath(
                save_annotated
            )

            os.makedirs(
                os.path.dirname(
                    save_annotated
                ),
                exist_ok=True,
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out_writer = cv2.VideoWriter(
                save_annotated,
                fourcc,
                fps,
                (
                    width,
                    height,
                ),
            )

            if not out_writer.isOpened():

                print(
                    "WARNING: Could not open "
                    "annotated video writer."
                )

                out_writer.release()
                out_writer = None

            else:

                print()
                print(
                    f"Annotated video:"
                    f"\n  {save_annotated}"
                )

        # ==========================================================
        #
        #  PROCESS VIDEO
        # ==========================================================

        frame_no = 0
        anomaly_count = 0
        inference_error_count = 0

        t_start = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_no += 1

            # ------------------------------------------------------
            # Optional: Save raw frame if save_all_frames is True
            # ------------------------------------------------------

            if save_all_frames:

                raw_path = os.path.join(
                    raw_dir,
                    f"frame_{frame_no:06d}.jpg",
                )

                raw_saved = _save_jpeg(
                    raw_path,
                    frame,
                )

                if not raw_saved:

                    print(
                        f"WARNING: Failed to save "
                        f"raw frame {frame_no}"
                    )

            # ------------------------------------------------------
            # Run DuckAnalyzer
            # ------------------------------------------------------

            try:

                result = analyzer.process_frame(
                    frame
                )

            except Exception as exc:

                inference_error_count += 1

                print()
                print(
                    f"ERROR processing frame "
                    f"{frame_no}: "
                    f"{type(exc).__name__}: {exc}"
                )

                # We already saved the raw frame.
                #
                # There is no valid annotated frame to save
                # when DuckAnalyzer itself failed.

                if (
                    max_frames
                    and frame_no >= max_frames
                ):

                    print(
                        f"Stopping at "
                        f"--max-frames={max_frames}"
                    )

                    break

                continue

            # ------------------------------------------------------
            # Print analyzer information
            # ------------------------------------------------------

            _print_frame_summary(
                frame_no,
                result,
                analyzer,
            )

            # ------------------------------------------------------
            # Get annotated frame
            #
            # This is the frame generated by DuckAnalyzer.
            #
            # DO NOT redraw bounding boxes here.
            # ------------------------------------------------------

            annotated_frame = result.get(
                "annotated_frame"
            )

            if annotated_frame is None:

                print(
                    f"WARNING: DuckAnalyzer returned "
                    f"no annotated_frame for frame "
                    f"{frame_no}"
                )

                if (
                    max_frames
                    and frame_no >= max_frames
                ):

                    print(
                        f"Stopping at "
                        f"--max-frames={max_frames}"
                    )

                    break

                continue

            # ------------------------------------------------------
            # Optional: SAVE ANNOTATED FRAME if save_all_frames is True
            # ------------------------------------------------------

            if save_all_frames:

                annotated_path = os.path.join(
                    annotated_dir,
                    f"frame_{frame_no:06d}.jpg",
                )

                annotated_saved = _save_jpeg(
                    annotated_path,
                    annotated_frame,
                )

                if not annotated_saved:

                    print(
                        f"WARNING: Failed to save "
                        f"annotated frame {frame_no}"
                    )

            # ------------------------------------------------------
            # CHECK ANOMALY
            # ------------------------------------------------------

            is_anomaly = _is_anomaly_frame(
                result,
                analyzer,
                expected_duck_count,
            )

            # ------------------------------------------------------
            # SAVE ANOMALY ANNOTATED FRAME
            # ------------------------------------------------------

            if is_anomaly:

                anomaly_count += 1

                anomaly_path = os.path.join(
                    anomaly_dir,
                    f"anomaly_frame_{frame_no:06d}.jpg",
                )

                anomaly_saved = _save_jpeg(
                    anomaly_path,
                    annotated_frame,
                )

                if not anomaly_saved:

                    print(
                        f"WARNING: Failed to save "
                        f"anomaly frame {frame_no}"
                    )

            # ------------------------------------------------------
            # WRITE ANNOTATED VIDEO
            # ------------------------------------------------------

            if out_writer is not None:

                try:

                    out_writer.write(
                        annotated_frame
                    )

                except Exception as exc:

                    print(
                        f"WARNING: Failed to write "
                        f"annotated video frame "
                        f"{frame_no}: "
                        f"{type(exc).__name__}: {exc}"
                    )

            # ------------------------------------------------------
            # MAX FRAME LIMIT
            # ------------------------------------------------------

            if (
                max_frames
                and frame_no >= max_frames
            ):

                print()
                print(
                    f"Stopping at "
                    f"--max-frames={max_frames}"
                )

                break

        # ==========================================================
        # FINISHED
        # ==========================================================

        elapsed = (
            time.time()
            - t_start
        )

        average_fps = (
            frame_no / elapsed
            if elapsed > 0
            else 0.0
        )

        # ==========================================================
        # FINAL SUMMARY
        # ==========================================================

        print()
        print(
            "=" * 70
        )

        print(
            "MANUAL INFERENCE COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Input video          : {video_path}"
        )

        print(
            f"Expected ducks       : "
            f"{expected_duck_count}"
        )

        print(
            f"Frames processed     : "
            f"{frame_no}"
        )

        print(
            f"Anomaly frames       : "
            f"{anomaly_count}"
        )

        print(
            f"Inference errors     : "
            f"{inference_error_count}"
        )

        print(
            f"Elapsed time         : "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"Average processing   : "
            f"{average_fps:.2f} FPS"
        )

        print()
        print(
            "Saved output:"
        )

        if save_all_frames:

            print(
                f"  RAW FRAMES"
                f"\n    {raw_dir}"
            )

            print(
                f"\n  ALL ANNOTATED FRAMES"
                f"\n    {annotated_dir}"
            )

        print(
            f"\n  ANOMALY ANNOTATED FRAMES"
            f"\n    {anomaly_dir}"
        )

        if save_annotated:

            print(
                f"\n  ANNOTATED VIDEO"
                f"\n    {save_annotated}"
            )


        print()
        print(
            "=" * 70
        )

        return 0

    finally:

        # ==========================================================
        # RELEASE VIDEO
        # ==========================================================

        if cap is not None:

            try:
                cap.release()
            except Exception:
                pass

        # ==========================================================
        # RELEASE OUTPUT VIDEO
        # ==========================================================

        if out_writer is not None:

            try:
                out_writer.release()
            except Exception:
                pass

        # ==========================================================
        # CLOSE DUCK ANALYZER
        # ==========================================================

        if analyzer is not None:

            try:
                analyzer.close()
            except Exception as exc:

                print(
                    f"WARNING: analyzer.close() "
                    f"failed: {exc}"
                )



        # ==========================================================
        # REMOVE TEMP CONFIG
        # ==========================================================

        if os.path.exists(
            session_config_path
        ):

            try:

                os.remove(
                    session_config_path
                )

            except OSError:

                pass


# ======================================================================
# PART 8
# MAIN
# ======================================================================


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    parser.add_argument(
        "--video",
        help="Path to a video file to run inference on",
    )

    parser.add_argument(
        "--image",
        help="Path to a single image to run inference on",
    )

    # --------------------------------------------------------------
    # Analyzer settings
    # --------------------------------------------------------------

    parser.add_argument(
        "--expected",
        type=int,
        default=18,
        help="Expected duck count (default: 18)",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config.yaml "
            "(default: auto-detected)"
        ),
    )

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. The script creates "
            "raw_frames, annotated_frames, and "
            "anomaly_frames inside it."
        ),
    )

    parser.add_argument(
        "--save-annotated",
        default=None,
        help=(
            "Path to write annotated output "
            "(image when using --image, "
            "video when using --video)"
        ),
    )



    parser.add_argument(
        "--save-all-frames",
        action="store_true",
        help=(
            "Save individual JPEG files for every single raw frame "
            "and annotated frame (default: False, only saves anomaly frames "
            "and annotated video to maximize speed)"
        ),
    )

    # --------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Stop after N frames "
            "(video only)"
        ),
    )

    # --------------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------------

    parser.add_argument(
        "--skip-diagnosis",
        action="store_true",
        help=(
            "Skip the import-diagnosis section"
        ),
    )

    args = parser.parse_args()

    # ==============================================================
    # DIAGNOSIS
    # ==============================================================

    if not args.skip_diagnosis:

        run_diagnosis()

    # ==============================================================
    # MANUAL INFERENCE
    # ==============================================================

    if args.video or args.image:

        print()
        print(
            "=" * 60
        )

        print(
            "manual inference check"
        )

        print(
            "=" * 60
        )

        return run_manual_inference(
            config_path=args.config,
            expected_duck_count=args.expected,
            video_path=args.video,
            image_path=args.image,
            save_annotated=args.save_annotated,
            max_frames=args.max_frames,
            output_dir=args.output_dir,
            save_all_frames=args.save_all_frames,
        )

    # ==============================================================
    # DIAGNOSIS ONLY
    # ==============================================================

    print()
    print(
        "(no --video/--image given - "
        "diagnosis only)"
    )

    print(
        "Add --video or --image to also "
        "run real inference."
    )

    return 0


# ======================================================================
# ENTRY POINT
# ======================================================================


if __name__ == "__main__":

    raise SystemExit(
        main()
    )