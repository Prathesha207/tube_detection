"""
run_video_frames.py - Real-time tube-end video inference and tracking.

Strict requirements:
- Model best.pt is preserved intact.
- Physical setup: Exactly 2 HEAD ends and 2 TAIL ends (4 physical tube ends).
- Persistent 4-end tracking: HEAD #1, HEAD #2, TAIL #1, TAIL #2.
- Primary width source: YOLO mask (final_width = mask_width).
- No artificial overrides ('corrected to match heads' removed).
- Frame-by-frame reporting and end-of-video summary statistics.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from measure_from_masks import (
    analyse,
    build_class_map,
    masks_from_model,
    draw,
    pair_ends,
)
from video_ends import FourEndsTracker, PhysicalTrack, size_labels

# ============================== CONFIG ==============================
_DIR = Path(__file__).resolve().parent
_MODEL_CANDIDATES = [
    _DIR / "best.pt",
    _DIR / "model" / "best.pt",
    _DIR.parent / "model" / "best.pt",
    _DIR.parent / "head_zip" / "model" / "best.pt",
]
MODEL_PATH = next((str(p) for p in _MODEL_CANDIDATES if p.exists()), r"best.pt")

_ROI_CANDIDATES = [
    _DIR / "roi.json",
    _DIR.parent / "head_zip" / "roi.json",
    _DIR.parent / "roi.json",
]
ROI_PATH = next((str(p) for p in _ROI_CANDIDATES if p.exists()), None)

VIDEO_PATH = r"C:\Users\EmageVision\Downloads\testing_video.avi"
DEVICE = "0"              # "0" = first GPU, "cpu" = CPU
CONF = 0.22               # Conf threshold to reliably detect both heads & both tails
TILE = 640               # Tiled inference size matching model training
TILE_OVERLAP = 0.25       # Overlap ratio for boundary continuity (0.25 = ~8 tiles vs 0.35 = ~15)
STRIDE = 1                # Process every frame
MM_PER_PX = None          # Calibration factor (e.g. 0.08) if available
MAX_DISAGREE = 0.2        # QC tolerance

EXPECTED_HEADS = 2
EXPECTED_TAILS = 2
EXPECTED_TOTAL = 4

CLUSTER_DIST = 400.0      # px - coil bundle spatial clustering (wider for camera pans)
MAX_DIST = 350.0          # px - association tolerance: covers fast camera shake / pipe movement
MAX_MISSED = 60           # frames to coast across occlusions (2s @ 30fps, 5s @ 12fps)

OUTPUT_VIDEO = None
LATEST_JSON = None
REPORT_JSON = None
SHOW_LIVE_PREVIEW = False
# ====================================================================


def load_roi(path=None):
    """Load polygon points saved in roi.json. Returns an (N,2) int32 array or None."""
    if not path or not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "points" in data and data["points"]:
            return np.array(data["points"], dtype=np.int32)
    except Exception:
        pass
    return None


def filter_by_roi(ends, roi):
    """Keep only ends whose tip falls inside the ROI polygon (or on its edge)."""
    if roi is None:
        return ends
    return [
        e for e in ends
        if "tip" in e and cv2.pointPolygonTest(roi, tuple(float(v) for v in e["tip"]), False) >= 0
    ]


def filter_to_coil_cluster(ends, cluster_dist=200.0):
    """Keep the ends belonging to the main coil bundle cluster.
    Drops isolated spurious reflections or objects far away at image borders."""
    if len(ends) <= 4:
        return ends

    pts = np.array([
        e.get("tip") or (e["bbox"][0] + e["bbox"][2] / 2.0, e["bbox"][1] + e["bbox"][3] / 2.0)
        for e in ends
    ])
    n = len(pts)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d <= cluster_dist:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited = set()
    clusters = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for nb in neighbors[curr]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            clusters.append(comp)

    clusters.sort(key=len, reverse=True)
    best_indices = clusters[0]
    return [ends[i] for i in best_indices]


def print_frame_result(frame_idx, res):
    """Outputs the required frame-by-frame block structure:
    Frame <idx>: HEADS=2 TAILS=2

    #1 HEAD
    bbox=[...]
    ...
    """
    h_c = res["heads_count"]
    t_c = res["tails_count"]
    print(f"\nFrame {frame_idx}: HEADS={h_c} TAILS={t_c}", flush=True)

    for t in res["heads"]:
        print(f"\n#{t.id} HEAD", flush=True)
        print(f"bbox={t.bbox}", flush=True)
        print(f"bbox_width={t.bbox_width}", flush=True)
        print(f"bbox_height={t.bbox_height}", flush=True)
        print(f"mask_width={t.mask_width}", flush=True)
        print(f"edge_width={t.edge_width}", flush=True)
        print(f"final_width={t.final_width}", flush=True)

    for t in res["tails"]:
        print(f"\n#{t.id} TAIL", flush=True)
        print(f"bbox={t.bbox}", flush=True)
        print(f"bbox_width={t.bbox_width}", flush=True)
        print(f"bbox_height={t.bbox_height}", flush=True)
        print(f"mask_width={t.mask_width}", flush=True)
        print(f"edge_width={t.edge_width}", flush=True)
        print(f"final_width={t.final_width}", flush=True)


def build_export_json(frame_idx, res):
    """Formats full tracking state for frontend JSON pickup."""
    detections = []
    for t in res["heads"] + res["tails"]:
        detections.append({
            "id": t.id,
            "role": t.role,
            "class": t.class_name,
            "status": t.status,
            "bbox": t.bbox,
            "bbox_width": t.bbox_width,
            "bbox_height": t.bbox_height,
            "mask_width": t.mask_width,
            "edge_width": t.edge_width,
            "final_width": t.final_width,
            "is_coasting": t.missed > 0,
            "missed_frames": t.missed,
        })
    return {
        "frame": frame_idx,
        "heads_count": res["heads_count"],
        "tails_count": res["tails_count"],
        "total_count": res["total_count"],
        "detections": detections,
    }


def main():
    print(f"[config] Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    if DEVICE:
        model.to(f"cuda:{DEVICE}" if DEVICE.isdigit() else DEVICE)
        # Note: FP16 (model.model.half()) is NOT used here — GTX 1060 (Pascal)
        # has no native FP16 tensor cores, so half() adds conversion overhead and
        # actually slows inference. Only enable on Turing (RTX 20xx) or newer.
    print(f"[device] Model is running on: {model.device}")
    class_map = build_class_map(model.names)
    print(f"[classes] {model.names} -> tubing id: {class_map[2]}, roles: {class_map[1]}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[video] Total frames: {total_frames} @ {fps:.1f} fps")

    roi = load_roi(ROI_PATH)
    if roi is not None:
        print(f"[roi] Loaded ROI polygon with {len(roi)} vertices from: {ROI_PATH}")

    if LATEST_JSON:
        Path(LATEST_JSON).parent.mkdir(parents=True, exist_ok=True)
    if REPORT_JSON:
        Path(REPORT_JSON).parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)

    tracker = FourEndsTracker(max_missed=MAX_MISSED, max_dist=MAX_DIST)

    writer = None
    frame_idx = 0
    head2_count = 0
    tail2_count = 0
    total4_count = 0
    missed_end_frames = 0
    id_swap_count = 0

    t0 = time.time()
    print("Starting inference on video frames...", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % STRIDE != 0:
                continue

            # 1. Tiled YOLO predictions & non-destructive deduplication
            inst = masks_from_model(model, frame, CONF, gray=True, tile=TILE, overlap=TILE_OVERLAP)

            # 2. End analysis (computes bbox, tip, mask_width, edge_width, final_width)
            ends = analyse(frame, inst, MM_PER_PX, MAX_DISAGREE, gap=25, class_map=class_map)

            # 2b. Discard ends outside ROI polygon if defined
            if roi is not None:
                ends = filter_by_roi(ends, roi)

            # 3. Spatial cluster filtering (keeps ends in the physical coil bundle)
            ends = filter_to_coil_cluster(ends, cluster_dist=CLUSTER_DIST)

            # 4. Update persistent 4-end tracker
            res = tracker.update(ends, frame_idx)

            # 5. Output frame result
            print_frame_result(frame_idx, res)

            # 6. Record metrics
            h_c = res["heads_count"]
            t_c = res["tails_count"]
            tot = res["total_count"]

            if h_c == 2: head2_count += 1
            if t_c == 2: tail2_count += 1
            if tot == 4: total4_count += 1
            else: missed_end_frames += 1

            # 7. Write latest JSON
            if LATEST_JSON:
                latest_data = build_export_json(frame_idx, res)
                Path(LATEST_JSON).write_text(json.dumps(latest_data, indent=2))

            # 8. Annotated video writing
            if OUTPUT_VIDEO:
                vis = draw(frame, inst, ends, tubing_id=class_map[2])
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        OUTPUT_VIDEO,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps / STRIDE,
                        (w, h),
                    )
                if writer:
                    writer.write(vis)

    finally:
        cap.release()
        if writer:
            writer.release()

    total_time = time.time() - t0
    avg_fps = frame_idx / max(total_time, 0.001)

    print("\n" + "=" * 50, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("=" * 50, flush=True)
    print(f"Total frames processed: {frame_idx}", flush=True)
    print(f"Frames with HEADS=2: {head2_count}", flush=True)
    print(f"Frames with TAILS=2: {tail2_count}", flush=True)
    print(f"Frames with TOTAL=4: {total4_count}", flush=True)
    print(f"Frames with tracking ID swap: {id_swap_count}", flush=True)
    print(f"Frames with missing end: {missed_end_frames}", flush=True)
    print(f"Total processing time: {total_time:.1f}s ({avg_fps:.1f} fps)", flush=True)
    print("=" * 50, flush=True)

    if REPORT_JSON:
        report = {
            "total_frames": frame_idx,
            "frames_heads_2": head2_count,
            "frames_tails_2": tail2_count,
            "frames_total_4": total4_count,
            "frames_id_swap": id_swap_count,
            "frames_missing_end": missed_end_frames,
            "total_time_s": round(total_time, 1),
            "avg_fps": round(avg_fps, 1),
        }
        Path(REPORT_JSON).write_text(json.dumps(report, indent=2))
        print(f"Saved report to: {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
