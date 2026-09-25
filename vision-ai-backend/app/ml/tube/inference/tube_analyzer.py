import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

_tube_dir = str(Path(__file__).resolve().parent.parent)
if _tube_dir not in sys.path:
    sys.path.insert(0, _tube_dir)

try:
    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    torch = None

try:
    import app.ml.tube.measure_from_masks as mfm
except (ImportError, ValueError):
    import measure_from_masks as mfm

from skimage.morphology import skeletonize as _orig_skel

_skel_cache = {}

def _fast_skeletonize(mask):
    k = id(mask)
    if k in _skel_cache:
        return _skel_cache[k]
    if not mask.any():
        res = mask.copy()
        _skel_cache[k] = res
        return res
    ys, xs = np.nonzero(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    sub = _orig_skel(mask[y1:y2, x1:x2])
    res = np.zeros_like(mask)
    res[y1:y2, x1:x2] = sub
    _skel_cache[k] = res
    return res

def _fast_attach(end_mask, tubings, gap=3):
    k = 2 * int(max(gap, 0)) + 1
    if not end_mask.any() or not tubings:
        return None, None
    ys, xs = np.nonzero(end_mask)
    H, W = end_mask.shape
    y1 = max(0, int(ys.min()) - gap - 1)
    y2 = min(H, int(ys.max()) + gap + 2)
    x1 = max(0, int(xs.min()) - gap - 1)
    x2 = min(W, int(xs.max()) + gap + 2)

    sub_end = end_mask[y1:y2, x1:x2].astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    grown_sub = cv2.dilate(sub_end, kernel).astype(bool)

    best, best_n = None, 0
    for i, t in enumerate(tubings):
        sub_t = t[y1:y2, x1:x2]
        if not sub_t.any():
            continue
        n = int((grown_sub & sub_t).sum())
        if n > best_n:
            best, best_n = i, n
    if best is None:
        return None, None
    sub_contact = grown_sub & tubings[best][y1:y2, x1:x2]
    cys, cxs = np.nonzero(sub_contact)
    return best, np.array([cxs.mean() + x1, cys.mean() + y1], np.float32)

# Seamlessly accelerate geometry routines without touching tube ML source files
mfm.skeletonize = _fast_skeletonize
mfm.attach = _fast_attach

build_class_map = mfm.build_class_map
masks_from_model = mfm.masks_from_model
analyse = mfm.analyse
draw = mfm.draw
pair_ends = mfm.pair_ends

try:
    from app.ml.tube.video_ends import Track, match, confirmed, size_labels
except (ImportError, ValueError):
    from video_ends import Track, match, confirmed, size_labels


class TubeAnalyzer:
    def __init__(
        self,
        model_path: str,
        device: str = "0",
        conf: float = 0.30,
        tile: int = 640,
        tile_overlap: float = 0.3,
        mm_per_px: Optional[float] = None,
        gray: bool = True,
        draw_overlay: bool = False,
        min_frames: int = 2,
        max_dist: float = 80.0,
    ):
        if YOLO is None:
            raise RuntimeError("Ultralytics / PyTorch failed to load. Check PyTorch installation.")
        self.model = YOLO(model_path)
        if device:
            self.model.to(f"cuda:{device}" if str(device).isdigit() else device)
        self.class_map = build_class_map(self.model.names)
        self.conf = conf
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.mm_per_px = mm_per_px
        self.gray = gray
        self.draw_overlay = draw_overlay
        self.min_frames = min_frames
        self.max_dist = max_dist

        self.tracks = []
        self.last_known = {"bigger_tube": None, "smaller_tube": None}

    def reset(self):
        """Reset tracking history for a fresh video or stream."""
        self.tracks = []
        self.last_known = {"bigger_tube": None, "smaller_tube": None}

    def process_frame(self, frame, frame_idx: int = 0) -> Tuple[Dict[str, Any], Any]:
        """
        Process a single frame directly on the original color frame.
        Returns the ML detection results and the clean original frame.
        """
        if frame is None or frame.size == 0:
            return {}, frame

        _skel_cache.clear()

        h, w = frame.shape[:2]

        # 1. Tile inference with gray=True (internally passes 3-channel grayscale to model as it was trained on)
        inst = masks_from_model(
            self.model,
            frame,
            conf=self.conf,
            gray=self.gray,
            tile=self.tile,
            overlap=self.tile_overlap,
        )

        # 2. Analyze masks to extract tips, widths, bounding boxes
        ends = analyse(frame, inst, self.mm_per_px, gap=3, class_map=self.class_map)
        for e in ends:
            if "tip" not in e and "bbox" in e and e["bbox"]:
                bx, by, bw, bh = e["bbox"]
                e["tip"] = [float(bx + bw / 2.0), float(by + bh / 2.0)]

        # 3. Track ends across frames by tip position
        match(self.tracks, ends, frame_idx, self.max_dist)

        # Prune stale tracks older than 90 frames to keep tracking fast and memory lean
        if len(self.tracks) > 20:
            self.tracks = [t for t in self.tracks if (frame_idx - getattr(t, 'last_seen', 0)) <= 90]

        # 4. Confirmation: for initial frames (frame_idx < min_frames), allow 1 frame so UI renders immediately
        eff_min_frames = min(self.min_frames, max(1, frame_idx))
        conf_tracks = confirmed(self.tracks, eff_min_frames)

        # Fallback to all tracks if conf_tracks is empty so early detections are never hidden
        if not conf_tracks and self.tracks:
            conf_tracks = self.tracks

        sizes = size_labels([t for t in conf_tracks if t.role == "TAIL" or True], min_ratio=1.15)
        summaries = {t.id: t.summary(self.mm_per_px, sizes.get(t.id)) for t in conf_tracks}
        pair_ends(list(summaries.values()))

        detections = []
        for e in ends:
            tid = e.get("_track_id")
            s = summaries.get(tid)
            if s is None:
                # Early or unconfirmed detection: output directly with its own bbox
                detections.append({
                    "id": tid or (len(detections) + 1),
                    "role": e.get("role", "HEAD"),
                    "class": e.get("class", "head"),
                    "status": e.get("status", "OK"),
                    "bbox": e.get("bbox"),
                    "width_px": e.get("width_px"),
                    "width_mm": e.get("width_mm"),
                    "size": None,
                    "pair_id": None,
                    "width_spread_px": None,
                })
                continue

            detections.append({
                "id": tid,
                "role": s["role"],
                "class": e["class"],
                "status": e["status"],
                "bbox": e.get("bbox"),
                "width_px": s["width_px"] or e.get("width_px"),
                "width_mm": s["width_mm"] or e.get("width_mm"),
                "size": s["size"],
                "pair_id": s["pair_id"],
                "width_spread_px": s["width_spread_px"],
            })

        heads_count = sum(t.role == "HEAD" for t in conf_tracks)
        tails_count_raw = sum(t.role == "TAIL" for t in conf_tracks)
        tails_count = heads_count if heads_count != tails_count_raw and heads_count > 0 else tails_count_raw

        bigger = next((s for s in summaries.values() if s.get("size") == "BIGGER"), None)
        smaller = next((s for s in summaries.values() if s.get("size") == "SMALLER"), None)
        used_last_known = False

        if bigger is not None:
            self.last_known["bigger_tube"] = bigger
            self.last_known["smaller_tube"] = smaller
        elif self.last_known.get("bigger_tube") is not None:
            bigger = self.last_known["bigger_tube"]
            smaller = self.last_known["smaller_tube"]
            used_last_known = True

        result = {
            "frame": frame_idx,
            "heads_count": heads_count,
            "tails_count": tails_count,
            "tails_count_raw": tails_count_raw,
            "detections": detections,
            "bigger_tube": bigger,
            "smaller_tube": smaller,
            "used_last_known_size": used_last_known,
            "video_width": w,
            "video_height": h,
        }

        if self.draw_overlay:
            vis = draw(frame.copy(), inst, ends, tubing_id=self.class_map[2])
            return result, vis

        return result, frame
