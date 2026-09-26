import json
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

# Patch skeletonize for the photo-CLI path (local_axis still uses it).
# Do NOT patch mfm.attach: measure_from_masks.attach() is now already
# local-crop optimised AND has a DT fallback for separated tubings.
mfm.skeletonize = _fast_skeletonize

build_class_map = mfm.build_class_map
masks_from_model = mfm.masks_from_model
analyse = mfm.analyse
draw = mfm.draw
pair_ends = mfm.pair_ends

try:
    from app.ml.tube.video_ends import Track, match, confirmed, size_labels
except (ImportError, ValueError):
    from video_ends import Track, match, confirmed, size_labels

# Default model and ROI paths resolved dynamically relative to this module
_TUBE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(_TUBE_DIR / "model" / "best.pt")
ROI_JSON = str(_TUBE_DIR / "head_zip" / "roi.json") if (_TUBE_DIR / "head_zip" / "roi.json").exists() else str(_TUBE_DIR / "roi.json")


def load_roi(path=None):
    """Load polygon points saved in roi.json. Returns an (N,2) int32 array or None."""
    target = None
    if path and Path(path).exists():
        target = Path(path)
    elif (_TUBE_DIR / "head_zip" / "roi.json").exists():
        target = _TUBE_DIR / "head_zip" / "roi.json"
    elif (_TUBE_DIR / "roi.json").exists():
        target = _TUBE_DIR / "roi.json"

    if not target or not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
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


def keep_largest_clusters(ends, cluster_dist=400.0, k=2):
    """Drop ends that are spatially isolated from the main physical tube groups."""
    if len(ends) <= k:
        return ends
    valid_ends = [e for e in ends if "tip" in e]
    if len(valid_ends) <= k:
        return ends

    pts = [np.asarray(e["tip"], float) for e in valid_ends]
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) <= cluster_dist:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    top = sorted(groups.values(), key=len, reverse=True)[:k]
    keep = set(i for group in top for i in group)
    return [valid_ends[i] for i in sorted(keep)]


class TubeAnalyzer:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "0",
        conf: float = 0.35,
        tile: int = 640,
        tile_overlap: float = 0.3,
        mm_per_px: Optional[float] = None,
        gray: bool = True,
        draw_overlay: bool = True,
        min_frames: int = 5,
        max_dist: float = 60.0,
        roi_path: Optional[str] = None,
        filter_isolated: bool = True,
        expected_tube_count: int = 2,
        cluster_dist: float = 400.0,
        min_ratio: float = 1.15,
        size_from: str = "tail",
    ):
        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            fallback = str(Path(__file__).resolve().parent.parent / "model" / "best.pt")
            if os.path.exists(fallback):
                path = fallback
            else:
                raise FileNotFoundError(f"Model not found at {path}")

        if YOLO is None:
            raise RuntimeError("Ultralytics / PyTorch failed to load. Check PyTorch installation.")

        self.model = YOLO(path)
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

        self.roi_path = roi_path or ROI_JSON
        self.roi = load_roi(self.roi_path)

        self.filter_isolated = filter_isolated
        self.expected_tube_count = expected_tube_count
        self.cluster_dist = cluster_dist
        self.min_ratio = min_ratio
        self.size_from = size_from

        self.tracks = []
        self.last_known = {"bigger_tube": None, "smaller_tube": None}

    def reset(self):
        """Reset tracking history for a fresh video or stream."""
        self.tracks = []
        self.last_known = {"bigger_tube": None, "smaller_tube": None}
        self.roi = load_roi(self.roi_path)

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple[Dict[str, Any], np.ndarray]:
        """Process a single frame directly on the original color frame."""
        if frame is None or frame.size == 0:
            return {}, frame

        _skel_cache.clear()
        h, w = frame.shape[:2]

        # 1. Tile inference with gray=True
        inst = masks_from_model(
            self.model,
            frame,
            conf=self.conf,
            gray=self.gray,
            tile=self.tile,
            overlap=self.tile_overlap,
        )

        # 2. Analyze masks to extract tips and widths
        ends = [
            e for e in analyse(frame, inst, self.mm_per_px, gap=3, class_map=self.class_map)
            if "tip" in e
        ]

        # 3. Spatial cluster filtering (drops isolated false detections)
        if self.filter_isolated:
            ends = keep_largest_clusters(ends, cluster_dist=self.cluster_dist, k=self.expected_tube_count)

        # 4. Whole-image tracking across frames (tracking does not drop ends outside ROI)
        match(self.tracks, ends, frame_idx, self.max_dist)

        # Prune stale tracks older than 90 frames to keep tracking fast and memory lean
        if len(self.tracks) > 20:
            self.tracks = [t for t in self.tracks if (frame_idx - getattr(t, 'last_seen', 0)) <= 90]

        # 5. Filter ends for display and reporting by ROI polygon
        ends_for_display = filter_by_roi(ends, self.roi)

        # 6. Build frame result dictionary matching run_video_frames.py
        eff_min_frames = min(self.min_frames, max(1, frame_idx))
        conf_tracks = confirmed(self.tracks, eff_min_frames)
        if not conf_tracks and self.tracks:
            conf_tracks = self.tracks

        sizes = size_labels(
            [t for t in conf_tracks if self.size_from == "all" or t.role == "TAIL"],
            self.min_ratio,
        )
        summaries = {t.id: t.summary(self.mm_per_px, sizes.get(t.id)) for t in conf_tracks}
        pair_ends(list(summaries.values()))

        detections = []
        for e in ends_for_display:
            tid = e.get("_track_id")
            s = summaries.get(tid)
            if s is None:
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
                "opaque": e.get("opaque", s.get("opaque", None)),
                "colour": e.get("colour", s.get("colour", None)),
                "is_coasting": False,
                "missed_frames": 0,
            })

        heads_count = sum(t.role == "HEAD" for t in conf_tracks)
        tails_count = sum(t.role == "TAIL" for t in conf_tracks)
        total_count = heads_count + tails_count

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
            "tails_count_raw": tails_count,
            "total_count": total_count,
            "detections": detections,
            "bigger_tube": bigger,
            "smaller_tube": smaller,
            "used_last_known_size": used_last_known,
            "video_width": w,
            "video_height": h,
        }

        if self.draw_overlay:
            vis = self.render_overlay(frame, result, inst=inst, tubing_id=self.class_map[2])
            if self.roi is not None and len(self.roi) > 2:
                cv2.polylines(vis, [self.roi], isClosed=True, color=(0, 255, 255), thickness=2)
            return result, vis

        return result, frame

    def render_overlay(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        inst=None,
        tubing_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Renders ML detections, non-overlapping quadrant callout badges, corner brackets,
        and high-contrast HUD banner directly onto the BGR color frame.
        """
        if frame is None or frame.size == 0:
            return frame

        vis = frame.copy()
        H, W = vis.shape[:2]

        # 1. Subtle segmentation masks if present
        if inst and tubing_id is not None:
            overlay = vis.copy()
            has_masks = False
            for c, m in inst:
                if c == tubing_id:
                    overlay[m] = (0, 180, 0)
                    has_masks = True
            if has_masks:
                cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, dst=vis)

        CYAN = (255, 200, 0)
        AMBER = (0, 165, 255)
        RED = (50, 50, 255)
        DARK_BG = (20, 20, 26)

        detections = result.get("detections", [])
        boxes_data = []

        for d in detections:
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            role = d.get("role", "TUBE")
            is_head = (role == "HEAD")
            color = CYAN if is_head else AMBER
            if d.get("status") == "CHECK":
                color = RED

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # High-tech corner brackets
            bracket_len = min(10, max(4, (x2 - x1) // 3, (y2 - y1) // 3))
            cv2.line(vis, (x1, y1), (x1 + bracket_len, y1), color, 3)
            cv2.line(vis, (x1, y1), (x1, y1 + bracket_len), color, 3)
            cv2.line(vis, (x2, y1), (x2 - bracket_len, y1), color, 3)
            cv2.line(vis, (x2, y1), (x2, y1 + bracket_len), color, 3)
            cv2.line(vis, (x1, y2), (x1 + bracket_len, y2), color, 3)
            cv2.line(vis, (x1, y2), (x1, y2 - bracket_len), color, 3)
            cv2.line(vis, (x2, y2), (x2 - bracket_len, y2), color, 3)
            cv2.line(vis, (x2, y2), (x2, y2 - bracket_len), color, 3)

            # Center target reticle dot
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis, (cx, cy), 3, (0, 0, 0), -1)
            cv2.circle(vis, (cx, cy), 2, color, -1)

            # Format badge text
            w_val = d.get("final_width") or d.get("width_px") or 0
            t_id = d.get("id", "")
            txt_parts = [f"#{t_id} {role}", f"{w_val:.1f}px"]
            if d.get("size"):
                txt_parts.append(f"[{d['size']}]")
            if d.get("pair_id"):
                txt_parts.append(f"T{d['pair_id']}")
            txt = " ".join(txt_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, font_thick)

            boxes_data.append({
                "d": d, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "color": color, "txt": txt,
                "tw": tw, "th": th, "font": font, "scale": font_scale, "thick": font_thick
            })

        # Compute cluster center for outward quadrant placement
        if boxes_data:
            avg_cx = sum(b["cx"] for b in boxes_data) / len(boxes_data)
            avg_cy = sum(b["cy"] for b in boxes_data) / len(boxes_data)
        else:
            avg_cx, avg_cy = W // 2, H // 2

        placed_badges = []
        pad = 4

        for b in boxes_data:
            tw, th = b["tw"], b["th"]
            dx = b["cx"] - avg_cx
            dy = b["cy"] - avg_cy

            if abs(dx) > abs(dy) * 0.7:
                # East or West
                if dx >= 0:
                    bx1 = b["x2"] + 8
                else:
                    bx1 = b["x1"] - tw - pad * 2 - 8
                by1 = b["cy"] - th // 2 - pad
            else:
                # North or South
                if dy >= 0:
                    by1 = b["y2"] + 8
                else:
                    by1 = b["y1"] - th - pad * 2 - 8
                bx1 = b["cx"] - tw // 2 - pad

            bx1 = max(4, min(bx1, W - tw - pad * 2 - 8))
            by1 = max(4, min(by1, H - th - pad * 2 - 8))
            bx2 = bx1 + tw + pad * 2
            by2 = by1 + th + pad * 2

            # Collision avoidance against earlier placed badges
            for _ in range(8):
                collision = False
                for pb in placed_badges:
                    if not (bx2 < pb["bx1"] or bx1 > pb["bx2"] or by2 < pb["by1"] or by1 > pb["by2"]):
                        collision = True
                        if dy >= 0:
                            by1 = pb["by2"] + 4
                        else:
                            by1 = pb["by1"] - (th + pad * 2) - 4
                        by2 = by1 + th + pad * 2
                        break
                if not collision:
                    break

            placed_badges.append({"bx1": bx1, "by1": by1, "bx2": bx2, "by2": by2, "b": b})

        # Draw stems and badges
        for pb in placed_badges:
            b = pb["b"]
            bx1, by1, bx2, by2 = pb["bx1"], pb["by1"], pb["bx2"], pb["by2"]
            color = b["color"]

            badge_cx = (bx1 + bx2) // 2
            badge_cy = (by1 + by2) // 2
            cv2.line(vis, (b["cx"], b["cy"]), (badge_cx, badge_cy), color, 1, cv2.LINE_AA)

            cv2.rectangle(vis, (bx1, by1), (bx2, by2), DARK_BG, -1)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 1)
            cv2.putText(vis, b["txt"], (bx1 + pad, by2 - pad - 1), b["font"], b["scale"], (255, 255, 255), b["thick"], cv2.LINE_AA)

        # Top HUD Banner
        heads_cnt = result.get("heads_count", 0)
        tails_cnt = result.get("tails_count", 0)
        tot_cnt = result.get("total_count", heads_cnt + tails_cnt)
        hud_txt = f"HEADS: {heads_cnt}  |  TAILS: {tails_cnt}  |  TOTAL: {tot_cnt}"
        if result.get("bigger_tube") and result.get("smaller_tube"):
            b_tube = result["bigger_tube"]
            s_tube = result["smaller_tube"]
            b_id = b_tube.get("pair_id", b_tube.get("id"))
            s_id = s_tube.get("pair_id", s_tube.get("id"))
            b_w = b_tube.get("final_width", 0)
            s_w = s_tube.get("final_width", 0)
            hud_txt += f"  |  BIGGER: T{b_id} ({b_w:.1f}px)  |  SMALLER: T{s_id} ({s_w:.1f}px)"

        (hw, hh), _ = cv2.getTextSize(hud_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (15, 15), (25 + hw + 24, 25 + hh + 10), DARK_BG, -1)
        hud_status_col = (0, 200, 0) if (heads_cnt == 2 and tails_cnt == 2) else (50, 50, 255)
        cv2.rectangle(vis, (15, 15), (25 + hw + 24, 25 + hh + 10), hud_status_col, 1)
        cv2.circle(vis, (30, 20 + hh // 2 + 2), 4, hud_status_col, -1)
        cv2.putText(vis, hud_txt, (42, 22 + hh), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        return vis

