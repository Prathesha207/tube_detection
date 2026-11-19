"""
run_video_frames.py - count HEADS and TAILS in every frame of a video, MEASURE
their WIDTH, say which tube is BIGGER, print it live in the terminal, write it
as JSON for a frontend, and save an annotated video.

    hta-frames --config config.yaml --video part_0.avi
    hta-frames --config config.yaml --video part_0.avi --output out.mp4 --json latest.json
    hta-frames --config config.yaml --video part_0.avi --mm-per-px 0.08
    python -m head_tail_analyzer.run_video_frames --config config.yaml --video part_0.avi

Where settings come from (later wins):
    built-in DEFAULTS  ->  config.yaml  ->  command-line flags
Relative paths INSIDE config.yaml are relative to the config file's folder, so
the config + model + roi.json can be copied to another machine as one folder.
Relative paths given ON THE COMMAND LINE are relative to the current folder.

From Python (e.g. a backend):
    from head_tail_analyzer.run_video_frames import load_config, run
    cfg = load_config("config.yaml", video="part_0.avi")
    for result in run(cfg):       # one dict per processed frame,
        ...                       # the same dict written to latest_json

CLASSES (matched by NAME, any number of them):
  * name contains "head" -> HEAD, "tail" -> TAIL, "tub" -> tube (several allowed)
    e.g. head_white, head_blue, tail_white, tail_blue, white_tube, blue_tube
  * class_conf: {class name: threshold} per class (wins over conf_head / conf_tail / conf)
  * each track votes its class over all frames (confidence-weighted) -> "class" in the JSON;
    the tube colour comes from that class name (color_source: auto/class/saturation)

How it works:
  * Heads and tails come from the model's boxes + masks (YOLO11-seg).
  * Tiled inference (tile px, tile_overlap) - the model was trained on tiles.
    Duplicates from overlapping tiles are merged.
  * Tracking: an end must be seen min_frames times before it counts, which
    filters one-off false detections.
  * ROI (roi_json): inference and tracking run on the WHOLE frame, but only
    detections whose center lies inside the polygon are drawn, written to
    JSON and counted.

WIDTH (width_method):
  * "tubing"  - model has a tubing class: the OUTER DIAMETER of the bare tube
                right behind each head/tail, from its two walls (sub-pixel).
                Head and tail widths then both mean "tube diameter".
  * "mask"    - model has only head/tail: the thickness of the head/tail mask
                itself (a head = connector, so its width is the connector's).
  * "auto"    - "tubing" if the model has a tubing class, else "mask".
  * "off"     - no width.
  Each end's width is the MEDIAN over every frame its track was measured, so
  it does not jitter from frame to frame.

BIGGER / SMALLER (per frame, confirmed ends inside the ROI):
  * heads are compared with heads, tails with tails -> size = BIGGER/SMALLER/SAME
    (split at the largest width gap; needs >= min_ratio, default 1.15 = 15 %).
  * bigger_tube = the widest end of the comparison group: tails only
    (width_from: tail, default - a tail is bare tube) or every end
    (width_from: all - fair only with width_method tubing).

TUBE COLOUR (tube_color, sat_threshold, color_min_v):
  * median HSV saturation inside each end's mask (eroded 2 px, pixels darker than
    color_min_v ignored - the black fitting and shadows have low saturation too).
  * colour = "blue" if the median saturation of the track's LAST color_window frames
    > sat_threshold, else "white". Short window: a mistake corrects itself quickly.
    Pick sat_threshold with check_tube_color.py. Always measured on the COLOUR frame,
    even with gray: True.
  * color_gate: a detection that is clearly blue is never matched to a white track
    (and the other way round), so two close heads of different colour cannot swap IDs.
    "Clearly" = at least color_margin away from sat_threshold; borderline detections
    may match any track.

3D / DEPTH (depth_video + camera_matrix, see depth3d.py):
  * depth_video = the 16-bit depth video recorded TOGETHER with `video`, aligned to
    the RGB camera (OAK: <name>_depth.mkv next to <name>_video.mp4; "auto" finds it).
    Read through ffmpeg, frame N of the depth = frame N of the RGB video.
  * per end: median X, Y, Z (metres, camera frame: X right, Y down, Z forward) of
    the depth pixels inside its mask (shrunk 2 px, edges are noisy); no mask or no
    valid depth there -> 7x7 patch at the box centre; nothing valid -> null.
  * xyz_m = median over the last depth_smooth frames of the track, xyz_now_m = this frame.
  * width_mm_depth = width_px * Z / fx: width in mm WITHOUT a ruler (mm_per_px).

OUTPUT VIDEO DRAWING:
  * line width and text size scale with the frame size (draw_scale overrides)
  * one colour per role, whatever the size: HEAD = cyan, TAIL = yellow
  * thin box + a compact callout card joined by a leader line:
        ID: 3  Head blue / Conf: 0.92 / Smaller  18.4 px
    (the px value = measured tube width, the number that decides Bigger/Smaller/Same)
  * the segmentation mask outline of each end is drawn too (draw_masks)
  * top-left info panel: FRAME n / total, FPS (processing speed), HEADS (cyan), TAILS (yellow)
    + (panel_details) one line per confirmed end (ID, role, colour, Big/Small/Equal, width, Z)
    and the bigger-tube result
  * show_cards: True = callout card next to every box (default); False = boxes only
    (+ a small ID tag so the panel lines can be matched to the boxes).
    Preview window: press c to toggle.

Per frame (terminal + JSON), all restricted to the ROI:
  heads_count / tails_count      confirmed ends visible in this frame
  heads_visible / tails_visible  same as *_count (kept for the frontend)
  heads_total / tails_total      distinct confirmed ends that ever appeared in the ROI
  detections                     id, role, confidence, bbox [x, y, w, h], center,
                                 width_px, width_mm, size, width_source, tube_color, sat
                                 + with depth: xyz_m, xyz_now_m, z_m, depth_source, width_mm_depth
                                 + polygon [[x, y], ...] (output_polygons)
  tubes                          (output_tubes) every white_tube / blue_tube in the frame:
                                 index, class, tube_color, conf, bbox [x, y, w, h],
                                 polygon (largest part), polygons (every visible part)
                                 + with track_tubes: id (stable), class = vote over frames,
                                 class_now (this frame), class_share, held (model lost it,
                                 last position shown for up to tube_hold frames)
  tubes_count / tube_counts      number of tubes, per class
  tubes_total / tube_totals      (track_tubes) distinct tubes seen so far, per class
  bigger_tube                    status, id, role, width_px, width_mm, ratio
  frame_total                    frames in the video (None for a webcam / stream)
  fps                            processing speed, frames per second (smoothed)
  video_fps                      the video's own frame rate
"""
import argparse
import gc
import json
import threading
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from skimage.morphology import skeletonize

try:
    from .measure_from_masks import attach, local_axis
    from .tube_ends import measure_width
    from . import depth3d
    from . import wrapper_trace
except ImportError:          # run as a plain script next to those files
    from measure_from_masks import attach, local_axis
    from tube_ends import measure_width
    import depth3d
    import wrapper_trace


class Profiler:
    """Milliseconds per pipeline stage, averaged since the last report (profile: true)."""

    def __init__(self):
        self.sum, self.n = {}, {}

    def add(self, name, seconds):
        self.sum[name] = self.sum.get(name, 0.0) + seconds
        self.n[name] = self.n.get(name, 0) + 1

    @contextmanager
    def time(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def report(self, reset=True):
        if not self.sum:
            return ""
        parts = [f"{k} {1000 * self.sum[k] / self.n[k]:.0f}" for k in self.sum]
        if reset:
            self.sum, self.n = {}, {}
        return "[profile ms/frame] " + "  ".join(parts)


PROF = Profiler()

DEFAULTS = {
    "model_path": None,       # trained YOLO11-seg weights (required)
    "video": None,            # video file, or "0" for webcam 0 (required)
    "device": "0",            # "0" = first GPU, "cpu" = CPU
    "conf": 0.35,
    "conf_head": None,        # per-role confidence; None = use conf (tails usually score lower)
    "conf_tail": None,
    "class_conf": None,       # per-CLASS confidence, wins over conf_head/conf_tail/conf, e.g.
                              # {head_white: 0.5, head_blue: 0.5, tail_white: 0.3, tail_blue: 0.3,
                              #  white_tube: 0.35, blue_tube: 0.35}
    "color_source": "auto",   # tube colour from: auto (class name if it says blue/white, else
                              # saturation) | class | saturation
    "max_heads": 0,           # most heads/tails drawn+counted per frame (best confidence wins); 0 = no limit
    "max_tails": 0,
    "iou": 0.5,               # NMS inside each tile
    "scale": 1.0,             # shrink the frame by this factor before inference (ends too BIG for the model -> 0.33)
    "imgsz": None,            # model input size; None = tile (or 640 if tile is 0). Keep = training imgsz
    "tile": 1280,             # tile size; match the training imgsz (0 = whole frame)
    "tile_overlap": 0.3,
    "merge_iou": 0.4,         # merge duplicate boxes from overlapping tiles...
    "merge_contain": 0.75,    # ...or when one box is mostly inside another
    "merge_gap": 12,          # px: same-role boxes this close are ONE end split in two (0 = off)
    "cross_role_contain": 0.5,  # a head/tail box this much inside the other role's box is dropped (0 = off)
    "gray": False,            # True only if the model was trained on grayscale images.
                              # Width measurement always uses grayscale internally, either way.
    "stride": 1,              # 1 = every frame
    "min_frames": 5,          # sightings before an end is confirmed and counted
    "min_frames_tail": None,  # same, for tails only (None = min_frames). Weak tails: try 2-3
    "max_dist": 60,           # px an end may move between frames and stay the same end
    "max_dist_tail": None,    # same, for tails only (None = max_dist)
    "max_missed": 15,         # frames a lost track can still be re-matched
    "roi_json": None,         # polygon file; None = whole frame
    "draw_roi": True,
    "output_video": None,     # annotated .mp4; None = don't write
    "latest_json": None,      # overwritten every frame; None = don't write
    "history_jsonl": None,    # one line per frame; None = don't write
    "summary_json": None,     # whole-video result (one entry per confirmed end), written at the end
    "output_polygons": True,  # JSON: "polygon" [[x, y], ...] on every head/tail detection
    "output_tubes": True,     # JSON: "tubes" list - every white_tube / blue_tube with bbox + polygon
    "polygon_epsilon": 2.0,   # px: polygon simplification (bigger = fewer points, smaller JSON)
    # ---- tubes: tracking so they do not flicker ----
    "tube_merge_any_class": True,  # a white_tube and a blue_tube on the SAME pixels = one tube (best conf wins)
    "track_tubes": True,      # follow each tube across frames: stable id, class by vote (no white/blue flicker)
    "tube_min_frames": 3,     # sightings before a tube is shown (drops one-frame ghosts); 1 = at once
    "tube_hold": 2,           # keep showing a tube this many frames after the model loses it ("held": true)
    "tube_match_iou": 0.2,    # overlap (of the grown masks) needed to call it the same tube as last frame
    "tube_match_grow": 15,    # px: masks are grown by this much before matching, so a thin tube that
                              # MOVES (bundle rotated / lifted) still matches its own track
                              # (raise for fast motion; 0 = exact masks)
    "print_frames": True,
    "show_preview": False,
    # ---- drawing (output video / preview) ----
    "draw_scale": None,       # size of lines + text; None = auto from frame size (1.0 at 1280 px)
    "box_fill": 0.2,          # translucent fill inside each confirmed box (0 = off, 0.4 = strong)
    "draw_masks": True,       # outline the segmentation mask of each end
    "draw_unconfirmed": True, # thin grey boxes for ends not yet confirmed
    "draw_legend": False,     # no longer drawn (kept so old config files still load)
    "show_cards": True,       # True = callout card (results) next to each box; False = boxes only
    "panel_details": True,    # list every confirmed end + bigger tube in the top-left panel
    "box_id_tag": True,       # with show_cards off: small ID tag on each box (matches the panel)
    "draw_tubes": True,       # outline every detected tube (blue / white) in the output video
    # ---- width / bigger tube ----
    "width_method": "auto",   # auto | tubing | mask | off  (see docstring)
    "width_from": "tail",     # which ends decide bigger_tube: tail | all
    "min_ratio": 1.15,        # width ratio needed to call one BIGGER (1.15 = 15 %)
    "mm_per_px": None,        # set for millimetres (ruler / ArUco calibration)
    "attach_gap": 3,          # px gap tolerated between an end mask and its tubing mask
    "max_disagree": 0.2,      # edge vs mask width disagreement before falling back to mask width
    # ---- depth / 3D (see depth3d.py) ----
    "depth_video": None,      # 16-bit depth video aligned to `video` (OAK *_depth.mkv); None = no 3D
    "camera_matrix": None,    # [[fx,0,cx],[0,fy,cy],[0,0,1]] of the RGB camera
    "calib_width": None,      # image size camera_matrix belongs to (None = frame size)
    "calib_height": None,
    "dist_coeffs": None,      # lens distortion list, None = ignore distortion
    "depth_scale": 0.001,     # raw depth unit -> metres (OAK = millimetres)
    "depth_min_mm": 100,      # raw values outside [min, max] = no depth
    "depth_max_mm": 10000,
    "depth_smooth": 15,       # frames in each track's median X/Y/Z (steadier than one frame)
    # ---- tube colour ----
    "tube_color": True,       # label each end blue / white from the saturation inside its mask
    "sat_threshold": 60,      # median HSV saturation above this = blue (pick with check_tube_color.py)
    "color_min_v": 60,        # ignore dark pixels (black fitting, shadows): low saturation too
    "color_window": 10,       # label = median of the last N frames of a track (fast correction)
    "color_gate": True,       # never match a clearly blue detection to a white track, or vice versa
    "color_margin": 15,       # a detection is "clearly" blue/white only this far from sat_threshold
    # ---- trace out of the wrapper (see wrapper_trace.py) ----
    "trace_px": None,         # follow each tube this many px (ORIGINAL frame px, along the tube)
                              # from where it leaves the wrapper -> X/Y/Z of the end. None/0 = off
    "trace_wrapper_gap": 4,   # px the wrapper mask is grown by (its edge is not the tube's start)
    "trace_min_area": 40,     # px: tube pieces smaller than this outside the wrapper are ignored
    "trace_center_weight": 4.0,  # how strongly the path keeps to the tube's centre line
    "trace_smooth": 9,        # path smoothing window (px) before measuring its length
    "trace_depth_radius": None,  # px around a point used for its depth; None = 0.6 x tube half-width
    "trace_path_step_px": 0,  # >0: also X/Y/Z every N px along the trace ("path_xyz")
    "draw_trace": True,       # draw the traces + end points in the output video
    "draw_wrapper": True,     # outline the wrapper in the output video
    # ---- speed ----
    "half": False,            # FP16 inference on GPU (about 1.5-2x faster model; check counts once)
    "prefetch": True,         # read the next RGB + depth frame in a background thread
    "profile": False,         # print ms per stage every 30 frames
}
PATH_KEYS = ("model_path", "video", "roi_json", "output_video", "latest_json", "history_jsonl",
             "summary_json", "depth_video")

ROLES = ("HEAD", "TAIL")
COLORS = {"HEAD": (255, 230, 0), "TAIL": (0, 220, 255)}   # BGR: cyan head, yellow tail (any size)
COL_BIG = (60, 230, 60)                                   # green = bigger tube
COL_PENDING = (170, 170, 170)
COL_SAME = (220, 90, 200)                                 # purple = tubes same size
TUBE_COLORS = {"blue": (255, 120, 20), "white": (235, 235, 235)}   # BGR tube outlines
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------- config
def _resolve(value, base):
    """Make a path absolute against `base`. Empty -> None, digits -> webcam index,
    "auto" stays "auto" (depth_video: find the pair next to the video)."""
    if value is None or str(value).strip() == "":
        return None
    s = str(value).strip()
    if s.isdigit() or s.lower() == "auto":
        return s
    p = Path(s).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path=None, **overrides):
    """DEFAULTS <- config.yaml <- overrides (None values in overrides are ignored).
    Returns a SimpleNamespace, e.g. cfg.conf, cfg.video."""
    cfg = dict(DEFAULTS)
    if path:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unknown = sorted(set(data) - set(DEFAULTS))
        if unknown:
            print(f"[config] ignoring unknown keys: {unknown}")
        for k, v in data.items():
            if k in DEFAULTS:
                cfg[k] = _resolve(v, path.parent) if k in PATH_KEYS else v
    for k, v in overrides.items():
        if k not in DEFAULTS:
            raise KeyError(f"unknown setting: {k}")
        if v is not None:
            cfg[k] = _resolve(v, Path.cwd()) if k in PATH_KEYS else v

    if not cfg["model_path"]:
        raise SystemExit("no model: set model_path in the config or pass --model")
    if not Path(cfg["model_path"]).exists():
        raise SystemExit(f"model not found: {cfg['model_path']}")
    if not cfg["video"]:
        raise SystemExit("no video: set video in the config or pass --video")
    if not cfg["video"].isdigit() and not Path(cfg["video"]).exists():
        raise SystemExit(f"video not found: {cfg['video']}")
    if cfg["depth_video"]:
        if cfg["depth_video"].lower() != "auto" and not Path(cfg["depth_video"]).exists():
            raise SystemExit(f"depth_video not found: {cfg['depth_video']}")
        if not cfg["camera_matrix"]:
            raise SystemExit("depth_video is set but camera_matrix is missing")
    if cfg["width_method"] is False:          # YAML reads an unquoted `off` as False
        cfg["width_method"] = "off"
    if cfg["width_method"] not in ("auto", "tubing", "mask", "off"):
        raise SystemExit(f"width_method must be auto/tubing/mask/off, got {cfg['width_method']}")
    if cfg["width_from"] not in ("tail", "all"):
        raise SystemExit(f"width_from must be tail/all, got {cfg['width_from']}")

    cfg["device"] = str(cfg["device"])
    cfg["tile"] = int(cfg["tile"] or 0)
    cfg["stride"] = max(int(cfg["stride"]), 1)
    cfg["box_fill"] = float(min(max(cfg["box_fill"] or 0, 0.0), 1.0))
    cfg["tube_color"] = bool(cfg["tube_color"])
    cfg["sat_threshold"] = float(cfg["sat_threshold"])
    cfg["color_min_v"] = int(cfg["color_min_v"])
    cfg["color_window"] = max(int(cfg["color_window"]), 1)
    cfg["color_gate"] = bool(cfg["color_gate"])
    cfg["color_margin"] = float(cfg["color_margin"])
    if cfg["class_conf"] is not None:
        if not isinstance(cfg["class_conf"], dict):
            raise SystemExit("class_conf must be a mapping, e.g. {head_white: 0.5, tail_blue: 0.3}")
        cfg["class_conf"] = {str(k): float(v) for k, v in cfg["class_conf"].items()}
    if cfg["color_source"] not in ("auto", "class", "saturation"):
        raise SystemExit(f"color_source must be auto/class/saturation, got {cfg['color_source']}")
    try:
        cfg["trace_px"] = float(cfg["trace_px"] or 0) or None
    except (TypeError, ValueError):
        raise SystemExit(f"trace_px must be a number of pixels, got {cfg['trace_px']}") from None
    return SimpleNamespace(**cfg)


# ------------------------------------------------------------------ detection
def role_ids_from_names(names):
    """{class_id: 'HEAD'/'TAIL'}, matched by NAME so any number of classes works:
    head / head_white / head_blue -> HEAD,  tail / tail_white / tail_blue -> TAIL.
    Tube classes (tubing, white_tube, blue_tube) and anything else are ignored here."""
    ids = {}
    for i, n in names.items():
        low = str(n).lower()
        if "tub" in low:
            continue
        if "head" in low:
            ids[int(i)] = "HEAD"
        elif "tail" in low:
            ids[int(i)] = "TAIL"
    missing = set(ROLES) - set(ids.values())
    if missing:
        raise SystemExit(f"model classes {names} have no {missing} class")
    return ids


def tubing_ids_from_names(names):
    """Every tube class (tubing, white_tube, blue_tube, ...) -> sorted list of ids."""
    return sorted(int(i) for i, n in names.items() if "tub" in str(n).lower())


def wrapper_ids_from_names(names):
    """Every wrapper class id (name contains 'wrap')."""
    return sorted(int(i) for i, n in names.items() if "wrap" in str(n).lower())


def mask_box(mask):
    """(x0, y0, x1, y1) of a bool mask, x1/y1 exclusive, or None. Faster than np.nonzero."""
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _crop(mask, box=None, pad=0):
    """(sub-mask, x0, y0) around the mask's pixels (box = mask_box result, if known)."""
    box = box or mask_box(mask)
    if box is None:
        return None, 0, 0
    H, W = mask.shape
    x0, y0 = max(box[0] - pad, 0), max(box[1] - pad, 0)
    x1, y1 = min(box[2] + pad, W), min(box[3] + pad, H)
    return mask[y0:y1, x0:x1], x0, y0


def tubing_id_from_names(names):
    """First tube class id or None (kept for old callers)."""
    ids = tubing_ids_from_names(names)
    return ids[0] if ids else None


def _as_id_list(tubing):
    """None / int / iterable of ints -> list of ints."""
    if tubing is None:
        return []
    if isinstance(tubing, (int, np.integer)):
        return [int(tubing)]
    return [int(i) for i in tubing]


def color_from_name(name):
    """'blue' / 'white' / 'black' if the class name says so, else None."""
    low = str(name or "").lower()
    for c in ("blue", "white", "black"):
        if c in low:
            return c
    return None


def class_conf_for(cfg, name, role=None):
    """Confidence threshold for one class: class_conf[name] -> conf_head/conf_tail -> conf."""
    cc = getattr(cfg, "class_conf", None) or {}
    if name in cc:
        return float(cc[name])
    if role == "HEAD" and cfg.conf_head is not None:
        return float(cfg.conf_head)
    if role == "TAIL" and cfg.conf_tail is not None:
        return float(cfg.conf_tail)
    return float(cfg.conf)


def conf_table(names, role_ids, tubing_ids, cfg):
    """{class name: threshold} for every class the pipeline uses - printed and sent to the frontend."""
    out = {}
    for i in list(role_ids) + _as_id_list(tubing_ids):
        n = names[i]
        out[n] = class_conf_for(cfg, n, role_ids.get(i))
    return out


def tile_origins(size, tile, overlap):
    last = max(size - tile, 0)
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, last + 1, step))
    if xs[-1] != last:
        xs.append(last)
    return xs


def box_overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0, 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter), inter / min(aa, ab)


def box_gap(a, b):
    """Pixels of empty space between two boxes (0 if they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def mask_overlap(ma, mb):
    """Like box_overlap, but on the actual mask pixels: (IoU, overlap / smaller mask)."""
    inter = int(np.logical_and(ma, mb).sum())
    if inter == 0:
        return 0.0, 0.0
    a, b = int(ma.sum()), int(mb.sum())
    return inter / (a + b - inter), inter / min(a, b)


def _det(role, conf, box, mask=None, cls_name=None):
    return {"role": role, "conf": conf, "xyxy": box, "mask": mask, "cls_name": cls_name,
            "cls_color": color_from_name(cls_name),
            "center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)}


def merge_duplicates(dets, cfg):
    """Clean up one frame's raw detections, in three steps:

    1. Same end seen twice (overlapping tiles, or the model firing twice on
       one object): same role and IoU >= merge_iou or one mostly inside
       the other (>= merge_contain) -> keep the most confident box.
       Compared on the MASK pixels when both have masks (two real ends that
       touch share almost no pixels, so they stay separate), else on the boxes.
    2. One end split into side-by-side pieces (e.g. a head reported as a left
       half and a right half): same role and at most merge_gap px apart ->
       replaced by ONE box (and one mask) around both. merge_gap = 0 turns this off.
    3. Wrong-role box on top of another end (a TAIL box sitting on a HEAD):
       different role and one box >= cross_role_contain inside the other ->
       keep only the more confident one. cross_role_contain = 0 turns this off.
    """
    # 1. duplicates
    kept = []
    for d in sorted(dets, key=lambda d: d["conf"], reverse=True):
        dup = None
        for k in kept:
            if k["role"] != d["role"]:
                continue
            if box_overlap(k.get("mbox_o") or k["xyxy"], d["xyxy"])[1] == 0:
                continue                      # boxes apart -> masks apart: skip the pixel test
            if k.get("mask") is not None and d.get("mask") is not None:
                iou, contain = mask_overlap(k["mask"], d["mask"])   # real pixels: touching tubes stay separate
            else:
                iou, contain = box_overlap(k["xyxy"], d["xyxy"])    # no masks -> fall back to boxes
            if iou >= cfg.merge_iou or contain >= cfg.merge_contain:
                dup = k
                break
        if dup is None:
            kept.append(d)
        elif dup.get("mask") is not None and d.get("mask") is not None:
            dup["mask"] = dup["mask"] | d["mask"]      # tile-cut halves of one mask -> whole mask
            a, b = dup.get("mbox_o") or dup["xyxy"], d["xyxy"]
            dup["mbox_o"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    # 2. split pieces -> union box (repeat until nothing changes)
    if cfg.merge_gap and cfg.merge_gap > 0:
        changed = True
        while changed:
            changed = False
            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    a, b = kept[i], kept[j]
                    if a["role"] == b["role"] and box_gap(a["xyxy"], b["xyxy"]) <= cfg.merge_gap:
                        box = [min(a["xyxy"][0], b["xyxy"][0]), min(a["xyxy"][1], b["xyxy"][1]),
                               max(a["xyxy"][2], b["xyxy"][2]), max(a["xyxy"][3], b["xyxy"][3])]
                        ma, mb = a.get("mask"), b.get("mask")
                        mask = ma | mb if (ma is not None and mb is not None) else (ma if ma is not None else mb)
                        best = a if a["conf"] >= b["conf"] else b      # class of the surer piece
                        kept[i] = _det(a["role"], max(a["conf"], b["conf"]), box, mask,
                                       best.get("cls_name"))
                        del kept[j]
                        changed = True
                        break
                if changed:
                    break

    # 3. wrong-role box on top of another end
    if cfg.cross_role_contain and cfg.cross_role_contain > 0:
        final = []
        for d in sorted(kept, key=lambda d: d["conf"], reverse=True):
            if any(k["role"] != d["role"] and
                   box_overlap(k["xyxy"], d["xyxy"])[1] >= cfg.cross_role_contain
                   for k in final):
                continue
            final.append(d)
        kept = final
    return kept


# ---------------------------------------------------------------------- width
def mask_thickness(mask):
    """Thickness of a blob across its own long axis, any rotation:
    2 x median distance-to-border along the skeleton's central part."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    y0, y1, x0, x1 = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
    sub = mask[y0:y1, x0:x1]
    dt = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(sub)
    if sk.any():
        vals = dt[sk]
        vals = vals[vals >= 0.7 * vals.max()]      # drop the short corner branches
        return float(2.0 * np.median(vals))
    return float(2.0 * dt.max())


# --------------------------------------------------------------------- colour
def mask_sat(mask, hsv, min_v, box=None):
    """(median saturation, median hue) inside a mask, or (None, None).
    Mask eroded 2 px (edges mix in background); pixels darker than min_v dropped
    (black fitting / shadows have low saturation and would vote 'white')."""
    if mask is None:
        return None, None
    box = box or mask_box(mask)
    if box is None:
        return None, None
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    er = cv2.erode(sub, np.ones((5, 5), np.uint8)).astype(bool)
    sub = er if er.sum() >= 20 else sub.astype(bool)
    px = hsv[y0:y1, x0:x1][sub]
    px = px[px[:, 2] >= min_v]
    if len(px) < 20:
        return None, None
    return float(np.median(px[:, 1])), float(np.median(px[:, 0]))


def tube_components(tub_masks):
    """All tubing masks of the frame -> one mask per connected tube piece
    (joins the pieces that tile borders cut apart)."""
    if not tub_masks:
        return []
    u = np.zeros_like(tub_masks[0])
    for m in tub_masks:
        u |= m
    n, lab = cv2.connectedComponents(u.astype(np.uint8))
    return [lab == i for i in range(1, n) if (lab == i).sum() > 50]


def tube_width_behind(end_mask, tubings, gray, cfg):
    """Outer diameter of the bare tube right behind an end (same method as
    measure_from_masks.analyse): edge walls if they agree with the mask, else
    the tubing mask width. None if the end touches no tubing."""
    ti, contact = attach(end_mask, tubings, cfg.attach_gap)
    if ti is None:
        return None
    tube = tubings[ti]
    ys, xs = np.nonzero(tube)
    x0, y0 = max(xs.min() - 5, 0), max(ys.min() - 5, 0)
    sub = tube[y0:ys.max() + 6, x0:xs.max() + 6]            # crop: skeletonize is slow on full frames
    off = np.array([x0, y0], np.float32)
    area_w = np.sqrt(sub.sum() / max(skeletonize(sub).sum(), 1))
    u, _, mask_w = local_axis(sub, contact - off, radius=4 * area_w + 5)
    if u is None or not mask_w:
        return None
    start = contact + u * max(0.5 * mask_w, 2)               # a little into bare tubing
    edge_w, _ = measure_width(gray, start - u * 10, start,
                              length=int(max(3 * mask_w, 15)),
                              search_px=int(0.5 * mask_w * 1.4) + 3,
                              refine=True, max_angle_deg=6, max_shift=3)
    if edge_w and abs(edge_w - mask_w) / mask_w <= cfg.max_disagree:
        return float(edge_w)
    return float(mask_w)


class TubeComp:
    """One connected tube piece of the frame, as a crop. The skeleton / distance map
    that the width measurement needs are made ONCE per frame and shared by every end
    touching this tube (the old code made them twice per end)."""

    PAD = 5                       # same padding as the old full-frame crop

    def __init__(self, lab, i, stats, shape):
        H, W = shape
        x, y, w, h = (int(v) for v in stats[i, :4])
        self.x0, self.y0 = max(x - self.PAD, 0), max(y - self.PAD, 0)
        self.x1, self.y1 = min(x + w + self.PAD, W), min(y + h + self.PAD, H)
        self.sub = lab[self.y0:self.y1, self.x0:self.x1] == i
        self._geo = None

    def geometry(self):
        """(skeleton points after the corner-branch filter, distance map, area width)."""
        if self._geo is None:
            sk = skeletonize(self.sub)
            dt = cv2.distanceTransform(self.sub.astype(np.uint8), cv2.DIST_L2, 5)
            area_w = np.sqrt(self.sub.sum() / max(int(sk.sum()), 1))
            if sk.any():
                sk &= dt >= 0.7 * dt[sk].max()
            ys, xs = np.nonzero(sk)
            self._geo = (np.stack([xs, ys], 1).astype(np.float32), dt, area_w)
        return self._geo


def tube_comps(tub_masks):
    """Like tube_components(), but returns TubeComp crops (no full-frame mask per piece)."""
    if not tub_masks:
        return []
    u = np.zeros(tub_masks[0].shape, np.uint8)
    for m in tub_masks:
        u |= m.view(np.uint8) if m.dtype == bool else m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    return [TubeComp(lab, i, stats, u.shape) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] > 50]


def _attach_comp(end_mask, box, comps, gap):
    """Same answer as measure_from_masks.attach(), computed on crops."""
    g = int(max(gap, 0))
    H, W = end_mask.shape
    x0, y0 = max(box[0] - g - 1, 0), max(box[1] - g - 1, 0)
    x1, y1 = min(box[2] + g + 1, W), min(box[3] + g + 1, H)
    grown = cv2.dilate(end_mask[y0:y1, x0:x1].astype(np.uint8),
                       np.ones((2 * g + 1, 2 * g + 1), np.uint8)).astype(bool)
    best, best_n, best_pts = None, 0, None
    for c in comps:
        ix0, iy0, ix1, iy1 = max(x0, c.x0), max(y0, c.y0), min(x1, c.x1), min(y1, c.y1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        hit = grown[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] & c.sub[iy0 - c.y0:iy1 - c.y0, ix0 - c.x0:ix1 - c.x0]
        n = int(hit.sum())
        if n > best_n:
            ys, xs = np.nonzero(hit)
            best, best_n, best_pts = c, n, (xs + ix0, ys + iy0)
    if best is None:
        return None, None
    return best, np.array([best_pts[0].mean(), best_pts[1].mean()], np.float32)


def _local_axis_comp(comp, contact_sub, radius):
    """Same as measure_from_masks.local_axis(), using the cached skeleton / distance map."""
    pts, dt, _ = comp.geometry()
    if len(pts) < 3:
        return None, None
    near = pts[np.linalg.norm(pts - contact_sub, axis=1) < radius]
    if len(near) < 3:
        near = pts[np.argsort(np.linalg.norm(pts - contact_sub, axis=1))[:10]]
    c = near.mean(0)
    _, _, vt = np.linalg.svd(near - c)
    u = vt[0]
    if np.dot(c - contact_sub, u) < 0:
        u = -u
    mask_width = 2.0 * float(np.median(dt[near[:, 1].astype(int), near[:, 0].astype(int)]))
    return u.astype(np.float32), mask_width


def measure_end_fast(mask, box, comps, gray, cfg):
    """measure_end() for TubeComp crops: (width_px, source), same numbers, much faster."""
    if mask is None or box is None:
        return None, None
    if comps:
        comp, contact = _attach_comp(mask, box, comps, cfg.attach_gap)
        if comp is not None:
            off = np.array([comp.x0, comp.y0], np.float32)
            area_w = comp.geometry()[2]
            u, mask_w = _local_axis_comp(comp, contact - off, 4 * area_w + 5)
            if u is not None and mask_w:
                start = contact + u * max(0.5 * mask_w, 2)
                edge_w, _ = measure_width(gray, start - u * 10, start,
                                          length=int(max(3 * mask_w, 15)),
                                          search_px=int(0.5 * mask_w * 1.4) + 3,
                                          refine=True, max_angle_deg=6, max_shift=3)
                if edge_w and abs(edge_w - mask_w) / mask_w <= cfg.max_disagree:
                    return float(edge_w), "tube"
                return float(mask_w), "tube"
    sub, _, _ = _crop(mask, box, pad=2)
    w = mask_thickness(sub) if sub is not None else None
    return (w, "end_mask") if w else (None, None)


def measure_end(mask, tubings, gray, cfg):
    """(width_px, source) for one end, in the coordinates of `mask`."""
    if mask is None or not mask.any():
        return None, None
    if tubings:
        w = tube_width_behind(mask, tubings, gray, cfg)
        if w:
            return w, "tube"
    w = mask_thickness(mask)
    return (w, "end_mask") if w else (None, None)


def _contours(mask, box=None):
    """External contours of a mask, found on a small crop, in full-frame coordinates."""
    if mask is None:
        return []
    sub, x0, y0 = _crop(mask, box, pad=1)
    if sub is None:
        return []
    cs, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                             offset=(x0, y0))
    return cs


def mask_outline(mask, k, box=None):
    """Largest outer contour of a mask, back in ORIGINAL pixel coordinates."""
    cs = _contours(mask, box)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).astype(np.float32)
    return np.round(c / k).astype(np.int32)


def mask_polygons(mask, k, epsilon=2.0, min_area=30, box=None):
    """Every separate part of a mask as a simplified polygon [[x, y], ...] in ORIGINAL
    pixels (JSON-ready ints), largest part first. A tube hidden in the middle by
    another tube comes out as 2+ parts."""
    cs = _contours(mask, box)
    if not cs:
        return []
    out = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        p = cv2.approxPolyDP(c, max(float(epsilon), 0.0), True) if epsilon else c
        p = np.round(p.reshape(-1, 2).astype(np.float32) / k).astype(int)
        if len(p) >= 3:
            out.append(p.tolist())
    return out


def mask_bbox(mask, k, box=None):
    """[x, y, w, h] of a mask in ORIGINAL pixels."""
    bx0, by0, bx1, by1 = box or mask_box(mask)
    x1, y1, x2, y2 = bx0 / k, by0 / k, bx1 / k, by1 / k
    return [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]


def merge_tubes(tubes, cfg):
    """Same tube seen twice (overlapping tiles, double firing) -> one tube, masks joined,
    best confidence (and its class) kept. With tube_merge_any_class a white_tube and a
    blue_tube on the same pixels also count as one tube - the model was just unsure."""
    any_cls = getattr(cfg, "tube_merge_any_class", True)
    kept = []
    for t in sorted(tubes, key=lambda t: t["conf"], reverse=True):
        dup = next((k for k in kept if (any_cls or k["cls_name"] == t["cls_name"]) and
                    ("xyxy" not in k or box_overlap(k["xyxy"], t["xyxy"])[1] > 0) and
                    (lambda o: o[0] >= cfg.merge_iou or o[1] >= cfg.merge_contain)(
                        mask_overlap(k["mask"], t["mask"]))), None)
        if dup is None:
            kept.append(t)
        else:
            dup["mask"] = dup["mask"] | t["mask"]
            if "xyxy" in dup and "xyxy" in t:
                a, b = dup["xyxy"], t["xyxy"]
                dup["xyxy"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
            if dup["cls_name"] != t["cls_name"]:      # remember the losing class for the vote
                dup.setdefault("alt", {})[t["cls_name"]] = max(
                    dup.get("alt", {}).get(t["cls_name"], 0.0), t["conf"])
    return kept


TUBE_MATCH_SCALE = 4      # tubes are matched across frames on a 1/4-size mask (fast, plenty exact)


def tube_record(t, k, cfg):
    """One tube for the JSON: class, colour, conf, bbox [x, y, w, h], polygon(s).
    '_small' (a 1/4-size mask, for matching across frames) and '_alt' are removed
    before the JSON is written."""
    box = t.get("mbox") or mask_box(t["mask"])
    polys = mask_polygons(t["mask"], k, cfg.polygon_epsilon, box=box)
    m = t["mask"].view(np.uint8) if t["mask"].dtype == bool else t["mask"].astype(np.uint8)
    small = cv2.resize(m, (max(m.shape[1] // TUBE_MATCH_SCALE, 1), max(m.shape[0] // TUBE_MATCH_SCALE, 1)),
                       interpolation=cv2.INTER_AREA) > 0
    return {"class": t["cls_name"], "tube_color": color_from_name(t["cls_name"]),
            "conf": round(float(t["conf"]), 3), "bbox": mask_bbox(t["mask"], k, box),
            "polygon": polys[0] if polys else None,      # largest visible part
            "polygons": polys,                            # every visible part (tube hidden in the middle -> 2+)
            "_small": small, "_alt": t.get("alt", {})}


class TubeTrack:
    """One physical tube followed across frames: its class is a confidence-weighted
    vote over every frame (so one unsure frame cannot flip white <-> blue)."""

    def __init__(self, tid, rec, idx):
        self.id = tid
        self.hits = 0
        self.votes = {}
        self.center = None
        self.vel = np.zeros(2, np.float32)      # px (1/4 scale) per frame, smoothed
        self.update(rec, idx)

    def update(self, rec, idx):
        ys, xs = np.nonzero(rec["_small"])
        c = np.array([xs.mean(), ys.mean()], np.float32) if len(xs) else self.center
        if self.center is not None and c is not None:
            step = (c - self.center) / max(idx - self.last_seen, 1)
            self.vel = step if not self.vel.any() else 0.6 * step + 0.4 * self.vel
        self.center = c
        self.hits += 1
        self.last_seen = idx
        self.small = rec["_small"]
        self.grown = rec.get("_grown", rec["_small"])
        self.rec = rec
        self.votes[rec["class"]] = self.votes.get(rec["class"], 0.0) + rec["conf"]
        for cl, s in (rec.get("_alt") or {}).items():         # the class it lost to in this frame
            self.votes[cl] = self.votes.get(cl, 0.0) + 0.5 * s

    def predicted(self, idx):
        """The grown mask moved to where the tube should be NOW (constant velocity)."""
        d = self.vel * (idx - self.last_seen)
        if abs(d[0]) < 0.5 and abs(d[1]) < 0.5:
            return self.grown
        M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
        h, w = self.grown.shape
        return cv2.warpAffine(self.grown.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0

    @property
    def cls(self):
        return max(self.votes, key=self.votes.get)

    @property
    def share(self):
        return round(self.votes[self.cls] / sum(self.votes.values()), 3)


class TubeTracker:
    """Matches this frame's tubes to the tubes of earlier frames by mask overlap.
    update() returns the tubes to OUTPUT: confirmed ones (seen >= tube_min_frames) with
    a stable id and voted class, plus ones lost for <= tube_hold frames ('held': true)."""

    def __init__(self):
        self.tracks = []
        self.gone = []            # tracks lost for good (still counted in totals)
        self.next_id = 1

    def update(self, recs, idx, cfg):
        min_frames = max(int(getattr(cfg, "tube_min_frames", 3) or 1), 1)
        hold = max(int(getattr(cfg, "tube_hold", 2) or 0), 0)
        thr = float(getattr(cfg, "tube_match_iou", 0.2))
        grow_px = float(getattr(cfg, "tube_match_grow", 15) or 0)
        ks = 2 * int(np.ceil(grow_px / TUBE_MATCH_SCALE)) + 1 if grow_px > 0 else 1
        kernel = np.ones((ks, ks), np.uint8)
        for r in recs:                         # grown mask: a moved thin tube still overlaps itself
            r["_grown"] = (cv2.dilate(r["_small"].astype(np.uint8), kernel).astype(bool)
                           if ks > 1 else r["_small"])
        live = []
        for t in self.tracks:                  # lost long ago -> stop matching against it
            (live if idx - t.last_seen <= max(hold, 1) + 30 else self.gone).append(t)
        self.tracks = live
        gate = bool(getattr(cfg, "color_gate", True))
        pairs = []
        for ti, t in enumerate(self.tracks):
            pred = t.predicted(idx)
            for ri, r in enumerate(recs):
                if pred.shape != r["_grown"].shape:
                    continue
                # where it should be (moving on) OR where it was (stopped / turned back): best of both
                iou, contain = max(mask_overlap(pred, r["_grown"]), mask_overlap(t.grown, r["_grown"]))
                tc, rc = color_from_name(t.cls), r["tube_color"]
                if (gate and t.hits >= 2 and tc and rc and tc != rc and t.share >= 0.7
                        and iou < 0.6):
                    continue        # other colour BESIDE a sure track = its neighbour, not this tube
                                    # (other colour on the SAME spot = model unsure -> still matches)
                if iou >= thr or contain >= 0.8:
                    same = r["class"] == t.cls       # neighbours touch: prefer the same colour
                    pairs.append((iou + (0.5 if same else 0.0), ti, ri))
        used_t, used_r = set(), set()
        for _, ti, ri in sorted(pairs, reverse=True):
            if ti in used_t or ri in used_r:
                continue
            used_t.add(ti)
            used_r.add(ri)
            self.tracks[ti].update(recs[ri], idx)
        for ri, r in enumerate(recs):
            if ri not in used_r:
                self.tracks.append(TubeTrack(self.next_id, r, idx))
                self.next_id += 1
        out = []
        seen_now = [t for t in self.tracks if t.last_seen == idx]
        for t in self.tracks:
            missed = idx - t.last_seen
            if t.hits < min_frames or missed > hold:
                continue
            if missed and any(mask_overlap(t.grown, s.grown)[1] >= 0.3 for s in seen_now):
                continue                       # a tube seen NOW sits where this one was: no ghost
            rec = {k: v for k, v in t.rec.items() if not k.startswith("_")}
            rec.update({"id": t.id, "class_now": t.rec["class"] if missed == 0 else None,
                        "class": t.cls, "class_share": t.share,
                        "tube_color": color_from_name(t.cls), "held": missed > 0})
            out.append(rec)
        return out

    def totals(self, min_frames, tube_classes=()):
        """Distinct confirmed tubes seen so far, per voted class."""
        tot = {n: 0 for n in tube_classes}
        for t in self.tracks + self.gone:
            if t.hits >= min_frames:
                tot[t.cls] = tot.get(t.cls, 0) + 1
        return tot


def plain_tubes(recs):
    """Tube records without tracking (track_tubes off): just drop the private keys."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs]


def tube_in_roi(tube, roi):
    """A tube counts as inside the ROI if any point of its outline is inside."""
    if roi is None:
        return True
    return any(cv2.pointPolygonTest(roi, (float(x), float(y)), False) >= 0
               for poly in tube["polygons"] for x, y in poly)


def detect(model, frame, role_ids, cfg, tubing_id=None, lock=None, depth=None, names=None,
           tubes_out=None, info=None):
    """lock: held only around model.predict (a YOLO model must not run two
    predictions at once); the post-processing below runs without it.
    depth: uint16 depth frame aligned to `frame` (or None) -> d["xyz"] per end.
    tubing_id: one id or a list of ids (white_tube + blue_tube).
    names: {id: class name}; default model.names. Each class is kept only above
    its own threshold (class_conf_for).
    tubes_out: a list -> filled with one tube_record() per detected tube (output_tubes);
    with trace_px each record also gets "wrapper_trace" (see wrapper_trace.py).
    info: a dict -> info["wrapper"] = the wrapper record (or None) when tracing."""
    t_start = time.perf_counter()
    tubing_ids = set(_as_id_list(tubing_id))
    if names is None:
        names = dict(getattr(model, "names", {}) or {})
    H0, W0 = frame.shape[:2]                       # original size (before scale)
    if depth is not None:
        depth = depth3d.align_to_frame(depth, W0, H0)
    color_src = frame                              # colour copy: gray mode would make saturation 0
    if cfg.gray:
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    k = float(cfg.scale or 1.0)
    if k != 1.0:
        frame = cv2.resize(frame, None, fx=k, fy=k,
                           interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    h, w = frame.shape[:2]
    t = cfg.tile
    if t and (h > t or w > t):
        crops, offsets = [], []
        for y in tile_origins(h, t, cfg.tile_overlap):
            for x in tile_origins(w, t, cfg.tile_overlap):
                crops.append(frame[y:y + t, x:x + t])
                offsets.append((x, y))
    else:
        crops, offsets = [frame], [(0, 0)]

    want_trace = bool(getattr(cfg, "trace_px", None)) and tubes_out is not None
    wrapper_ids = set(wrapper_ids_from_names(names)) if want_trace else set()
    want_width = cfg.width_method != "off"
    want_color = getattr(cfg, "tube_color", False)
    want_tubes = (bool(getattr(cfg, "output_tubes", False)) or want_trace) and tubes_out is not None
    want_polys = bool(getattr(cfg, "output_polygons", False))
    want_masks = (want_width or cfg.draw_masks or depth is not None or want_color
                  or want_tubes or want_polys)
    classes = (list(role_ids) + (sorted(tubing_ids) if (want_width or want_tubes) else [])
               + sorted(wrapper_ids))
    thr = {c: class_conf_for(cfg, names.get(c, str(c)), role_ids.get(c)) for c in classes}
    t_pred = time.perf_counter()
    with (lock or nullcontext()):
        results = model.predict(crops, conf=min(thr.values()), iou=cfg.iou,
                                classes=classes, imgsz=int(cfg.imgsz or t or 640),
                                device=cfg.device, retina_masks=want_masks,
                                half=bool(getattr(cfg, "half", False)), verbose=False)
    PROF.add("predict", time.perf_counter() - t_pred)
    t_post = time.perf_counter()
    dets, tub_masks, tubes, wraps = [], [], [], []
    for r, (ox, oy) in zip(results, offsets):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        # threshold ON THE GPU, then copy bool (1 byte/px) instead of float32 (4 bytes/px)
        ms = (r.masks.data > 0.5).cpu().numpy() if (want_masks and r.masks is not None) else None
        for j, (b, c, s) in enumerate(zip(xyxy, cls, conf)):
            full = None
            if ms is not None:                    # tile mask -> full (scaled) frame
                m = ms[j]
                if ox == 0 and oy == 0 and m.shape == (h, w):
                    full = m                      # whole-frame inference: already full size
                else:
                    th, tw = min(m.shape[0], h - oy), min(m.shape[1], w - ox)
                    full = np.zeros((h, w), bool)
                    full[oy:oy + th, ox:ox + tw] = m[:th, :tw]
            box_s = [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]
            if c in wrapper_ids:
                if full is not None and s >= thr.get(c, cfg.conf) and full.any():
                    wraps.append((float(s), full))
                continue
            if c in tubing_ids:
                if full is not None and s >= thr.get(c, cfg.conf) and full.any():
                    tub_masks.append(full)
                    tubes.append({"cls_name": names.get(c, str(c)), "conf": float(s), "mask": full,
                                  "xyxy": box_s})
                continue
            if c not in role_ids or s < thr[c]:
                continue
            box = [v / k for v in box_s]          # back to original pixels
            dets.append(_det(role_ids[c], float(s), box, full, names.get(c, str(c))))
    dets = merge_duplicates(dets, cfg)
    for d in dets:                                # one bbox scan per mask, reused below
        d["mbox"] = mask_box(d["mask"]) if d.get("mask") is not None else None
    if want_tubes:
        merged = merge_tubes(tubes, cfg)
        for tb in merged:
            tb["mbox"] = mask_box(tb["mask"])
        wrapper = None
        if want_trace:
            gap = float(getattr(cfg, "trace_wrapper_gap", 4) or 0) * k
            wrapper = wrapper_trace.prepare_wrapper([m for _, m in wraps], gap)
            if info is not None:
                info["wrapper"] = wrapper_trace.wrapper_record(
                    wrapper, k, max(s for s, _ in wraps) if wraps else None)
        cam = depth3d.camera_for(cfg, W0, H0) if (depth is not None and want_trace) else None
        all_traces = []
        for tb in merged:
            rec = tube_record(tb, k, cfg)
            if want_trace:
                rec["wrapper_trace"] = wrapper_trace.trace_tube(tb["mask"], wrapper, cfg, k, depth,
                                                                cam, tb["mbox"])
                all_traces += rec["wrapper_trace"]
            tubes_out.append(rec)
        if want_trace:
            wrapper_trace.set_sides(all_traces, wrapper, k)
            for tr in all_traces:
                tr.pop("_out", None)
    elif info is not None:
        info["wrapper"] = None

    if want_width:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        comps = tube_comps(tub_masks)
        for d in dets:
            wpx, src = measure_end_fast(d.get("mask"), d.get("mbox"), comps, gray, cfg)
            d["width_px"] = wpx / k if wpx else None          # back to original pixels
            d["width_src"] = src
    if want_color:                                             # colour, while the masks still exist
        src = color_src if k == 1.0 else cv2.resize(color_src, (w, h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        for d in dets:
            d["sat"], d["hue"] = (mask_sat(d.get("mask"), hsv, cfg.color_min_v, d.get("mbox"))
                                  if d.get("mbox") else (None, None))
            d["sat_thr"] = cfg.sat_threshold
            d["color_window"] = cfg.color_window
    for d in dets:
        d["color_source"] = getattr(cfg, "color_source", "auto")
    if depth is not None:                                      # 3D, while the masks still exist
        cam = depth3d.camera_for(cfg, W0, H0)
        for d in dets:
            p = depth3d.end_xyz(depth, cam, d.get("mask") if d.get("mbox") else None, k,
                                d["center"], cfg, box=d.get("mbox"))
            d["xyz"] = p["xyz"] if p else None
            d["xyz_src"] = p["src"] if p else None
            d["fx"] = cam.fx
    for d in dets:
        d["smooth"] = cfg.depth_smooth
        m = d.pop("mask", None)                                # free memory, keep only the outline
        mb = d.pop("mbox", None)
        d["outline"] = mask_outline(m, k, mb) if (cfg.draw_masks and mb) else None
        if want_polys:
            polys = mask_polygons(m, k, cfg.polygon_epsilon, box=mb) if mb else []
            d["polygon"] = polys[0] if polys else None
    PROF.add("post", time.perf_counter() - t_post)
    PROF.add("pre", t_pred - t_start)
    return dets


# ------------------------------------------------------------------- tracking
class Track:
    def __init__(self, tid, det, idx, min_frames):
        self.id = tid
        self.role = det["role"]
        self.first_seen = idx
        self.first_center = det["center"]
        self.min_frames = min_frames
        self.hits = 0
        self.in_roi_ever = False
        self.widths = deque(maxlen=300)
        self.width_src = None
        self.xyzs = deque(maxlen=int(det.get("smooth") or 15))
        self.xyz_now = None
        self.fx = None
        self.sats = deque(maxlen=int(det.get("color_window") or 10))   # last N frames only
        self.sat_thr = det.get("sat_thr")
        self.cls_votes = {}                  # {class name: summed confidence} over the whole track
        self.color_source = det.get("color_source", "auto")
        self.update(det, idx)

    def update(self, det, idx):
        self.center = det["center"]
        self.xyxy = det["xyxy"]
        self.conf = det["conf"]
        self.last_seen = idx
        self.hits += 1
        if det.get("cls_name"):
            self.cls_votes[det["cls_name"]] = self.cls_votes.get(det["cls_name"], 0.0) + float(det["conf"])
        if det.get("width_px"):
            self.widths.append(det["width_px"])
            self.width_src = det.get("width_src")
        self.xyz_now = det.get("xyz")
        if det.get("xyz") is not None:
            self.xyzs.append(det["xyz"])
            self.fx = det.get("fx")
        if det.get("sat") is not None:
            self.sats.append(det["sat"])
            if det.get("sat_thr") is not None:
                self.sat_thr = det["sat_thr"]

    @property
    def xyz(self):
        """Median X, Y, Z over the last depth_smooth frames (metres), or None."""
        return tuple(float(v) for v in np.median(np.array(self.xyzs), axis=0)) if self.xyzs else None

    @property
    def width_mm_depth(self):
        """Width in mm from the measured px width and this end's distance Z."""
        z = self.xyz
        return depth3d.width_mm(self.width, z[2] if z else None, self.fx)

    @property
    def confirmed(self):
        return self.hits >= self.min_frames

    @property
    def width(self):
        """Median width over every frame this end was measured (stable)."""
        return float(np.median(self.widths)) if self.widths else None

    @property
    def width_spread(self):
        return float(np.std(self.widths)) if len(self.widths) > 2 else None

    @property
    def sat(self):
        """Median saturation inside the mask over the last color_window frames."""
        return float(np.median(self.sats)) if self.sats else None

    @property
    def cls(self):
        """The class this end is, by confidence-weighted vote over every frame (e.g. head_blue)."""
        return max(self.cls_votes, key=self.cls_votes.get) if self.cls_votes else None

    @property
    def cls_share(self):
        """How sure the vote is: share of the winning class (1.0 = always the same class)."""
        tot = sum(self.cls_votes.values())
        return round(self.cls_votes[self.cls] / tot, 3) if tot else None

    @property
    def color_from(self):
        """'class' or 'saturation' - where self.color comes from."""
        if self.color_source != "saturation" and color_from_name(self.cls):
            return "class"
        if self.color_source != "class" and self.sat is not None and self.sat_thr is not None:
            return "saturation"
        return None

    @property
    def color(self):
        """'blue' / 'white' from the voted class name (head_blue -> blue), or from the
        median saturation when the class carries no colour (color_source decides)."""
        src = self.color_from
        if src == "class":
            return color_from_name(self.cls)
        if src == "saturation":
            return "blue" if self.sat > self.sat_thr else "white"
        return None


def det_color(d, cfg):
    """Colour of ONE detection for the colour gate: its class name (head_blue -> blue)
    when color_source allows, else 'blue' / 'white' only if its saturation is CLEARLY on
    one side of sat_threshold (at least color_margin away); None = borderline / unknown."""
    src = getattr(cfg, "color_source", "auto")
    if src != "saturation" and d.get("cls_color"):
        return d["cls_color"]
    if src == "class":
        return None
    s = d.get("sat")
    if s is None:
        return None
    if s > cfg.sat_threshold + cfg.color_margin:
        return "blue"
    if s < cfg.sat_threshold - cfg.color_margin:
        return "white"
    return None


def update_tracks(tracks, dets, idx, next_id, cfg):
    """Greedy nearest-center matching, per role (a head never matches a tail).
    color_gate: a clearly blue detection never matches a white track and vice versa."""
    gate = getattr(cfg, "color_gate", False) and (
        getattr(cfg, "tube_color", False) or getattr(cfg, "color_source", "auto") != "saturation")
    for role in ROLES:
        live = [t for t in tracks if t.role == role and idx - t.last_seen <= cfg.max_missed]
        cand = [d for d in dets if d["role"] == role]
        pairs = []
        for ti, t in enumerate(live):
            for di, d in enumerate(cand):
                dist = float(np.hypot(t.center[0] - d["center"][0],
                                      t.center[1] - d["center"][1]))
                limit = cfg.max_dist_tail if (role == "TAIL" and cfg.max_dist_tail) else cfg.max_dist
                if dist > limit:
                    continue
                if gate:
                    dc, tc = det_color(d, cfg), t.color
                    if dc and tc and dc != tc:      # blue end vs white track -> never the same end
                        continue
                pairs.append((dist, ti, di))
        used_t, used_d = set(), set()
        for dist, ti, di in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            live[ti].update(cand[di], idx)
            cand[di]["track"] = live[ti]
        for di, d in enumerate(cand):
            if di not in used_d:
                mf = cfg.min_frames_tail if (role == "TAIL" and cfg.min_frames_tail) else cfg.min_frames
                t = Track(next_id, d, idx, mf)
                next_id += 1
                tracks.append(t)
                d["track"] = t
    return next_id


# ----------------------------------------------------------------- size logic
def size_labels(items, min_ratio):
    """items: [(key, width)] -> {key: BIGGER/SMALLER/SAME}, split at the largest gap."""
    if len(items) < 2:
        return {}
    ws = np.sort([w for _, w in items])
    ratios = ws[1:] / ws[:-1]
    if ratios.max() < min_ratio:
        return {key: "SAME" for key, _ in items}
    i = int(ratios.argmax())
    cut = (ws[i] + ws[i + 1]) / 2
    return {key: ("BIGGER" if w > cut else "SMALLER") for key, w in items}


def assign_sizes(dets, cfg):
    """Sets d['size'] (heads vs heads, tails vs tails) and returns bigger_tube."""
    measured = [d for d in dets if d["track"].confirmed and d["track"].width]
    for d in dets:
        d["size"] = None
    for role in ROLES:
        group = [d for d in measured if d["role"] == role]
        if any(d["track"].xyz for d in group):
            # with depth: only heads/tails that HAVE a Z value take part in Big/Small/Equal
            group = [d for d in group if d["track"].xyz]
        labels = size_labels([(id(d), d["track"].width) for d in group], cfg.min_ratio)
        for d in group:
            d["size"] = labels.get(id(d))

    pool = measured if cfg.width_from == "all" else [d for d in measured if d["role"] == "TAIL"]
    if len(pool) < 2:
        return {"status": "NOT_ENOUGH", "compared": len(pool)}
    widths = [d["track"].width for d in pool]
    big = pool[int(np.argmax(widths))]
    ratio = max(widths) / min(widths)
    if ratio < cfg.min_ratio:
        return {"status": "SAME", "compared": len(pool), "ratio": round(ratio, 3)}
    t = big["track"]
    return {
        "status": "OK", "compared": len(pool), "id": t.id, "role": t.role,
        "width_px": round(t.width, 1),
        "width_mm": round(t.width * cfg.mm_per_px, 2) if cfg.mm_per_px else None,
        "ratio": round(ratio, 3),
    }


def fmt_w(w, mm_per_px):
    if not w:
        return "-"
    return f"{w * mm_per_px:.2f}mm" if mm_per_px else f"{w:.1f}px"


# -------------------------------------------------------------------- output
def load_roi(path):
    if not path:
        return None
    if not Path(path).exists():
        raise SystemExit(f"roi_json is set but not found: {path}  (use --no-roi to run without)")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(data["points"], dtype=np.int32)


def draw_roi_interactive(video, out_path, frame_no=0):
    """Show one frame and let the user click a polygon. Saves {"points": [...]} to
    out_path and returns the points (None if cancelled).
        left click = add point     right click / Backspace = undo last point
        Enter or Space = save      Esc = cancel"""
    cap = cv2.VideoCapture(int(video) if video.isdigit() else video)
    if frame_no:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read a frame from {video}")
    s = min(1.0, 900 / frame.shape[0])           # fit the window on screen
    pts = []                                     # in ORIGINAL pixel coordinates
    win = "draw ROI: click corners | Enter=save | right-click=undo | Esc=cancel"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(round(x / s)), int(round(y / s))))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        raise SystemExit("--draw-roi needs a screen and the GUI build of OpenCV (opencv-python); "
                         "draw roi.json on a desktop machine and copy it in") from None
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = cv2.resize(frame, None, fx=s, fy=s) if s != 1 else frame.copy()
        if pts:
            p = np.array([(int(x * s), int(y * s)) for x, y in pts], dtype=np.int32)
            cv2.polylines(vis, [p.reshape(-1, 1, 2)], len(pts) > 2, (0, 220, 0), 2, cv2.LINE_AA)
            for q in p:
                cv2.circle(vis, tuple(q), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"{len(pts)} points  (need 3+)  Enter=save  Esc=cancel", (10, 28),
                    FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            pts = None
            break
        if key == 8 and pts:
            pts.pop()
        if key in (13, 32):
            if len(pts) >= 3:
                break
            print("[roi] need at least 3 points")
    cv2.destroyWindow(win)
    if pts is None:
        print("[roi] cancelled")
        return None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"points": [list(p) for p in pts]}, indent=2))
    print(f"[roi] saved {len(pts)} points -> {out_path}")
    return pts


def in_roi(det, roi):
    if roi is None:
        return True
    cx, cy = det["center"]
    return cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0


def cap_per_role(dets, cfg):
    """Keep only the best max_heads / max_tails confirmed ends in this frame."""
    limits = {"HEAD": cfg.max_heads, "TAIL": cfg.max_tails}
    out = []
    for role in ROLES:
        mine = [d for d in dets if d["role"] == role]
        confirmed = sorted((d for d in mine if d["track"].confirmed),
                           key=lambda d: d["conf"], reverse=True)
        if limits[role] and limits[role] > 0:
            confirmed = confirmed[:limits[role]]
        out += confirmed + [d for d in mine if not d["track"].confirmed]
    return out


def build_frame_result(idx, dets, tracks, bigger, cfg, end_classes=None, tubes=None,
                       tube_classes=None, tube_totals=None):
    """`dets` = this frame's detections INSIDE the ROI.
    end_classes = head/tail class names of the model, so class_counts / class_totals
    always carry every class (0 when absent) - stable keys for a frontend.
    tubes = this frame's tube records inside the ROI (tube_record), tube_classes = their
    class names (white_tube, blue_tube) for stable keys in tube_counts."""
    counts = {}
    for role in ROLES:
        key = role.lower() + "s"
        visible = sum(d["track"].confirmed and d["role"] == role for d in dets)
        counts[f"{key}_count"] = visible
        counts[f"{key}_visible"] = visible
        counts[f"{key}_total"] = sum(t.role == role and t.confirmed and t.in_roi_ever
                                     for t in tracks)
    class_counts = {n: 0 for n in (end_classes or [])}
    for d in dets:
        if d["track"].confirmed and d["track"].cls:
            class_counts[d["track"].cls] = class_counts.get(d["track"].cls, 0) + 1
    class_totals = {n: 0 for n in (end_classes or [])}
    for t in tracks:
        if t.confirmed and t.in_roi_ever and t.cls:
            class_totals[t.cls] = class_totals.get(t.cls, 0) + 1
    counts["class_counts"] = class_counts
    counts["class_totals"] = class_totals

    detections = []
    for d in dets:
        t = d["track"]
        if not t.confirmed:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        w = t.width
        detections.append({
            "id": t.id,
            "role": t.role,
            "class": t.cls,                             # voted over the track, e.g. head_blue
            "class_now": d.get("cls_name"),             # what the model said in THIS frame
            "class_share": t.cls_share,                 # 1.0 = the track was always this class
            "conf": round(d["conf"], 3),
            "bbox": [int(round(x1)), int(round(y1)),
                     int(round(x2 - x1)), int(round(y2 - y1))],
            "center": [int(round(d["center"][0])), int(round(d["center"][1]))],
            "width_px": round(w, 1) if w else None,
            "width_mm": round(w * cfg.mm_per_px, 2) if (w and cfg.mm_per_px) else None,
            "width_now_px": round(d["width_px"], 1) if d.get("width_px") else None,
            "width_spread_px": round(t.width_spread, 2) if t.width_spread is not None else None,
            "width_source": t.width_src,
            "size": d.get("size"),
            "tube_color": t.color,                      # blue / white / None
            "color_from": t.color_from,                 # class / saturation / None
            "sat": round(t.sat, 1) if t.sat is not None else None,
            "sat_now": round(d["sat"], 1) if d.get("sat") is not None else None,
            "xyz_m": depth3d.r3(t.xyz),                 # median of the last depth_smooth frames
            "xyz_now_m": depth3d.r3(d.get("xyz")),      # this frame only
            "z_m": round(t.xyz[2], 4) if t.xyz else None,
            "depth_source": d.get("xyz_src"),           # mask / center / None (no valid depth)
            "width_mm_depth": round(t.width_mm_depth, 2) if t.width_mm_depth else None,
            "polygon": d.get("polygon"),                # mask outline [[x, y], ...] (output_polygons)
        })
    out = {"frame": idx, **counts, "detections": detections, "bigger_tube": bigger}
    if getattr(cfg, "output_tubes", False):
        tubes = list(tubes or [])
        tube_counts = {n: 0 for n in (tube_classes or [])}
        for i, tb in enumerate(tubes, 1):
            tb["index"] = i                             # position in this frame; "id" = stable (track_tubes)
            tube_counts[tb["class"]] = tube_counts.get(tb["class"], 0) + 1
        out["tubes"] = tubes
        out["tubes_count"] = len(tubes)
        out["tube_counts"] = tube_counts
        if tube_totals is not None:                     # distinct tubes seen so far (track_tubes)
            out["tube_totals"] = tube_totals
            out["tubes_total"] = sum(tube_totals.values())
    return out


def collect_traces(tubes, hist, cfg):
    """Flat list of this frame's wrapper traces from the tube records (after tube
    tracking, so each trace carries the tube's stable id). The trace is taken OUT of
    each tube record (one copy in the JSON). end_xyz_m = median of the end point's
    X/Y/Z over the last depth_smooth frames of the same tube + side."""
    out = []
    tracked = getattr(cfg, "track_tubes", True)
    seen = set()
    for i, tb in enumerate(tubes, 1):
        trs = tb.pop("wrapper_trace", None) or []
        if tb.get("held"):
            continue                       # the model lost this tube this frame: old trace, skip
        for tr in trs:
            e = dict(tr)
            tid = tb.get("id") if tracked else None
            e.update({"tube_id": tid, "tube_index": i, "tube_class": tb.get("class"),
                      "tube_color": tb.get("tube_color")})
            key = (tid, e.get("side"))
            now = e.get("end_xyz_now_m")
            if tid is not None and key not in seen and now is not None:
                seen.add(key)
                dq = hist.setdefault(key, deque(maxlen=max(int(cfg.depth_smooth or 1), 1)))
                dq.append(now)
                e["end_xyz_m"] = depth3d.r3(np.median(np.array(dq), axis=0))
            else:
                e["end_xyz_m"] = now
            out.append(e)
    return out


def trace_summary_text(tr):
    xyz = tr.get("end_xyz_m")
    p = f"({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:.3f}) m" if xyz else "no depth"
    short = "" if tr["status"] == "OK" else f"  SHORT {tr['reached_px']:.0f}/{tr['trace_px']:.0f}px"
    return (f"trace tube #{tr.get('tube_id') or tr.get('tube_index')} {tr.get('tube_class') or ''} "
            f"{tr.get('side', '?')}: end {tr['end_px']} XYZ={p}{short}")


def output_tubes(tubes, roi, tracker, idx, cfg, tube_classes):
    """ROI filter -> tracking (or not) -> (tubes to output, tube_totals or None)."""
    tubes = [t for t in tubes if tube_in_roi(t, roi)]
    if getattr(cfg, "track_tubes", True) and tracker is not None:
        out = tracker.update(tubes, idx, cfg)
        mf = max(int(getattr(cfg, "tube_min_frames", 3) or 1), 1)
        return out, tracker.totals(mf, tube_classes)
    return plain_tubes(tubes), None


def summarize(tracks, cfg, end_classes=None, roi_only=True, tubes=None, tube_classes=()):
    """Whole-video result from the tracks (call when the video has ended).
    One entry per confirmed end that appeared in the ROI, with its voted class, colour
    and median width; Big/Small/Equal per role and the bigger tube, over the WHOLE video."""
    keep = [t for t in sorted(tracks, key=lambda t: t.id)
            if t.confirmed and (t.in_roi_ever or not roi_only)]
    sizes = {}
    for role in ROLES:
        sizes.update(size_labels([(t.id, t.width) for t in keep if t.role == role and t.width],
                                 cfg.min_ratio))
    ends = []
    for t in keep:
        w = t.width
        ends.append({
            "id": t.id, "role": t.role, "class": t.cls, "class_share": t.cls_share,
            "tube_color": t.color, "color_from": t.color_from,
            "first_frame": t.first_seen, "last_frame": t.last_seen, "frames_seen": t.hits,
            "width_px": round(w, 1) if w else None,
            "width_mm": round(w * cfg.mm_per_px, 2) if (w and cfg.mm_per_px) else None,
            "width_spread_px": round(t.width_spread, 2) if t.width_spread is not None else None,
            "measured_frames": len(t.widths),
            "width_source": t.width_src,
            "size": sizes.get(t.id),
            "xyz_m": depth3d.r3(t.xyz),
            "width_mm_depth": round(t.width_mm_depth, 2) if t.width_mm_depth else None,
        })
    pool = [e for e in ends if e["width_px"] and (cfg.width_from == "all" or e["role"] == "TAIL")]
    if len(pool) < 2:
        bigger = {"status": "NOT_ENOUGH", "compared": len(pool)}
    else:
        big = max(pool, key=lambda e: e["width_px"])
        ratio = big["width_px"] / min(e["width_px"] for e in pool)
        bigger = ({"status": "SAME", "compared": len(pool), "ratio": round(ratio, 3)}
                  if ratio < cfg.min_ratio else
                  {"status": "OK", "compared": len(pool), "id": big["id"], "role": big["role"],
                   "class": big["class"], "tube_color": big["tube_color"],
                   "width_px": big["width_px"], "width_mm": big["width_mm"],
                   "ratio": round(ratio, 3)})
    class_totals = {n: 0 for n in (end_classes or [])}
    for e in ends:
        if e["class"]:
            class_totals[e["class"]] = class_totals.get(e["class"], 0) + 1
    out = {
        "heads_total": sum(e["role"] == "HEAD" for e in ends),
        "tails_total": sum(e["role"] == "TAIL" for e in ends),
        "class_totals": class_totals,
        "ends": ends,
        "bigger_tube": bigger,
    }
    if tubes is not None and getattr(cfg, "output_tubes", False) and getattr(cfg, "track_tubes", True):
        mf = max(int(getattr(cfg, "tube_min_frames", 3) or 1), 1)
        tt = tubes.totals(mf, tube_classes)
        out["tube_totals"] = tt
        out["tubes_total"] = sum(tt.values())
        out["tube_list"] = [{"id": t.id, "class": t.cls, "class_share": t.share,
                             "tube_color": color_from_name(t.cls), "frames_seen": t.hits,
                             "last_frame": t.last_seen}
                            for t in sorted(tubes.tracks + tubes.gone, key=lambda t: t.id)
                            if t.hits >= mf]
    return out


def bigger_text(b, mm_per_px):
    if b["status"] == "OK":
        return f"BIGGER TUBE: {b['role'].lower()} #{b['id']} {fmt_w(b['width_px'], mm_per_px)} (x{b['ratio']:.2f})"
    if b["status"] == "SAME":
        return f"TUBES SAME SIZE (x{b['ratio']:.2f})"
    return "BIGGER TUBE: need 2 measured tubes"


def frame_summary(result, cfg):
    tot = f"/{result['frame_total']}" if result.get("frame_total") else ""
    fps = f"  {result['fps']:.1f} fps" if result.get("fps") else ""
    print(f"\nFrame {result['frame']}{tot}{fps}: HEADS={result['heads_count']}  "
          f"TAILS={result['tails_count']}  "
          f"(peak at once: {result['heads_peak']} heads, {result['tails_peak']} tails)")
    for d in result["detections"]:
        xyz = d.get("xyz_m")
        p3 = (f"  XYZ=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:.3f}) m" if xyz else
              ("  XYZ=no depth" if cfg.depth_video else ""))
        wmm = f" ~{d['width_mm_depth']:.1f}mm" if d.get("width_mm_depth") else ""
        col = ""
        if d.get("color_from") == "saturation":
            sat = f"{d['sat']:.0f}" if d.get("sat") is not None else "-"
            col = f"{(d.get('tube_color') or '-'):5s} sat={sat:>3} "
        print(f"  #{d['id']:<3} {(d.get('class') or d['role']):10s} {col}conf={d['conf']:.2f} "
              f"width={fmt_w(d['width_px'], cfg.mm_per_px):>9}{wmm} {d['size'] or '':7s} "
              f"bbox={d['bbox']}{p3}")
    if result.get("tubes"):
        print("  tubes: " + "  ".join(f"{t['class']} conf={t['conf']:.2f} bbox={t['bbox']}"
                                      for t in result["tubes"]))
    if cfg.trace_px:
        if result.get("wrapper") is None:
            print("  trace: no wrapper in this frame")
        for tr in result.get("traces") or []:
            print("  " + trace_summary_text(tr))
    print(f"  -> {bigger_text(result['bigger_tube'], cfg.mm_per_px)}")


# ------------------------------------------------------------------- drawing
def _text_color(bgr):
    """Black text on light colours, white text on dark ones."""
    b, g, r = bgr
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)


def _light(bgr, k=0.35):
    """A lighter tint of a colour - readable as text on the dark panel."""
    return tuple(int(c + (255 - c) * k) for c in bgr)


def _overlaps(r, rects):
    return any(r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3] for o in rects)


def _label_tag(img, lines, box, color, fs, ft, occupied, obstacles=()):
    """Small grey tag for ends that are not confirmed yet."""
    H, W = img.shape[:2]
    sizes = [cv2.getTextSize(t, FONT, fs, ft) for t in lines]
    tw = max(z[0][0] for z in sizes)
    lh = max(z[0][1] for z in sizes)
    base = max(z[1] for z in sizes)
    m = max(2, int(round(lh * 0.3)))
    tag_w, tag_h = tw + 2 * m, lh + base + 2 * m
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    spots = [(x1, y1 - tag_h), (x1, y2), (x2 + 2, y1), (x1 - tag_w - 2, y1)]

    def clamp(x, y):
        return min(max(x, 0), max(W - tag_w, 0)), min(max(y, 0), max(H - tag_h, 0))

    tx, ty = clamp(*spots[0])
    for sx, sy in spots:
        cx, cy = clamp(sx, sy)
        if not _overlaps((cx, cy, cx + tag_w, cy + tag_h), list(occupied) + list(obstacles)):
            tx, ty = cx, cy
            break
    occupied.append((tx, ty, tx + tag_w, ty + tag_h))
    cv2.rectangle(img, (tx, ty), (tx + tag_w, ty + tag_h), color, -1)
    cv2.putText(img, lines[0], (tx + m, ty + m + lh), FONT, fs, _text_color(color), ft, cv2.LINE_AA)


def _rounded_rect(img, x1, y1, x2, y2, r, color, thick):
    """Rectangle with rounded corners; thick < 0 = filled."""
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thick < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, cv2.LINE_AA)
    for (cx, cy), ang in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                          ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, thick, cv2.LINE_AA)


def _closest(rect, px, py):
    """Point on/inside rect closest to (px, py)."""
    return (int(min(max(px, rect[0]), rect[2])), int(min(max(py, rect[1]), rect[3])))


def _px(v, mm_per_px):
    if not v:
        return "-"
    return f"{v * mm_per_px:.2f} mm" if mm_per_px else f"{v:.1f} px"


def _end_text(d, cfg):
    """(title, body lines) for one confirmed end - used by the card and the panel."""
    t = d["track"]
    col = getattr(t, "color", None)
    title = f"ID: {t.id}  {t.role.title()}" + (f" {col}" if col else "")
    size_word = {"BIGGER": "Big", "SMALLER": "Small", "SAME": "Equal"}.get(d.get("size"), "-")
    wmm = t.width_mm_depth if t.xyz else None
    if wmm and not cfg.mm_per_px:
        wtxt = f"W: {wmm:.1f} mm ({_px(t.width, None)})"
    else:
        wtxt = f"W: {_px(t.width, cfg.mm_per_px)}"
    body = [size_word, wtxt]
    if t.xyz:
        body.append(f"Z: {t.xyz[2] * 1000:.0f} mm")
    elif cfg.depth_video:
        body.append("Z: -")
    return title, body


def draw(frame, dets, result, roi, cfg, show_cards=None):
    """show_cards: None = cfg.show_cards; True = cards + boxes; False = boxes only."""
    if show_cards is None:
        show_cards = getattr(cfg, "show_cards", True)
    vis = frame.copy()
    H, W = vis.shape[:2]
    s = float(cfg.draw_scale) if cfg.draw_scale else max(H, W) / 1280.0
    s = max(s, 0.6)
    th = max(2, int(round(1.4 * s)))           # box line (thin: 2 px at 1920x1080)
    pad = int(round(3 * s))                    # grow boxes just a little

    def color_of(d):
        return COLORS[d["track"].role] if d["track"].confirmed else COL_PENDING

    def rect(d):
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        return (max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, W - 1), min(y2 + pad, H - 1))

    shown = [d for d in dets if d["track"].confirmed or cfg.draw_unconfirmed]
    shown.sort(key=lambda d: d["track"].confirmed)          # unconfirmed first, confirmed on top
    boxes = [rect(d) for d in shown]

    # 1. translucent fill (box + mask) for confirmed ends  (box_fill: 0 = off)
    if cfg.box_fill > 0:
        overlay = vis.copy()
        for d in shown:
            if d["track"].confirmed:
                x1, y1, x2, y2 = rect(d)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_of(d), -1)
                if d.get("outline") is not None:
                    cv2.fillPoly(overlay, [d["outline"]], color_of(d))
        vis = cv2.addWeighted(overlay, cfg.box_fill, vis, 1 - cfg.box_fill, 0)

    # 2. ROI
    if roi is not None and cfg.draw_roi:
        cv2.polylines(vis, [roi.reshape(-1, 1, 2)], True, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.polylines(vis, [roi.reshape(-1, 1, 2)], True, (0, 220, 0), th, cv2.LINE_AA)

    # 2b. tubes (blue / white outline, light fill) - under the head/tail boxes
    tubes = result.get("tubes") or []
    if tubes and getattr(cfg, "draw_tubes", True):
        polys = [(np.array(p, np.int32).reshape(-1, 1, 2), TUBE_COLORS.get(tb.get("tube_color"), (0, 200, 0)))
                 for tb in tubes for p in tb.get("polygons") or []]
        layer = vis.copy()
        for p, col in polys:
            cv2.fillPoly(layer, [p], col)
        vis = cv2.addWeighted(layer, 0.18, vis, 0.82, 0)
        for p, col in polys:
            cv2.polylines(vis, [p], True, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(vis, [p], True, col, 1, cv2.LINE_AA)

    # 2c. wrapper outline + traces out of it (trace_px)
    wr = result.get("wrapper")
    if wr and wr.get("polygon") and getattr(cfg, "draw_wrapper", True):
        p = np.array(wr["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [p], True, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.polylines(vis, [p], True, (0, 200, 255), th, cv2.LINE_AA)
    if getattr(cfg, "draw_trace", True):
        tfs_ = max(0.4, 0.42 * s)
        for tr in result.get("traces") or []:
            col = (60, 60, 255) if tr["status"] == "OK" else (0, 140, 255)   # red / orange = SHORT
            p = np.array(tr["path"], np.float32).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [p], False, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.polylines(vis, [p], False, col, th + 1, cv2.LINE_AA)
            sx, sy = (int(round(v)) for v in tr["start_px"])
            ex, ey = (int(round(v)) for v in tr["end_px"])
            r_ = max(4, int(round(4 * s)))
            cv2.circle(vis, (sx, sy), r_, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (sx, sy), r_ - 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (ex, ey), r_ + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (ex, ey), r_ + 1, col, -1, cv2.LINE_AA)
            xyz = tr.get("end_xyz_m")
            txt = (f"#{tr.get('tube_id') or tr.get('tube_index')} Z {xyz[2] * 1000:.0f}mm" if xyz
                   else f"#{tr.get('tube_id') or tr.get('tube_index')} Z -")
            (tw_, th_), _ = cv2.getTextSize(txt, FONT, tfs_, 1)
            tx, ty = min(max(ex + r_ + 4, 0), W - tw_ - 2), min(max(ey - r_ - 4, th_ + 2), H - 2)
            cv2.putText(vis, txt, (tx, ty), FONT, tfs_, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (tx, ty), FONT, tfs_, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. boxes (one colour per role) + thin mask outline
    for d in shown:
        x1, y1, x2, y2 = rect(d)
        thick = th if d["track"].confirmed else 1
        if d.get("outline") is not None and d["track"].confirmed:
            cv2.polylines(vis, [d["outline"]], True, color_of(d), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_of(d), thick, cv2.LINE_AA)

    occupied = []
    # 4. info panel (top-left)
    total = result.get("frame_total")
    ftxt = f"FRAME {result['frame']}" + (f" / {total}" if total else "")
    fps = result.get("fps")                                 # real processing speed (changes per video/GPU)
    rows = [[(ftxt, (215, 215, 215)), (f"{fps:.1f} FPS" if fps else "- FPS", (215, 215, 215))],
            [(f"HEADS {result['heads_count']}", COLORS["HEAD"]),
             (f"TAILS {result['tails_count']}", COLORS["TAIL"])]]
    if getattr(cfg, "panel_details", True):
        conf_ends = sorted((d for d in shown if d["track"].confirmed),
                           key=lambda d: (d["track"].role, d["track"].id))
        for d in conf_ends:
            title, body = _end_text(d, cfg)
            rows.append([("   ".join([title] + body), _light(COLORS[d["track"].role], 0.15))])
        b = result.get("bigger_tube")
        if b:
            bcol = COL_BIG if b.get("status") == "OK" else (COL_SAME if b.get("status") == "SAME"
                                                            else (215, 215, 215))
            rows.append([(bigger_text(b, cfg.mm_per_px), bcol)])
        if result.get("trace_px"):
            if result.get("wrapper") is None:
                rows.append([(f"TRACE {result['trace_px']:.0f}px: no wrapper", (215, 215, 215))])
            for tr in result.get("traces") or []:
                xyz = tr.get("end_xyz_m")
                p3 = (f"X {xyz[0] * 1000:+.0f} Y {xyz[1] * 1000:+.0f} Z {xyz[2] * 1000:.0f} mm"
                      if xyz else "no depth")
                short = "" if tr["status"] == "OK" else f"  short {tr['reached_px']:.0f}px"
                rows.append([(f"Tube {tr.get('tube_id') or tr.get('tube_index')} {tr.get('side', '')} "
                              f"+{tr['trace_px']:.0f}px: {p3}{short}", (130, 130, 255))])
    pfs = max(0.42, 0.45 * s)
    pft = 1 if s < 1.3 else 2
    lh = cv2.getTextSize("Ag", FONT, pfs, pft)[0][1]
    m = max(5, int(round(7 * s)))
    dot = max(5, int(round(lh * 0.75)))
    gap = int(round(lh * 1.3))
    row_h = lh + max(5, int(round(lh * 0.8)))

    def item_w(text):
        return dot + int(round(dot * 0.6)) + cv2.getTextSize(text, FONT, pfs, pft)[0][0]

    widths = [sum(item_w(t) for t, _ in r) + gap * (len(r) - 1) for r in rows]
    pw, ph = max(widths) + 2 * m, len(rows) * row_h - (row_h - lh) + 2 * m
    px, py = m, m
    x2p, y2p = min(px + pw, W - 1), min(py + ph, H - 1)
    panel = vis.copy()
    _rounded_rect(panel, px, py, x2p, y2p, int(6 * s), (20, 20, 20), -1)
    vis = cv2.addWeighted(panel, 0.85, vis, 0.15, 0)
    _rounded_rect(vis, px, py, x2p, y2p, int(6 * s), (90, 90, 90), 1)
    y = py + m + lh
    for r in rows:
        x = px + m
        for text, col in r:
            cv2.rectangle(vis, (x, y - dot), (x + dot, y), col, -1)
            tx = x + dot + int(round(dot * 0.6))
            cv2.putText(vis, text, (tx, y), FONT, pfs, col, pft, cv2.LINE_AA)
            x = tx + cv2.getTextSize(text, FONT, pfs, pft)[0][0] + gap
        y += row_h
    occupied.append((px, py, x2p, y2p))

    # 5. callout cards for confirmed ends (like: ID / Conf / Size / W H)
    tfs, tft = max(0.42, 0.42 * s), 2 if s >= 1.2 else 1   # title
    bfs, bft = max(0.36, 0.36 * s), 1                       # body
    cpad = max(4, int(round(5 * s)))
    off = int(round(16 * s))                                # gap between box and card
    cards = []
    for d in sorted((d for d in shown if d["track"].confirmed and show_cards),
                    key=lambda d: d["conf"], reverse=True):
        # card: ID + class / Big-Small-Equal / width (mm from depth + px) / Z
        title, body = _end_text(d, cfg)
        (tw, tlh), tb = cv2.getTextSize(title, FONT, tfs, tft)
        bsz = [cv2.getTextSize(b, FONT, bfs, bft) for b in body]
        blh = max(z[0][1] for z in bsz)
        bgap = int(round(blh * 0.6))
        cw = max([tw] + [z[0][0] for z in bsz]) + 2 * cpad
        ch = tlh + tb + int(round(blh * 0.4)) + len(body) * (blh + bgap) - bgap + 2 * cpad
        bx = rect(d)
        cand = [(bx[0] - int(cw * 0.7), bx[1] - ch - off), (bx[2] - int(cw * 0.3), bx[1] - ch - off),
                (bx[2] + off, bx[1] - ch // 2), (bx[0] - off - cw, bx[1] - ch // 2),
                (bx[2] + off, bx[3] - ch // 2), (bx[0] - off - cw, bx[3] - ch // 2),
                (bx[0] - int(cw * 0.7), bx[3] + off), (bx[2] - int(cw * 0.3), bx[3] + off)]
        cand += [(cx, cy - k * (ch // 2)) for k in (1, 2) for cx, cy in cand[:2]]
        cand += [(cx, cy + k * (ch // 2)) for k in (1, 2) for cx, cy in cand[6:8]]

        def clamp(x, y):
            return min(max(x, 0), max(W - 1 - cw, 0)), min(max(y, 0), max(H - 1 - ch, 0))

        cx, cy = clamp(*cand[0])
        for sx, sy in cand:
            qx, qy = clamp(sx, sy)
            r_ = (qx, qy, qx + cw, qy + ch)
            if not _overlaps(r_, occupied) and not _overlaps(r_, boxes):
                cx, cy = qx, qy
                break
        occupied.append((cx, cy, cx + cw, cy + ch))
        cards.append((d, (cx, cy, cx + cw, cy + ch), title, body, tlh, tb, blh, bgap))

    # leader lines first (cards are drawn on top of them)
    for d, c, *_ in cards:
        col, bx = color_of(d), rect(d)
        bcx, bcy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        p_card = _closest(c, bcx, bcy)
        p_box = _closest(bx, (c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
        cv2.line(vis, p_card, p_box, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(vis, p_card, p_box, col, 2, cv2.LINE_AA)
        r_dot = max(3, int(round(3 * s)))
        cv2.circle(vis, p_box, r_dot + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, p_box, r_dot, col, -1, cv2.LINE_AA)

    # card backgrounds: dark tint of the role colour, slightly see-through
    if cards:
        layer = vis.copy()
        rad = int(round(5 * s))
        for d, c, *_ in cards:
            fill = tuple(int(v * 0.42) for v in color_of(d))
            _rounded_rect(layer, c[0], c[1], c[2], c[3], rad, fill, -1)
        vis = cv2.addWeighted(layer, 0.88, vis, 0.12, 0)
        for d, c, title, body, tlh, tb, blh, bgap in cards:
            col = color_of(d)
            _rounded_rect(vis, c[0], c[1], c[2], c[3], rad, col, 2)
            x, y = c[0] + cpad, c[1] + cpad + tlh
            cv2.putText(vis, title, (x, y), FONT, tfs, (255, 255, 255), tft, cv2.LINE_AA)
            y += tb + int(round(blh * 0.4)) + blh
            for b in body:
                cv2.putText(vis, b, (x, y), FONT, bfs, (235, 235, 235), bft, cv2.LINE_AA)
                y += blh + bgap

    # 5b. cards off: small ID tag on each confirmed box, so the panel lines can be matched
    if not show_cards and getattr(cfg, "box_id_tag", True):
        for d in shown:
            if d["track"].confirmed:
                _label_tag(vis, [str(d["track"].id)], rect(d), color_of(d),
                           max(0.35, 0.4 * s), 1, occupied, boxes)

    # 6. small grey tag for not-yet-confirmed ends
    for d in shown:
        if not d["track"].confirmed:
            _label_tag(vis, [f"{d['track'].role.lower()}? {d['conf']:.2f}"], rect(d), COL_PENDING,
                       max(0.35, 0.4 * s), 1, occupied, boxes)

    return vis


def print_diagnosis(tracks, min_frames, mm_per_px=None):
    print(f"\n[diagnosis] {len(tracks)} tracks created, "
          f"{sum(t.confirmed for t in tracks)} confirmed (min_frames={min_frames})")
    for role in ROLES:
        rows = sorted((t for t in tracks if t.role == role), key=lambda t: t.first_seen)
        print(f"  {role}: {len(rows)} created, {sum(t.confirmed for t in rows)} confirmed")
        for t in rows:
            tag = "CONFIRMED" if t.confirmed else "discarded (flicker)"
            col = f"  colour={t.color} (sat {t.sat:.0f})" if t.sat is not None else ""
            if t.cls:
                col = f"  class={t.cls} ({t.cls_share:.0%})" + col
            print(f"    id={t.id:<4} frames {t.first_seen}-{t.last_seen}  seen {t.hits}x  "
                  f"width={fmt_w(t.width, mm_per_px)} over {len(t.widths)} frames{col}  "
                  f"first center={tuple(round(v) for v in t.first_center)}  [{tag}]")
    print("  -> many short confirmed tracks of one role = the same end re-counted "
          "(raise max_dist or max_missed)")


# ======================================================================
# CORE used by main.py (HeadTailAnalyzer): model manager, per-frame pipeline,
# video in/out. main.py only wraps these - it holds no pipeline logic itself.
# ======================================================================

# ------------------------------------------------------------------ errors
class AnalyzerError(RuntimeError):
    """Anything that stops the analyzer: bad config, missing model, unreadable video."""


@contextmanager
def as_error():
    """The CLI code reports problems with SystemExit (fine for a terminal, bad inside a
    backend - it would stop the whole process). Turn them into AnalyzerError."""
    try:
        yield
    except SystemExit as e:
        raise AnalyzerError(str(e.code) if e.code not in (None, 0) else "stopped") from None


# ------------------------------------------------------------------ the model
@dataclass
class LoadedModel:
    """A model in memory plus what the pipeline needs to know about its classes."""
    path: str
    model: Any
    role_ids: dict                 # {class_id: "HEAD" / "TAIL"}
    tubing_id: Optional[int]       # first tube class (old callers)
    task: str                      # "segment" or "detect"
    names: dict
    serial: int                    # increases with every load; changes after unload + reload
    infer_lock: Any = field(default_factory=threading.Lock, repr=False)
    tubing_ids: list = field(default_factory=list)   # every tube class (white_tube, blue_tube)

    @property
    def end_classes(self):
        """Head/tail class names in id order, e.g. [head_white, head_blue, tail_white, tail_blue]."""
        return [self.names[i] for i in sorted(self.role_ids)]
    # infer_lock: ONE prediction at a time on this model. Every analyzer / session
    # that shares the model takes it around model.predict, so two simultaneous
    # requests queue for the GPU instead of running predict on the same object at once.


def _default_loader(path):
    from ultralytics import YOLO   # imported here so config errors show up fast
    return YOLO(path)


class ModelManager:
    """Keeps loaded models in a dict, one per model file.

    get()    -> loads on the FIRST request, afterwards returns the same object
    unload() -> waits for a running prediction, removes the model and frees GPU memory;
                nothing else ever removes it
    """

    def __init__(self, loader=None, verbose=True):
        self._loader = loader or _default_loader
        self._models = {}
        self._lock = threading.RLock()
        self.verbose = verbose
        self.load_count = 0        # how many times a model was really loaded

    @staticmethod
    def _key(path):
        return str(Path(path).expanduser().resolve())

    def is_loaded(self, path):
        return self._key(path) in self._models

    def loaded_paths(self):
        return list(self._models)

    def get(self, path, device=None, imgsz=640, warmup=True):
        key = self._key(path)
        entry = self._models.get(key)          # fast path: already loaded, no lock needed
        if entry is not None:
            return entry
        with self._lock:                       # slow path: only the first call ever gets here
            entry = self._models.get(key)      # another thread may have loaded it meanwhile
            if entry is not None:
                return entry
            if not Path(key).exists():
                raise AnalyzerError(f"model not found: {key}")
            t0 = time.time()
            model = self._loader(key)
            names = dict(model.names)
            with as_error():
                role_ids = role_ids_from_names(names)
            self.load_count += 1
            entry = LoadedModel(path=key, model=model, role_ids=role_ids,
                                tubing_id=tubing_id_from_names(names),
                                task=getattr(model, "task", "segment"),
                                names=names, serial=self.load_count,
                                tubing_ids=tubing_ids_from_names(names))
            if warmup:
                self._warmup(entry, device, imgsz)
            self._models[key] = entry          # published only when fully ready
            if self.verbose:
                print(f"[model] loaded {key} in {time.time() - t0:.1f}s "
                      f"(task={entry.task}, classes={names})")
            return entry

    @staticmethod
    def _warmup(entry, device, imgsz):
        """One dummy prediction so the first real frame is not slow (CUDA start-up)."""
        try:
            with entry.infer_lock:
                entry.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz,
                                    device=device, verbose=False)
        except Exception as e:     # a failed warm-up must never stop the pipeline
            print(f"[model] warm-up skipped: {e}")

    def unload(self, path=None):
        """Unload one model (path) or all of them. Returns how many were unloaded.
        A prediction running on that model right now (from ANY session) finishes first."""
        with self._lock:
            keys = [self._key(path)] if path else list(self._models)
            removed = [self._models.pop(k, None) for k in keys]
        n = 0
        for e in removed:
            if e is not None:
                with e.infer_lock:         # wait for an in-flight predict, then drop it
                    e.model = None
                n += 1
        del removed
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if self.verbose:
            print(f"[model] unloaded {n} model(s)" if n else "[model] nothing to unload")
        return n


MODELS = ModelManager()            # shared by every analyzer unless you pass your own


def load_model(model_path, device=None, imgsz=640, manager=None):
    """Load a model file once (does nothing if it is already loaded)."""
    return (manager or MODELS).get(model_path, device=device, imgsz=imgsz)


def unload_model(model_path=None, manager=None):
    """Unload one model file, or every loaded model if model_path is None."""
    return (manager or MODELS).unload(model_path)


def is_model_loaded(model_path, manager=None):
    return (manager or MODELS).is_loaded(model_path)


def configure_for_model(cfg, entry):
    """The decisions run() makes right after loading a model, for a given cfg.
    Returns False when the model has no masks (so mask outlines must stay off)."""
    masks_ok = True
    if entry.task != "segment":
        if cfg.width_method != "off":
            print("[width] model is not a segmentation model -> no masks, width turned OFF")
            cfg.width_method = "off"
        if getattr(cfg, "tube_color", False):
            print("[color] model is not a segmentation model -> no masks, tube colour turned OFF")
            cfg.tube_color = False
        masks_ok = False
    if cfg.width_method in ("auto", "tubing"):
        if entry.tubing_id is None and cfg.width_method == "tubing":
            raise AnalyzerError(f"width_method=tubing but the model has no tubing class: {entry.names}")
        cfg.width_method = "tubing" if entry.tubing_id is not None else "mask"
    check_class_conf(cfg, entry.names, entry.role_ids, entry.tubing_ids)
    return masks_ok


def check_class_conf(cfg, names, role_ids, tubing_ids):
    """Warn about class_conf keys the model does not have (typo -> silently default conf)."""
    cc = getattr(cfg, "class_conf", None) or {}
    unknown = sorted(set(cc) - set(names.values()))
    if unknown:
        print(f"[conf] WARNING class_conf names not in the model, ignored: {unknown} "
              f"(model classes: {list(names.values())})")
    print(f"[conf] per class: {conf_table(names, role_ids, tubing_ids, cfg)}")


# ------------------------------------------------------------- input / config
def as_bgr(frame):
    """Any numpy image -> 3-channel BGR (gray and BGRA are converted)."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty numpy image (BGR, as read by cv2)")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"unsupported frame shape {frame.shape}")


def config_video(config):
    """The `video:` value written in config.yaml (made absolute), or None."""
    if not config:
        return None
    path = Path(config).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _resolve(data.get("video"), path.parent)


def load_stream_config(config=None, use_roi=True, **overrides):
    """Config for frame-by-frame use (no video needed). Returns (cfg, roi).
    config = path to config.yaml, or None to use only DEFAULTS + overrides
    (then model_path=... must be passed)."""
    user_video = overrides.pop("video", None)
    with as_error():
        # load_config insists on a video; a frame-by-frame backend has none, so give
        # it a placeholder and put the real value (if any) back afterwards.
        cfg = load_config(config, video="0", **overrides)
        if not use_roi:
            cfg.roi_json = None
        roi = load_roi(cfg.roi_json)
    cfg.video = _resolve(user_video, Path.cwd()) if user_video else config_video(config)
    for p in (cfg.latest_json, cfg.history_jsonl):
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
    return cfg, roi


# ---------------------------------------------------------- per-stream state
class StreamState:
    """Everything one video stream remembers between frames."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.frame_index = 0
        self.peak = {}
        self.last_t = None
        self.fps = None              # processing speed, smoothed
        self.frame_total = None      # frames in the video (None = stream / webcam)
        self.video_fps = None        # the video's own frame rate
        self.last_draw = None        # (frame, dets snapshot, result, roi) of the last annotated frame
        self.end_classes = None      # head/tail class names of the model (set on the first frame)
        self.tubes = TubeTracker()   # tube tracks (track_tubes)
        self.tube_classes = []
        self.trace_hist = {}         # {(tube id, side): recent end X/Y/Z} for end_xyz_m

    def summary(self, cfg):
        """Whole-stream result so far (see summarize)."""
        return summarize(self.tracks, cfg, self.end_classes, tubes=self.tubes,
                         tube_classes=self.tube_classes)

    def tick_fps(self):
        now = time.time()
        if self.last_t is not None and now > self.last_t:
            inst = 1.0 / (now - self.last_t)
            self.fps = inst if self.fps is None else 0.8 * self.fps + 0.2 * inst
        self.last_t = now
        return round(self.fps, 1) if self.fps else None

    def totals(self):
        """Distinct confirmed heads / tails that appeared in the ROI so far."""
        return {f"{role.lower()}s_total": sum(t.role == role and t.confirmed and t.in_roi_ever
                                              for t in self.tracks) for role in ROLES}


# ------------------------------------------------------------ one frame, all steps
def analyze_frame(entry, frame, state, roi, cfg, frame_index=None, annotate=False, masks=True,
                  depth=None, show_cards=None):
    """The full pipeline for ONE frame: detect -> track -> ROI -> sizes -> result
    (-> drawing). Returns (result dict, annotated frame or None).
    depth: uint16 depth frame aligned to `frame` -> X/Y/Z per end (None = no 3D)."""
    frame = as_bgr(frame)
    cfg.draw_masks = bool(masks and annotate)                 # outlines only when drawing
    state.frame_index = int(frame_index) if frame_index is not None else state.frame_index + 1
    idx = state.frame_index

    if depth is not None and not cfg.camera_matrix:
        raise AnalyzerError("a depth frame was passed but camera_matrix is not set in the config")
    tub = entry.tubing_ids or entry.tubing_id
    tubes, info = [], {}
    dets = detect(entry.model, frame, entry.role_ids, cfg, tub, lock=entry.infer_lock,
                  depth=depth, names=entry.names, tubes_out=tubes, info=info)
    t_after = time.perf_counter()
    state.next_id = update_tracks(state.tracks, dets, idx, state.next_id, cfg)   # ALL detections
    shown = [d for d in dets if in_roi(d, roi)]                                    # ROI = output only
    shown = cap_per_role(shown, cfg)
    for d in shown:
        d["track"].in_roi_ever = True
    bigger = assign_sizes(shown, cfg)
    state.end_classes = entry.end_classes
    tube_classes = [entry.names[i] for i in entry.tubing_ids]
    state.tube_classes = tube_classes
    tubes, tube_totals = output_tubes(tubes, roi, state.tubes, idx, cfg, tube_classes)
    traces = collect_traces(tubes, state.trace_hist, cfg) if cfg.trace_px else None
    result = build_frame_result(idx, shown, state.tracks, bigger, cfg, entry.end_classes,
                                tubes, tube_classes, tube_totals)
    if cfg.trace_px:
        result["trace_px"] = cfg.trace_px
        result["wrapper"] = info.get("wrapper")
        result["traces"] = traces
    for role in ROLES:                                        # most ends seen at once so far
        key = role.lower() + "s"
        state.peak[key] = max(state.peak.get(key, 0), result[f"{key}_count"])
        result[f"{key}_peak"] = state.peak[key]
    result["frame_total"] = state.frame_total
    result["fps"] = state.tick_fps()
    result["video_fps"] = state.video_fps
    result["time_s"] = round((idx - 1) / state.video_fps, 3) if state.video_fps else None
    result["depth"] = depth is not None

    if cfg.print_frames:
        frame_summary(result, cfg)
    if cfg.latest_json:
        Path(cfg.latest_json).write_text(json.dumps(result, indent=2))
    if cfg.history_jsonl:
        with open(cfg.history_jsonl, "a") as f:
            f.write(json.dumps(result) + "\n")
    PROF.add("track+json", time.perf_counter() - t_after)
    vis = None
    if annotate:
        with PROF.time("draw"):
            vis = draw(frame, shown, result, roi, cfg, show_cards=show_cards)
        # lets the preview redraw this frame (c / p keys, stepping back)
        state.last_draw = (frame, snapshot_dets(shown), result, roi)
    return result, vis


def snapshot_dets(dets):
    """Frozen copy of what draw() needs, so an older frame can be redrawn later
    (tracks keep changing while the video runs)."""
    out = []
    for d in dets:
        t = d["track"]
        snap = SimpleNamespace(id=t.id, role=t.role, confirmed=t.confirmed, width=t.width,
                               xyz=t.xyz, width_mm_depth=t.width_mm_depth, color=t.color,
                               cls=t.cls)
        out.append({"track": snap, "role": d.get("role"), "conf": d["conf"], "xyxy": d["xyxy"],
                    "size": d.get("size"), "outline": d.get("outline")})
    return out


# ------------------------------------------------------------- video in / out
def open_video(video):
    """-> (cv2.VideoCapture, fps, total_frames or None). webcam = "0"."""
    cap = cv2.VideoCapture(int(video) if str(video).isdigit() else str(video))
    if not cap.isOpened():
        raise AnalyzerError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, (total if total > 0 else None)


class VideoOutput:
    """Annotated .mp4 writer and/or live preview window.
    write() returns False when the user pressed q in the preview.

    Preview keys (while inference runs):
        c   result cards on / off      (on_key["c"])
        p   panel details on / off     (on_key["p"])
        q   quit
    The change shows at once (the frame on screen is redrawn) and is used for every
    frame after it - also in the saved .mp4 from that point on."""

    WIN = "tube frames  (c = cards on/off | p = panel on/off | q = quit)"

    def __init__(self, path=None, fps=30.0, preview=False, threaded=True):
        self.path, self.fps, self.preview = path, fps, preview
        self.writer = None
        self.on_key = {}             # {key char: callback} for the preview window, e.g. {"c": toggle}
        self.threaded = threaded     # encode the .mp4 in a background thread (it costs ~10-20 ms/frame)
        self._q = None
        self._thread = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _writer_loop(self):
        while True:
            img = self._q.get()
            if img is None:
                break
            self.writer.write(img)

    def _show(self, img):
        s = 900 / img.shape[0]
        cv2.imshow(self.WIN, cv2.resize(img, None, fx=s, fy=s))

    def write(self, vis, redraw=None):
        """vis = drawn frame; redraw() = the same frame drawn again with the current
        settings (so c / p show at once instead of on the next frame)."""
        if self.path and self.writer is None:
            h, w = vis.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (w, h))
            if not self.writer.isOpened():
                raise AnalyzerError(f"cannot write {self.path} - must be a .mp4 file path, not a folder")
        if self.writer is not None:
            if self.threaded:
                if self._thread is None:
                    import queue
                    self._q = queue.Queue(maxsize=8)
                    self._thread = threading.Thread(target=self._writer_loop, daemon=True)
                    self._thread.start()
                self._q.put(vis)             # draw() returns a new image every frame: no copy needed
            else:
                self.writer.write(vis)
        if self.preview:
            try:
                self._show(vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return False
                if key != 255 and chr(key) in self.on_key:
                    on = self.on_key[chr(key)]()
                    name = "cards" if chr(key) == "c" else "panel details"
                    print(f"\n[preview] {name} {'on' if on else 'off'}", flush=True)
                    img = redraw() if redraw is not None else None
                    if img is not None:
                        self._show(img)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            return False
            except cv2.error:
                self.preview = False
                print("[info] no GUI support in this OpenCV build - preview disabled")
        return True

    def close(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join()
            self._thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


class FrameSource:
    """(index, BGR frame, depth frame or None) for every processed frame of a video.
    prefetch=True: a background thread decodes the NEXT frames (RGB + depth) while
    the GPU works on this one. Depth is read for EVERY frame (also the ones stride
    skips) so RGB and depth stay in step."""

    def __init__(self, cap, depth_reader=None, stride=1, prefetch=True, queue_size=4):
        import queue
        self.cap, self.depth_reader = cap, depth_reader
        self.stride = max(int(stride or 1), 1)
        self.prefetch = prefetch
        self.frames_read = 0
        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._error = None
        self._thread = None

    def _frames(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                return
            depth = self.depth_reader.read() if self.depth_reader else None
            self.frames_read += 1
            if self.frames_read % self.stride:
                continue
            yield self.frames_read, frame, depth

    def _put(self, item):
        import queue
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _worker(self):
        try:
            for item in self._frames():
                self._put(item)
        except Exception as e:           # surfaced in the consumer
            self._error = e
        finally:
            self._put(None)

    def __iter__(self):
        if not self.prefetch:
            yield from self._frames()
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        while True:
            item = self._q.get()
            if item is None:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def close(self):
        """Stop the reader thread (safe to call twice). Call before releasing cap / depth."""
        self._stop.set()
        if self._thread is not None:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break
            self._thread.join(timeout=5)
            self._thread = None


# ------------------------------------------------------------------------ run
def run(cfg):
    """Process the video described by `cfg` (from load_config). Yields one result
    dict per processed frame and writes the video / JSON outputs set in cfg.
    Same per-frame pipeline as main.HeadTailAnalyzer (analyze_frame)."""
    user_masks = bool(cfg.draw_masks)
    annotate = bool(cfg.output_video or cfg.show_preview)
    with as_error():
        entry = MODELS.get(cfg.model_path, device=cfg.device, imgsz=int(cfg.imgsz or cfg.tile or 640))
        masks_ok = configure_for_model(cfg, entry) and user_masks
    print(f"[model] {cfg.model_path}  (task={entry.task})")
    print(f"[classes] {entry.names} -> heads/tails {entry.role_ids}")
    print(f"[width] {cfg.width_method}   [size] bigger tube from "
          f"{'all ends' if cfg.width_from == 'all' else 'tails'}, min ratio {cfg.min_ratio}")
    if cfg.trace_px:
        wid = wrapper_ids_from_names(entry.names)
        print(f"[trace] {cfg.trace_px:g} px along each tube out of the wrapper "
              f"(wrapper class ids {wid or 'NONE - no wrapper class in this model!'})")
    roi = load_roi(cfg.roi_json)
    print(f"[roi] {f'{len(roi)}-point polygon' if roi is not None else 'none - whole frame'}")
    print(f"[input] {'grayscale' if cfg.gray else 'colour'}, tile={cfg.tile or 'off'}, "
          f"imgsz={cfg.imgsz}, half={getattr(cfg, 'half', False)}, conf={cfg.conf}, device={cfg.device}")
    for p in (cfg.latest_json, cfg.history_jsonl, cfg.output_video):
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    cap, fps, total = open_video(cfg.video)
    state = StreamState()
    state.frame_total, state.video_fps = total, round(fps, 2)
    depth_reader = depth3d.open_depth(cfg, fps, video=cfg.video)
    out = VideoOutput(cfg.output_video, fps / cfg.stride, cfg.show_preview) if annotate else None
    if out:
        def _toggle():
            cfg.show_cards = not cfg.show_cards
            return cfg.show_cards
        out.on_key["c"] = _toggle
    src = FrameSource(cap, depth_reader, cfg.stride, getattr(cfg, "prefetch", True))
    t0, n = time.time(), 0
    try:
        for idx, frame, depth in src:
            result, vis = analyze_frame(entry, frame, state, roi, cfg, idx, annotate=annotate,
                                        masks=masks_ok, depth=depth)
            if out and not out.write(vis):
                break
            n += 1
            yield result
            if idx % 10 == 0:
                print(f"[progress] frame {idx}/{total or '?'}  {n / (time.time() - t0):.2f} frames/s",
                      flush=True)
            if getattr(cfg, "profile", False) and n % 30 == 0:
                print(PROF.report(), flush=True)
    finally:
        src.close()
        cap.release()
        if depth_reader:
            report_depth_sync(depth_reader, src.frames_read)
            depth_reader.close()
        if out:
            out.close()
    print_diagnosis(state.tracks, cfg.min_frames, cfg.mm_per_px)
    if getattr(cfg, "summary_json", None):
        Path(cfg.summary_json).write_text(json.dumps(state.summary(cfg), indent=2))
        print(f"[summary] {cfg.summary_json}")


def report_depth_sync(reader, rgb_frames):
    """Warn when the RGB and depth videos do not have the same number of frames."""
    if reader.ended and reader.frames_read < rgb_frames:
        print(f"[depth] WARNING: depth video ended after {reader.frames_read} frames but RGB "
              f"has {rgb_frames} - later frames had no depth, check the recordings are a pair")
    elif not reader.ended and reader.read() is not None:
        print(f"[depth] NOTE: depth video has more frames than the {rgb_frames} RGB frames read "
              "- if this is the full video, the pair may be out of sync")
    else:
        print(f"[depth] RGB and depth in step: {reader.frames_read} depth frames")


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Count heads and tails per video frame, measure widths, find the bigger tube.",
        epilog="Any flag overrides the same setting in the config file.")
    ap.add_argument("--config", help="config.yaml (paths inside it are relative to its folder)")
    ap.add_argument("--video", help="video file, or 0 for webcam")
    ap.add_argument("--model", help="trained YOLO11-seg weights (.pt)")
    ap.add_argument("--output", help="annotated video, e.g. out.mp4")
    ap.add_argument("--json", help="latest-frame JSON, overwritten every frame")
    ap.add_argument("--history", help="JSONL with one line per frame")
    ap.add_argument("--summary", help="whole-video JSON (every confirmed end), written at the end")
    ap.add_argument("--roi", help="roi.json polygon")
    ap.add_argument("--no-roi", action="store_true", help="ignore any ROI, use the whole frame")
    ap.add_argument("--draw-roi", action="store_true",
                    help="click an ROI polygon on the first frame, save it (to --roi, else the "
                         "config's roi_json, else roi.json next to the config), then run with it")
    ap.add_argument("--roi-frame", type=int, default=0, help="frame number to draw the ROI on")
    ap.add_argument("--device", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--stride", type=int, help="process every Nth frame")
    ap.add_argument("--scale", type=float, help="shrink frames before inference, e.g. 0.33 when ends look huge")
    ap.add_argument("--color", action="store_true",
                    help="feed colour frames (only if the model was trained on colour)")
    ap.add_argument("--width-method", choices=["auto", "tubing", "mask", "off"],
                    help="how width is measured (default auto)")
    ap.add_argument("--width-from", choices=["tail", "all"],
                    help="which ends decide the bigger tube (default tail)")
    ap.add_argument("--min-ratio", type=float, help="width ratio to call one BIGGER (default 1.15)")
    ap.add_argument("--mm-per-px", type=float, help="millimetres per pixel, for widths in mm")
    ap.add_argument("--sat-threshold", type=float,
                    help="median saturation above this = blue tube (default 60, see check_tube_color.py)")
    ap.add_argument("--no-tube-color", action="store_true", help="don't label ends blue / white")
    ap.add_argument("--color-window", type=int, help="colour label = median of the last N frames (default 10)")
    ap.add_argument("--no-color-gate", action="store_true",
                    help="allow a blue detection to match a white track (old behaviour)")
    ap.add_argument("--depth", help="16-bit depth video aligned to --video (X/Y/Z per end)")
    ap.add_argument("--no-depth", action="store_true", help="ignore depth_video from the config")
    ap.add_argument("--draw-scale", type=float,
                    help="size of boxes/labels in the output video (default auto, e.g. 2 = twice as big)")
    ap.add_argument("--box-fill", type=float, help="translucent fill inside boxes, 0-1 (default 0.2)")
    ap.add_argument("--no-masks", action="store_true", help="don't draw mask outlines")
    ap.add_argument("--hide-unconfirmed", action="store_true", help="don't draw grey not-yet-confirmed boxes")
    ap.add_argument("--no-legend", action="store_true", help="don't draw the colour key")
    ap.add_argument("--no-cards", action="store_true",
                    help="boxes only, no result cards next to them (results stay in the top-left panel)")
    ap.add_argument("--no-panel-details", action="store_true",
                    help="top-left panel shows only frame / FPS / counts, not the per-end results")
    ap.add_argument("--no-tubes", action="store_true",
                    help="no 'tubes' list (white_tube / blue_tube bbox + polygon) in the JSON / video")
    ap.add_argument("--no-polygons", action="store_true", help="no head/tail 'polygon' in the JSON")
    ap.add_argument("--trace-px", type=float,
                    help="follow each tube this many px out of the wrapper -> X/Y/Z of the end")
    ap.add_argument("--half", action="store_true", help="FP16 inference (faster on GPU)")
    ap.add_argument("--profile", action="store_true", help="print ms per stage every 30 frames")
    ap.add_argument("--preview", action="store_true", help="show a live window (q to stop)")
    ap.add_argument("--quiet", action="store_true", help="don't print every frame")
    a = ap.parse_args(argv)

    # installed wheel: with no --config, use the config shipped inside the package
    bundled = Path(__file__).parent / "default_config.yaml"
    config = a.config or (str(bundled) if bundled.exists() else None)

    cfg = load_config(
        config,
        model_path=a.model, video=a.video, output_video=a.output,
        latest_json=a.json, history_jsonl=a.history, summary_json=a.summary, roi_json=a.roi,
        device=a.device, conf=a.conf, stride=a.stride, scale=a.scale,
        gray=False if a.color else None,
        show_preview=True if a.preview else None,
        print_frames=False if a.quiet else None,
        width_method=a.width_method, width_from=a.width_from,
        min_ratio=a.min_ratio, mm_per_px=a.mm_per_px, depth_video=a.depth,
        sat_threshold=a.sat_threshold,
        tube_color=False if a.no_tube_color else None,
        color_window=a.color_window,
        color_gate=False if a.no_color_gate else None,
        draw_scale=a.draw_scale, box_fill=a.box_fill,
        draw_masks=False if a.no_masks else None,
        draw_unconfirmed=False if a.hide_unconfirmed else None,
        draw_legend=False if a.no_legend else None,
        show_cards=False if a.no_cards else None,
        panel_details=False if a.no_panel_details else None,
        output_tubes=False if a.no_tubes else None,
        draw_tubes=False if a.no_tubes else None,
        output_polygons=False if a.no_polygons else None,
        trace_px=a.trace_px, half=True if a.half else None,
        profile=True if a.profile else None,
    )
    if a.draw_roi:
        base = Path(a.config).expanduser().resolve().parent if a.config else Path.cwd()   # never inside the package
        target = cfg.roi_json or str(base / "roi.json")
        if draw_roi_interactive(cfg.video, target, a.roi_frame) is None:
            raise SystemExit("no ROI drawn - stopped")
        cfg.roi_json = target
    elif a.no_roi:
        cfg.roi_json = None
    if a.no_depth:
        cfg.depth_video = None

    n = sum(1 for _ in run(cfg))
    print(f"\n[done] {n} frames processed")
    for label, p in (("video", cfg.output_video), ("json", cfg.latest_json),
                     ("history", cfg.history_jsonl), ("summary", cfg.summary_json)):
        if p:
            print(f"  {label}: {p}")


if __name__ == "__main__":
    main()