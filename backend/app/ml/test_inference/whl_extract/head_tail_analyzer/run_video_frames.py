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

OUTPUT VIDEO DRAWING:
  * line width and text size scale with the frame size (draw_scale overrides)
  * one colour per role, whatever the size: HEAD = cyan, TAIL = yellow
  * thin box + a compact callout card joined by a leader line:
        ID: 3  Head / Conf: 0.92 / Smaller  18.4 px
    (the px value = measured tube width, the number that decides Bigger/Smaller/Same)
  * the segmentation mask outline of each end is drawn too (draw_masks)
  * top-left info panel: FRAME n / total, FPS (processing speed), HEADS (cyan), TAILS (yellow)

Per frame (terminal + JSON), all restricted to the ROI:
  heads_count / tails_count      confirmed ends visible in this frame
  heads_visible / tails_visible  same as *_count (kept for the frontend)
  heads_total / tails_total      distinct confirmed ends that ever appeared in the ROI
  detections                     id, role, confidence, bbox [x, y, w, h], center,
                                 width_px, width_mm, size, width_source
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
except ImportError:          # run as a plain script next to those files
    from measure_from_masks import attach, local_axis
    from tube_ends import measure_width

DEFAULTS = {
    "model_path": None,       # trained YOLO11-seg weights (required)
    "video": None,            # video file, or "0" for webcam 0 (required)
    "device": "0",            # "0" = first GPU, "cpu" = CPU
    "conf": 0.35,
    "conf_head": None,        # per-role confidence; None = use conf (tails usually score lower)
    "conf_tail": None,
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
    "print_frames": True,
    "show_preview": False,
    # ---- drawing (output video / preview) ----
    "draw_scale": None,       # size of lines + text; None = auto from frame size (1.0 at 1280 px)
    "box_fill": 0.2,          # translucent fill inside each confirmed box (0 = off, 0.4 = strong)
    "draw_masks": True,       # outline the segmentation mask of each end
    "draw_unconfirmed": True, # thin grey boxes for ends not yet confirmed
    "draw_legend": False,     # no longer drawn (kept so old config files still load)
    # ---- width / bigger tube ----
    "width_method": "auto",   # auto | tubing | mask | off  (see docstring)
    "width_from": "tail",     # which ends decide bigger_tube: tail | all
    "min_ratio": 1.15,        # width ratio needed to call one BIGGER (1.15 = 15 %)
    "mm_per_px": None,        # set for millimetres (ruler / ArUco calibration)
    "attach_gap": 3,          # px gap tolerated between an end mask and its tubing mask
    "max_disagree": 0.2,      # edge vs mask width disagreement before falling back to mask width
}
PATH_KEYS = ("model_path", "video", "roi_json", "output_video", "latest_json", "history_jsonl")

ROLES = ("HEAD", "TAIL")
COLORS = {"HEAD": (255, 230, 0), "TAIL": (0, 220, 255)}   # BGR: cyan head, yellow tail (any size)
COL_BIG = (60, 230, 60)                                   # green = bigger tube
COL_PENDING = (170, 170, 170)
COL_SAME = (220, 90, 200)                                 # purple = tubes same size
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------- config
def _resolve(value, base):
    """Make a path absolute against `base`. Empty -> None, digits -> webcam index."""
    if value is None or str(value).strip() == "":
        return None
    s = str(value).strip()
    if s.isdigit():
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
    return SimpleNamespace(**cfg)


# ------------------------------------------------------------------ detection
def role_ids_from_names(names):
    """{class_id: 'HEAD'/'TAIL'} - tubing and anything else is ignored."""
    ids = {int(i): n.upper() for i, n in names.items() if n.upper() in ROLES}
    missing = set(ROLES) - set(ids.values())
    if missing:
        raise SystemExit(f"model classes {names} have no {missing} class")
    return ids


def tubing_id_from_names(names):
    for i, n in names.items():
        if "tub" in n.lower():
            return int(i)
    return None


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


def _det(role, conf, box, mask=None):
    return {"role": role, "conf": conf, "xyxy": box, "mask": mask,
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
                        kept[i] = _det(a["role"], max(a["conf"], b["conf"]), box, mask)
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


def mask_outline(mask, k):
    """Largest outer contour of a mask, back in ORIGINAL pixel coordinates."""
    if mask is None or not mask.any():
        return None
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).astype(np.float32)
    return np.round(c / k).astype(np.int32)


def detect(model, frame, role_ids, cfg, tubing_id=None, lock=None):
    """lock: held only around model.predict (a YOLO model must not run two
    predictions at once); the post-processing below runs without it."""
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

    want_width = cfg.width_method != "off"
    want_masks = want_width or cfg.draw_masks
    classes = list(role_ids) + ([tubing_id] if (want_width and tubing_id is not None) else [])
    role_conf = {"HEAD": cfg.conf if cfg.conf_head is None else cfg.conf_head,
                 "TAIL": cfg.conf if cfg.conf_tail is None else cfg.conf_tail}
    with (lock or nullcontext()):
        results = model.predict(crops, conf=min(role_conf.values()), iou=cfg.iou,
                                classes=classes, imgsz=int(cfg.imgsz or t or 640),
                                device=cfg.device, retina_masks=want_masks,
                                verbose=False)
    dets, tub_masks = [], []
    for r, (ox, oy) in zip(results, offsets):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        ms = (r.masks.data.cpu().numpy() > 0.5) if (want_masks and r.masks is not None) else None
        for j, (b, c, s) in enumerate(zip(xyxy, cls, conf)):
            full = None
            if ms is not None:                    # tile mask -> full (scaled) frame
                m = ms[j]
                th, tw = min(m.shape[0], h - oy), min(m.shape[1], w - ox)
                full = np.zeros((h, w), bool)
                full[oy:oy + th, ox:ox + tw] = m[:th, :tw]
            if tubing_id is not None and c == tubing_id:
                if full is not None:
                    tub_masks.append(full)
                continue
            if c not in role_ids or s < role_conf[role_ids[c]]:
                continue
            box = [float(b[0] + ox) / k, float(b[1] + oy) / k,
                   float(b[2] + ox) / k, float(b[3] + oy) / k]   # back to original pixels
            dets.append(_det(role_ids[c], float(s), box, full))
    dets = merge_duplicates(dets, cfg)

    if want_width:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tubings = tube_components(tub_masks)
        for d in dets:
            wpx, src = measure_end(d.get("mask"), tubings, gray, cfg)
            d["width_px"] = wpx / k if wpx else None          # back to original pixels
            d["width_src"] = src
    for d in dets:
        m = d.pop("mask", None)                                # free memory, keep only the outline
        d["outline"] = mask_outline(m, k) if cfg.draw_masks else None
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
        self.update(det, idx)

    def update(self, det, idx):
        self.center = det["center"]
        self.xyxy = det["xyxy"]
        self.conf = det["conf"]
        self.last_seen = idx
        self.hits += 1
        if det.get("width_px"):
            self.widths.append(det["width_px"])
            self.width_src = det.get("width_src")

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


def update_tracks(tracks, dets, idx, next_id, cfg):
    """Greedy nearest-center matching, per role (a head never matches a tail)."""
    for role in ROLES:
        live = [t for t in tracks if t.role == role and idx - t.last_seen <= cfg.max_missed]
        cand = [d for d in dets if d["role"] == role]
        pairs = []
        for ti, t in enumerate(live):
            for di, d in enumerate(cand):
                dist = float(np.hypot(t.center[0] - d["center"][0],
                                      t.center[1] - d["center"][1]))
                limit = cfg.max_dist_tail if (role == "TAIL" and cfg.max_dist_tail) else cfg.max_dist
                if dist <= limit:
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


def build_frame_result(idx, dets, tracks, bigger, cfg):
    """`dets` = this frame's detections INSIDE the ROI."""
    counts = {}
    for role in ROLES:
        key = role.lower() + "s"
        visible = sum(d["track"].confirmed and d["role"] == role for d in dets)
        counts[f"{key}_count"] = visible
        counts[f"{key}_visible"] = visible
        counts[f"{key}_total"] = sum(t.role == role and t.confirmed and t.in_roi_ever
                                     for t in tracks)

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
        })
    return {"frame": idx, **counts, "detections": detections, "bigger_tube": bigger}


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
        print(f"  #{d['id']:<3} {d['role']:4s} conf={d['conf']:.2f} "
              f"width={fmt_w(d['width_px'], cfg.mm_per_px):>9} {d['size'] or '':7s} bbox={d['bbox']}")
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


def draw(frame, dets, result, roi, cfg):
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
    for d in sorted((d for d in shown if d["track"].confirmed),
                    key=lambda d: d["conf"], reverse=True):
        t = d["track"]
        x1b, y1b, x2b, y2b = d["xyxy"]
        title = f"ID: {t.id}  {t.role.title()}"
        body = [f"Conf: {d['conf']:.2f}",
                f"{(d.get('size') or '-').title()}  {_px(t.width, cfg.mm_per_px)}"]
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
            print(f"    id={t.id:<4} frames {t.first_seen}-{t.last_seen}  seen {t.hits}x  "
                  f"width={fmt_w(t.width, mm_per_px)} over {len(t.widths)} frames  "
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
    tubing_id: Optional[int]
    task: str                      # "segment" or "detect"
    names: dict
    serial: int                    # increases with every load; changes after unload + reload
    infer_lock: Any = field(default_factory=threading.Lock, repr=False)
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
                                names=names, serial=self.load_count)
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
        masks_ok = False
    if cfg.width_method in ("auto", "tubing"):
        if entry.tubing_id is None and cfg.width_method == "tubing":
            raise AnalyzerError(f"width_method=tubing but the model has no tubing class: {entry.names}")
        cfg.width_method = "tubing" if entry.tubing_id is not None else "mask"
    return masks_ok


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
def analyze_frame(entry, frame, state, roi, cfg, frame_index=None, annotate=False, masks=True):
    """The full pipeline for ONE frame: detect -> track -> ROI -> sizes -> result
    (-> drawing). Returns (result dict, annotated frame or None)."""
    frame = as_bgr(frame)
    cfg.draw_masks = bool(masks and annotate)                 # outlines only when drawing
    state.frame_index = int(frame_index) if frame_index is not None else state.frame_index + 1
    idx = state.frame_index

    dets = detect(entry.model, frame, entry.role_ids, cfg, entry.tubing_id, lock=entry.infer_lock)
    state.next_id = update_tracks(state.tracks, dets, idx, state.next_id, cfg)   # ALL detections
    shown = [d for d in dets if in_roi(d, roi)]                                    # ROI = output only
    shown = cap_per_role(shown, cfg)
    for d in shown:
        d["track"].in_roi_ever = True
    bigger = assign_sizes(shown, cfg)
    result = build_frame_result(idx, shown, state.tracks, bigger, cfg)
    for role in ROLES:                                        # most ends seen at once so far
        key = role.lower() + "s"
        state.peak[key] = max(state.peak.get(key, 0), result[f"{key}_count"])
        result[f"{key}_peak"] = state.peak[key]
    result["frame_total"] = state.frame_total
    result["fps"] = state.tick_fps()
    result["video_fps"] = state.video_fps

    if cfg.print_frames:
        frame_summary(result, cfg)
    if cfg.latest_json:
        Path(cfg.latest_json).write_text(json.dumps(result, indent=2))
    if cfg.history_jsonl:
        with open(cfg.history_jsonl, "a") as f:
            f.write(json.dumps(result) + "\n")
    vis = draw(frame, shown, result, roi, cfg) if annotate else None
    return result, vis


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
    write() returns False when the user pressed q in the preview."""

    def __init__(self, path=None, fps=30.0, preview=False):
        self.path, self.fps, self.preview = path, fps, preview
        self.writer = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def write(self, vis):
        if self.path and self.writer is None:
            h, w = vis.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (w, h))
            if not self.writer.isOpened():
                raise AnalyzerError(f"cannot write {self.path} - must be a .mp4 file path, not a folder")
        if self.writer is not None:
            self.writer.write(vis)
        if self.preview:
            try:
                s = 900 / vis.shape[0]
                cv2.imshow("tube frames", cv2.resize(vis, None, fx=s, fy=s))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False
            except cv2.error:
                self.preview = False
                print("[info] no GUI support in this OpenCV build - preview disabled")
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ------------------------------------------------------------------------ run
def run(cfg):
    """Process the video described by `cfg` (from load_config). Yields one result
    dict per processed frame and writes the video / JSON outputs set in cfg."""
    from ultralytics import YOLO   # imported here so config errors show up fast

    model = YOLO(cfg.model_path)
    role_ids = role_ids_from_names(model.names)
    print(f"[model] {cfg.model_path}  (task={model.task})")
    print(f"[classes] {model.names} -> heads/tails {role_ids}")

    tubing_id = None
    if model.task != "segment":
        if cfg.width_method != "off":
            print("[width] model is not a segmentation model -> no masks, width turned OFF")
            cfg.width_method = "off"
        cfg.draw_masks = False
    # mask outlines are only needed when something is drawn
    if not (cfg.output_video or cfg.show_preview):
        cfg.draw_masks = False
    if cfg.width_method in ("auto", "tubing"):
        tubing_id = tubing_id_from_names(model.names)
        if tubing_id is None and cfg.width_method == "tubing":
            raise SystemExit(f"width_method=tubing but the model has no tubing class: {model.names}")
        cfg.width_method = "tubing" if tubing_id is not None else "mask"
    if cfg.width_method == "tubing":
        print(f"[width] tube diameter behind each end, from the tube walls (tubing class id {tubing_id})")
    elif cfg.width_method == "mask":
        print("[width] thickness of each head/tail mask (a head width = its connector width)")
        if cfg.width_from == "all":
            print("[width] WARNING width_from=all compares connector width with tube width - "
                  "use width_from=tail for a fair 'bigger tube'")
    print(f"[size] bigger tube decided by {'all ends' if cfg.width_from == 'all' else 'tails'}, "
          f"min ratio {cfg.min_ratio}, units {'mm' if cfg.mm_per_px else 'px'}")

    roi = load_roi(cfg.roi_json)
    print(f"[roi] {f'{len(roi)}-point polygon - only detections inside it are shown/counted' if roi is not None else 'none - whole frame shown'}")
    print(f"[input] {'grayscale' if cfg.gray else 'colour'}, tile={cfg.tile or 'off'}, "
          f"conf={cfg.conf}, device={cfg.device}")

    for p in (cfg.latest_json, cfg.history_jsonl, cfg.output_video):
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    src = int(cfg.video) if cfg.video.isdigit() else cfg.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {cfg.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks, next_id = [], 1
    peak = {}
    last_t, fps_now = None, None
    writer = None
    preview = cfg.show_preview
    idx, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % cfg.stride:
                continue

            dets = detect(model, frame, role_ids, cfg, tubing_id)
            next_id = update_tracks(tracks, dets, idx, next_id, cfg)   # ALL detections

            shown = [d for d in dets if in_roi(d, roi)]                # ROI = output only
            shown = cap_per_role(shown, cfg)
            for d in shown:
                d["track"].in_roi_ever = True
            bigger = assign_sizes(shown, cfg)
            result = build_frame_result(idx, shown, tracks, bigger, cfg)
            for role in ROLES:                       # most ends seen at once so far
                key = role.lower() + "s"
                peak[key] = max(peak.get(key, 0), result[f"{key}_count"])
                result[f"{key}_peak"] = peak[key]
            now = time.time()                        # processing speed, smoothed
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                fps_now = inst if fps_now is None else 0.8 * fps_now + 0.2 * inst
            last_t = now
            result["frame_total"] = total if total > 0 else None
            result["fps"] = round(fps_now, 1) if fps_now else None
            result["video_fps"] = round(fps, 2)
            if cfg.print_frames:
                frame_summary(result, cfg)

            if cfg.latest_json:
                Path(cfg.latest_json).write_text(json.dumps(result, indent=2))
            if cfg.history_jsonl:
                with open(cfg.history_jsonl, "a") as f:
                    f.write(json.dumps(result) + "\n")

            if cfg.output_video or preview:
                vis = draw(frame, shown, result, roi, cfg)
                if cfg.output_video and writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(cfg.output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / cfg.stride, (w, h))
                    if not writer.isOpened():
                        raise SystemExit(f"cannot write {cfg.output_video} - must be a "
                                         ".mp4 file path, not a folder")
                if writer:
                    writer.write(vis)
                if preview:
                    try:
                        s = 900 / vis.shape[0]
                        cv2.imshow("tube frames", cv2.resize(vis, None, fx=s, fy=s))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        preview = False
                        print("[info] no GUI support in this OpenCV build - preview disabled")

            yield result

            if idx % 10 == 0:
                rate = idx / (time.time() - t0)
                print(f"[progress] frame {idx}/{total}  {rate:.2f} frames/s", flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
        if preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    print_diagnosis(tracks, cfg.min_frames, cfg.mm_per_px)


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
    ap.add_argument("--draw-scale", type=float,
                    help="size of boxes/labels in the output video (default auto, e.g. 2 = twice as big)")
    ap.add_argument("--box-fill", type=float, help="translucent fill inside boxes, 0-1 (default 0.2)")
    ap.add_argument("--no-masks", action="store_true", help="don't draw mask outlines")
    ap.add_argument("--hide-unconfirmed", action="store_true", help="don't draw grey not-yet-confirmed boxes")
    ap.add_argument("--no-legend", action="store_true", help="don't draw the colour key")
    ap.add_argument("--preview", action="store_true", help="show a live window (q to stop)")
    ap.add_argument("--quiet", action="store_true", help="don't print every frame")
    a = ap.parse_args(argv)

    # installed wheel: with no --config, use the config shipped inside the package
    bundled = Path(__file__).parent / "default_config.yaml"
    config = a.config or (str(bundled) if bundled.exists() else None)

    cfg = load_config(
        config,
        model_path=a.model, video=a.video, output_video=a.output,
        latest_json=a.json, history_jsonl=a.history, roi_json=a.roi,
        device=a.device, conf=a.conf, stride=a.stride, scale=a.scale,
        gray=False if a.color else None,
        show_preview=True if a.preview else None,
        print_frames=False if a.quiet else None,
        width_method=a.width_method, width_from=a.width_from,
        min_ratio=a.min_ratio, mm_per_px=a.mm_per_px,
        draw_scale=a.draw_scale, box_fill=a.box_fill,
        draw_masks=False if a.no_masks else None,
        draw_unconfirmed=False if a.hide_unconfirmed else None,
        draw_legend=False if a.no_legend else None,
    )
    if a.draw_roi:
        base = Path(a.config).expanduser().resolve().parent if a.config else Path.cwd()   # never inside the package
        target = cfg.roi_json or str(base / "roi.json")
        if draw_roi_interactive(cfg.video, target, a.roi_frame) is None:
            raise SystemExit("no ROI drawn - stopped")
        cfg.roi_json = target
    elif a.no_roi:
        cfg.roi_json = None

    n = sum(1 for _ in run(cfg))
    print(f"\n[done] {n} frames processed")
    for label, p in (("video", cfg.output_video), ("json", cfg.latest_json),
                     ("history", cfg.history_jsonl)):
        if p:
            print(f"  {label}: {p}")


if __name__ == "__main__":
    main()