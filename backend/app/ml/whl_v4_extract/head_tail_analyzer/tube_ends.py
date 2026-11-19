"""
tube_ends.py - MEASUREMENT LIBRARY used by measure_from_masks.py / run_video_frames.py.

Only measure_width(), transparency_score(), colour_score() (and their helpers)
are used by the current pipeline. They work on pixel coordinates only (a tip
point and a point back along the tube axis) and know NOTHING about class
names - that matching is done in measure_from_masks.py's build_class_map(),
which reads the names straight from your trained model. So there is no class
list to keep "in sync" with your data.yaml here.

CLASS_NAMES / ROLE / EndTrack / process() / run_video() / main() below are a
SEPARATE, OLDER prototype pipeline for a YOLO-POSE model (tip + axis
keypoints), not the YOLO11-SEG model you actually trained
(head/tail/tubing). It is NOT used by measure_from_masks.py, video_ends.py,
or run_video_frames.py - do not run this file directly with --model/--video;
use run_video_frames.py (or video_ends.py) instead, which use your real
segmentation model correctly. Kept here only because size_groups()/EndTrack
are exercised by test_measure.py's own unit tests, and as a reference if a
future project trains a pose model instead of a segmentation one.

  python tube_ends.py --model pose_best.pt --video clip.mp4 --output out.mp4
      (only meaningful with a YOLO-POSE checkpoint - see the warning above)
"""
import argparse
import json
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["head", "tail", "tubing"]     # legacy pose-pipeline only, see docstring
ROLE = {0: "HEAD", 1: "TAIL"}                 # class 2 (tubing) carries no role
ROLE_COLOR = {"HEAD": (255, 160, 0), "TAIL": (0, 200, 255)}


# ---------------------------------------------------------------- monochrome
def to_gray3(bgr):
    """Grayscale for measurement + 3-channel grayscale for the model."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), g


def rectify_strip(gray, start, direction, length, half_width):
    """Sample an axis-aligned strip: rows run along the tube (away from the tip),
    columns across it, centre column = tube axis."""
    u = direction / (np.linalg.norm(direction) + 1e-9)
    n = np.array([-u[1], u[0]])
    rows = np.arange(length, dtype=np.float32)
    cols = np.arange(-half_width, half_width + 1, dtype=np.float32)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    mx = (start[0] + R * u[0] + C * n[0]).astype(np.float32)
    my = (start[1] + R * u[1] + C * n[1]).astype(np.float32)
    return cv2.remap(gray, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def _wall_profile(gray, start, direction, length, half):
    """Median across-tube gradient of a strip starting at `start`, running along `direction`."""
    strip = cv2.GaussianBlur(rectify_strip(gray, start, direction, length, half), (0, 0), 1.0)
    gx = np.abs(cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3))
    return np.median(gx, axis=0)                       # len = 2*half + 1


def _edge_peak(profile, a, b, outer):
    """Outermost local gradient maximum in profile[a:b], refined to sub-pixel."""
    a, b = max(a, 1), min(b, len(profile) - 1)
    idx = [i for i in range(a, b)
           if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]]
    if not idx:
        idx = [a + int(np.argmax(profile[a:b]))]
    top = max(profile[i] for i in idx)
    idx = [i for i in idx if profile[i] >= 0.6 * top]
    i = min(idx) if outer == "left" else max(idx)
    if 0 < i < len(profile) - 1:                         # parabolic sub-pixel refinement
        y0, y1, y2 = profile[i - 1], profile[i], profile[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-6:
            return i + 0.5 * (y0 - y2) / den
    return float(i)


def _walls(profile, c, min_half, max_half, rel_score):
    """Outermost strong symmetric edge pair around column c.
    Returns (lo, hi, peak_score, quality) or None."""
    n = len(profile)
    pmax = np.maximum(profile, np.maximum(np.roll(profile, 1), np.roll(profile, -1)))  # +-1 px tolerance
    kmax = min(max_half, c - 1, n - 2 - c)
    if kmax <= min_half:
        return None
    r = np.arange(min_half, kmax + 1)
    sym = pmax[c - r] * pmax[c + r]
    if sym.max() <= 0:
        return None
    strong = np.where(sym >= rel_score * sym.max())[0]
    k = int(r[strong.max()])                           # outermost strong pair = outer wall
    lo0, hi0 = max(c - k - 2, 0), c + k - 2
    lo = _edge_peak(profile, lo0, c - k + 3, outer="left")    # outer edge of the left wall
    hi = _edge_peak(profile, hi0, min(c + k + 3, n), outer="right")
    if hi - lo < 2 * min_half:
        return None
    quality = float(np.sqrt(sym[strong.max()]) / (np.median(profile) + 1e-6))
    return lo, hi, float(sym[strong.max()]), quality


def measure_width(gray, tip, back, length=None, search_px=None,
                  min_half=4, rel_score=0.5, min_contrast=4.0,
                  refine=True, max_angle_deg=15.0, angle_step_deg=1.5, max_shift=None):
    """Outer diameter of the bare tubing behind kp1, in pixels.

    Transparent tubing has no reliable fill colour, but its walls refract light
    and show up as two parallel edges. Taking the MEDIAN of the across-tube
    gradient along a straightened strip keeps the walls (aligned) and suppresses
    the countertop speckle (not aligned).

    The keypoints only give a ROUGH position: a model is typically a few px /
    several degrees off, which is enough to ruin a naive measurement. With
    refine=True the tube direction (+-max_angle_deg) and the axis centre
    (+-max_shift px) are searched, keeping the combination whose walls are
    sharpest and most symmetric, i.e. the true tube axis.
    Returns (width_px, quality) or (None, 0).
    """
    tip = np.asarray(tip, np.float32)
    back = np.asarray(back, np.float32)
    d = back - tip
    seg = float(np.linalg.norm(d))
    if seg < 4:
        return None, 0.0
    length = int(length or np.clip(1.5 * seg, 15, 200))
    half = int(search_px or np.clip(1.2 * seg, 15, 150))
    shift = int(max_shift if max_shift is not None else max(4, 0.4 * half)) if refine else 0
    angles = (np.deg2rad(np.arange(-max_angle_deg, max_angle_deg + 1e-6, angle_step_deg))
              if refine else [0.0])

    best = None
    for a in angles:
        ca, sa = np.cos(a), np.sin(a)
        da = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]], np.float32)
        prof = _wall_profile(gray, back, da, length, half + shift)
        if prof.max() < min_contrast:
            continue
        c0 = half + shift
        for s in range(-shift, shift + 1):
            w = _walls(prof, c0 + s, min_half, half, rel_score)
            if w is None:
                continue
            score = w[2] * (1.0 - 0.5 * abs(s) / (shift + 1))   # prefer the model's axis when tied
            if best is None or score > best[0]:
                best = (score, w)
    if best is None:
        return None, 0.0
    lo, hi, _, quality = best[1]
    return float(hi - lo), quality


def transparency_score(gray, tip, back, width_px):
    """~1.0 -> background texture shows through (transparent)
       <<1  -> end is opaque (connector / cap).
    Compares texture inside the end with texture beside it, in grayscale."""
    tip = np.asarray(tip, np.float32)
    back = np.asarray(back, np.float32)
    d = back - tip
    seg = float(np.linalg.norm(d))
    if seg < 4 or not width_px:
        return None
    half = int(max(2.2 * width_px, 12))
    strip = rectify_strip(gray, tip, d, int(max(seg, 8)), half)
    lap = np.abs(cv2.Laplacian(strip, cv2.CV_32F, ksize=3))
    n = lap.shape[0]
    lap = lap[int(0.25 * n): max(int(0.75 * n), int(0.25 * n) + 1)]   # skip the tip / connector-end edges
    c = half
    inner = lap[:, c - max(int(0.25 * width_px), 1): c + max(int(0.25 * width_px), 1) + 1]
    outer = np.concatenate([lap[:, : max(c - int(1.6 * width_px), 1)],
                            lap[:, min(c + int(1.6 * width_px), 2 * half): ]], axis=1)
    if outer.size == 0:
        return None
    return float(inner.mean() / (outer.mean() + 1e-6))


def colour_score(bgr, box):
    """Mean saturation and dominant hue of the end crop (colour is only reported,
    the decision itself is made on grayscale)."""
    x1, y1, x2, y2 = map(int, box)
    crop = bgr[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv = hsv[hsv[:, 2] > 40]
    if len(hsv) == 0:
        return None
    sat = float(hsv[:, 1].mean())
    hue = int(np.median(hsv[hsv[:, 1] > 60][:, 0])) if (hsv[:, 1] > 60).any() else None
    return {"saturation": round(sat, 1), "hue": hue, "coloured": sat > 60}


# ---------------------------------------------------------------- tracking
class EndTrack:
    """Accumulates evidence for one tracked end and locks it after N frames."""

    def __init__(self, tid, lock_after, smooth=0.5):
        self.id = tid
        self.lock_after = lock_after
        self.smooth = smooth
        self.cls_votes = Counter()
        self.widths = deque(maxlen=120)
        self.trans = deque(maxlen=120)
        self.colour = deque(maxlen=120)
        self.tip = None
        self.back = None
        self.frames = 0
        self.last_frame = 0
        self.path = []

    def update(self, frame_idx, cls_id, conf, tip, back, width, quality, trans, colour):
        a = self.smooth
        tip, back = np.asarray(tip, float), np.asarray(back, float)
        self.tip = tip if self.tip is None else a * tip + (1 - a) * self.tip
        self.back = back if self.back is None else a * back + (1 - a) * self.back
        self.cls_votes[cls_id] += conf
        if width is not None and quality > 1.5:
            self.widths.append(width)
        if trans is not None:
            self.trans.append(trans)
        if colour is not None:
            self.colour.append(colour["saturation"])
        self.frames += 1
        self.last_frame = frame_idx
        self.path.append([frame_idx, round(self.tip[0], 1), round(self.tip[1], 1)])

    @property
    def cls(self):
        return self.cls_votes.most_common(1)[0][0] if self.cls_votes else None

    @property
    def locked(self):
        return self.frames >= self.lock_after and len(self.widths) >= self.lock_after // 2

    @property
    def width(self):
        return float(np.median(self.widths)) if self.widths else None

    def summary(self, mm_per_px=None):
        w = self.width
        return {
            "track_id": self.id,
            "role": ROLE.get(self.cls),
            "class": CLASS_NAMES[self.cls] if self.cls is not None else None,
            "locked": self.locked,
            "frames": self.frames,
            "width_px": round(w, 2) if w else None,
            "width_mm": round(w * mm_per_px, 2) if (w and mm_per_px) else None,
            "width_spread_px": round(float(np.std(self.widths)), 2) if self.widths else None,
            "transparency": round(float(np.median(self.trans)), 2) if self.trans else None,
            "saturation": round(float(np.median(self.colour)), 1) if self.colour else None,
        }


def size_groups(tracks, min_ratio=1.15):
    """Split locked widths at their largest gap. Returns {track_id: 'BIG'/'SMALL'/'SAME'}."""
    items = sorted((t.width, t.id) for t in tracks if t.locked and t.width)
    if len(items) < 2:
        return {tid: "SAME" for _, tid in items}
    ws = [w for w, _ in items]
    ratios = [ws[i + 1] / ws[i] for i in range(len(ws) - 1)]
    i = int(np.argmax(ratios))
    if ratios[i] < min_ratio:
        return {tid: "SAME" for _, tid in items}
    cut = (ws[i] + ws[i + 1]) / 2
    return {tid: ("BIG" if w > cut else "SMALL") for w, tid in items}


# ---------------------------------------------------------------- per-frame
def process(model, frame, frame_idx, tracks, args, use_tracker=True):
    gray3, gray = to_gray3(frame)
    kw = dict(conf=args.conf, verbose=False, imgsz=args.imgsz)
    res = (model.track(gray3, persist=True, tracker=args.tracker, **kw)
           if use_tracker else model(gray3, **kw))[0]
    dets = []
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return dets
    kxy = res.keypoints.xy.cpu().numpy()
    ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
    for j, box in enumerate(res.boxes):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        tip, back = kxy[j][0], kxy[j][1]
        if not (tip.any() and back.any()):
            continue
        width, quality = measure_width(gray, tip, back, search_px=args.search_px)
        trans = transparency_score(gray, tip, back, width)
        colour = colour_score(frame, box.xyxy[0].tolist())
        tid = ids[j] if ids[j] is not None else -(j + 1)
        if tid not in tracks:
            tracks[tid] = EndTrack(tid, args.lock_after)
        tracks[tid].update(frame_idx, cls_id, conf, tip, back, width, quality, trans, colour)
        dets.append(tid)
    return dets


def draw(frame, tracks, visible, groups, mm_per_px):
    for tid in visible:
        t = tracks[tid]
        role = ROLE.get(t.cls, "?")
        col = ROLE_COLOR.get(role, (200, 200, 200))
        tip = tuple(int(v) for v in t.tip)
        back = tuple(int(v) for v in t.back)
        cv2.line(frame, tip, back, col, 2, cv2.LINE_AA)
        cv2.circle(frame, tip, 7, (0, 0, 0), -1)
        cv2.circle(frame, tip, 5, col, -1)
        w = t.width
        wtxt = "w=?" if w is None else (f"w={w * mm_per_px:.2f}mm" if mm_per_px else f"w={w:.1f}px")
        size = groups.get(tid, "") if t.locked else "..."
        text = f"#{tid} {role} {CLASS_NAMES[t.cls]} {wtxt} {size}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x, y = tip[0] + 10, tip[1] - 10
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- runners
def run_video(model, args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tracks, writer, frame_idx = {}, None, 0
    preview = [not args.no_preview]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if writer is None and args.output:   # size from the real frame (phone rotation!)
                h, w = frame.shape[:2]
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"cannot write {args.output} (must be a file path, e.g. out.mp4)")
            visible = process(model, frame, frame_idx, tracks, args)
            groups = size_groups(tracks.values(), args.min_ratio)
            draw(frame, tracks, visible, groups, args.mm_per_px)
            if writer is not None:
                writer.write(frame)
            if preview[0]:
                try:
                    scale = 900 / frame.shape[0]
                    cv2.imshow("tube ends", cv2.resize(frame, None, fx=scale, fy=scale))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:      # headless OpenCV build: carry on without a window
                    preview[0] = False
                    print("[info] no GUI support in this OpenCV build - preview disabled")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if preview[0]:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    groups = size_groups(tracks.values(), args.min_ratio)
    min_frames = max(3, args.lock_after // 2)
    report = {"video": args.video, "frames": frame_idx, "ends": []}
    for t in sorted(tracks.values(), key=lambda t: t.id):
        if t.frames < min_frames:           # drop flicker tracks
            continue
        s = t.summary(args.mm_per_px)
        s["size"] = groups.get(t.id)
        if args.keep_paths:
            s["path"] = t.path
        report["ends"].append(s)
    return report


def run_images(model, args):
    out_dir = Path(args.output or "tube_ends_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    report = {"images": []}
    for p in paths:
        frame = cv2.imread(str(p))
        tracks = {}
        args.lock_after = 1                  # single image: nothing to accumulate
        visible = process(model, frame, 1, tracks, args, use_tracker=False)
        groups = size_groups(tracks.values(), args.min_ratio)
        draw(frame, tracks, visible, groups, args.mm_per_px)
        cv2.imwrite(str(out_dir / p.name), frame)
        ends = []
        for t in tracks.values():
            s = t.summary(args.mm_per_px)
            s["size"] = groups.get(t.id)
            ends.append(s)
        report["images"].append({"image": p.name, "ends": ends})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images", help="folder of stills")
    ap.add_argument("--output", help="video file (video mode) or folder (image mode)")
    ap.add_argument("--report", help="JSON report path")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--lock-after", type=int, default=15, help="frames before an end is locked")
    ap.add_argument("--search-px", type=int, default=None, help="max tube half-width to search, px")
    ap.add_argument("--min-ratio", type=float, default=1.15, help="width ratio to call BIG vs SMALL")
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--keep-paths", action="store_true", help="store tip trajectories in the report")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--i-know-this-needs-a-pose-model", action="store_true",
                    help="confirms --model is a YOLO-POSE checkpoint (tip+axis keypoints), "
                        "not the YOLO11-seg model your real pipeline uses (run_video_frames.py)")
    args = ap.parse_args()
    if not args.i_know_this_needs_a_pose_model:
        raise SystemExit(
            "tube_ends.py's own CLI is a LEGACY pose-model pipeline, not your working "
            "YOLO11-seg pipeline. For your head/tail/tubing model use run_video_frames.py "
            "(or video_ends.py / measure_from_masks.py) instead - those read masks and "
            "match classes by name from your checkpoint. If you really do have a separate "
            "YOLO-pose model and mean to run this, add --i-know-this-needs-a-pose-model.")

    from ultralytics import YOLO   # imported here so the measurement code is testable without it
    model = YOLO(args.model)
    report = run_video(model, args) if args.video else run_images(model, args)
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text)
    print(text if len(text) < 4000 else f"[done] report -> {args.report}")


if __name__ == "__main__":
    main()