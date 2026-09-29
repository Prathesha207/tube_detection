"""
run_video_frames.py - count HEADS and TAILS in every frame of a video, print
them live in the terminal, write them as JSON for a frontend, and save an
annotated video.

    hta-frames --config config.yaml --video part_0.avi
    hta-frames --config config.yaml --video part_0.avi --output out.mp4 --json latest.json
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
  * Heads and tails come STRAIGHT from the model's boxes; the TUBING class is
    ignored (not even requested from the model).
  * Tiled inference (tile px, tile_overlap) - the model was trained on tiles.
    Duplicates from overlapping tiles are merged.
  * Tracking: an end must be seen min_frames times before it counts, which
    filters one-off false detections.
  * ROI (roi_json): inference and tracking run on the WHOLE frame, but only
    detections whose center lies inside the polygon are drawn, written to
    JSON and counted.

Per frame (terminal + JSON), all restricted to the ROI:
  heads_count / tails_count      confirmed ends visible in this frame
  heads_visible / tails_visible  same as *_count (kept for the frontend)
  heads_total / tails_total      distinct confirmed ends that ever appeared in the ROI
  detections                     id, role, confidence, bbox [x, y, w, h], center
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

DEFAULTS = {
    "model_path": None,       # trained YOLO11-seg weights (required)
    "video": None,            # video file, or "0" for webcam 0 (required)
    "device": "0",            # "0" = first GPU, "cpu" = CPU
    "conf": 0.35,
    "iou": 0.5,               # NMS inside each tile
    "tile": 640,              # must match training (0 = whole frame)
    "tile_overlap": 0.3,
    "merge_iou": 0.4,         # merge duplicate boxes from overlapping tiles...
    "merge_contain": 0.75,    # ...or when one box is mostly inside another
    "gray": True,             # feed the model grayscale (it was trained on grayscale)
    "stride": 1,              # 1 = every frame
    "min_frames": 5,          # sightings before an end is confirmed and counted
    "max_dist": 60,           # px an end may move between frames and stay the same end
    "max_missed": 15,         # frames a lost track can still be re-matched
    "roi_json": None,         # polygon file; None = whole frame
    "draw_roi": True,
    "output_video": None,     # annotated .mp4; None = don't write
    "latest_json": None,      # overwritten every frame; None = don't write
    "history_jsonl": None,    # one line per frame; None = don't write
    "print_frames": True,
    "show_preview": False,
}
PATH_KEYS = ("model_path", "video", "roi_json", "output_video", "latest_json", "history_jsonl")

ROLES = ("HEAD", "TAIL")
COLORS = {"HEAD": (255, 40, 20), "TAIL": (230, 220, 0)}   # BGR: blue, cyan


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

    cfg["device"] = str(cfg["device"])
    cfg["tile"] = int(cfg["tile"] or 0)
    cfg["stride"] = max(int(cfg["stride"]), 1)
    return SimpleNamespace(**cfg)


# ------------------------------------------------------------------ detection
def role_ids_from_names(names):
    """{class_id: 'HEAD'/'TAIL'} - tubing and anything else is ignored."""
    ids = {int(i): n.upper() for i, n in names.items() if n.upper() in ROLES}
    missing = set(ROLES) - set(ids.values())
    if missing:
        raise SystemExit(f"model classes {names} have no {missing} class")
    return ids


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


def merge_duplicates(dets, cfg):
    """The same end seen in two overlapping tiles -> keep the most confident box."""
    kept = []
    for d in sorted(dets, key=lambda d: d["conf"], reverse=True):
        dup = False
        for k in kept:
            if k["role"] != d["role"]:
                continue
            iou, contain = box_overlap(k["xyxy"], d["xyxy"])
            if iou >= cfg.merge_iou or contain >= cfg.merge_contain:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def detect(model, frame, role_ids, cfg):
    if cfg.gray:
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
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

    results = model.predict(crops, conf=cfg.conf, iou=cfg.iou, classes=list(role_ids),
                            imgsz=t or 640, device=cfg.device, verbose=False)
    dets = []
    for r, (ox, oy) in zip(results, offsets):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, conf):
            if c not in role_ids:
                continue
            box = [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]
            dets.append({
                "role": role_ids[c],
                "conf": float(s),
                "xyxy": box,
                "center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            })
    return merge_duplicates(dets, cfg)


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
        self.update(det, idx)

    def update(self, det, idx):
        self.center = det["center"]
        self.xyxy = det["xyxy"]
        self.conf = det["conf"]
        self.last_seen = idx
        self.hits += 1

    @property
    def confirmed(self):
        return self.hits >= self.min_frames


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
                if dist <= cfg.max_dist:
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
                t = Track(next_id, d, idx, cfg.min_frames)
                next_id += 1
                tracks.append(t)
                d["track"] = t
    return next_id


# -------------------------------------------------------------------- output
def load_roi(path):
    if not path:
        return None
    if not Path(path).exists():
        raise SystemExit(f"roi_json is set but not found: {path}  (use --no-roi to run without)")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(data["points"], dtype=np.int32)


def in_roi(det, roi):
    if roi is None:
        return True
    cx, cy = det["center"]
    return cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0


def build_frame_result(idx, dets, tracks):
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
        detections.append({
            "id": t.id,
            "role": t.role,
            "conf": round(d["conf"], 3),
            "bbox": [int(round(x1)), int(round(y1)),
                     int(round(x2 - x1)), int(round(y2 - y1))],
            "center": [int(round(d["center"][0])), int(round(d["center"][1]))],
        })
    return {"frame": idx, **counts, "detections": detections}


def frame_summary(result):
    print(f"\nFrame {result['frame']}: HEADS={result['heads_count']}  "
          f"TAILS={result['tails_count']}  "
          f"(total seen: {result['heads_total']} heads, {result['tails_total']} tails)")
    for d in result["detections"]:
        print(f"  #{d['id']:<3} {d['role']:4s} conf={d['conf']:.2f} bbox={d['bbox']}")


def draw(frame, dets, result, roi, cfg):
    vis = frame.copy()
    if roi is not None and cfg.draw_roi:
        cv2.polylines(vis, [roi.reshape(-1, 1, 2)], True, (0, 200, 0), 2, cv2.LINE_AA)
    for d in dets:
        t = d["track"]
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        if t.confirmed:
            color, thick = COLORS[t.role], 2
            label = f"{t.role.lower()} #{t.id} {d['conf']:.2f}"
        else:                                   # not yet confirmed - thin grey
            color, thick = (160, 160, 160), 1
            label = f"{t.role.lower()}?"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2 if t.confirmed else 1, cv2.LINE_AA)
    header = (f"frame {result['frame']}   HEADS {result['heads_count']}   "
              f"TAILS {result['tails_count']}")
    cv2.rectangle(vis, (0, 0), (560, 40), (0, 0, 0), -1)
    cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def print_diagnosis(tracks, min_frames):
    print(f"\n[diagnosis] {len(tracks)} tracks created, "
          f"{sum(t.confirmed for t in tracks)} confirmed (min_frames={min_frames})")
    for role in ROLES:
        rows = sorted((t for t in tracks if t.role == role), key=lambda t: t.first_seen)
        print(f"  {role}: {len(rows)} created, {sum(t.confirmed for t in rows)} confirmed")
        for t in rows:
            tag = "CONFIRMED" if t.confirmed else "discarded (flicker)"
            print(f"    id={t.id:<4} frames {t.first_seen}-{t.last_seen}  seen {t.hits}x  "
                  f"first center={tuple(round(v) for v in t.first_center)}  [{tag}]")
    print("  -> many short confirmed tracks of one role = the same end re-counted "
          "(raise max_dist or max_missed)")


# ------------------------------------------------------------------------ run
def run(cfg):
    """Process the video described by `cfg` (from load_config). Yields one result
    dict per processed frame and writes the video / JSON outputs set in cfg."""
    from ultralytics import YOLO   # imported here so config errors show up fast

    model = YOLO(cfg.model_path)
    role_ids = role_ids_from_names(model.names)
    print(f"[model] {cfg.model_path}")
    print(f"[classes] {model.names} -> using {role_ids} (tubing ignored)")
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

            dets = detect(model, frame, role_ids, cfg)
            next_id = update_tracks(tracks, dets, idx, next_id, cfg)   # ALL detections

            shown = [d for d in dets if in_roi(d, roi)]                # ROI = output only
            for d in shown:
                d["track"].in_roi_ever = True
            result = build_frame_result(idx, shown, tracks)
            if cfg.print_frames:
                frame_summary(result)

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
    print_diagnosis(tracks, cfg.min_frames)


# ----------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Count heads and tails per video frame.",
        epilog="Any flag overrides the same setting in the config file.")
    ap.add_argument("--config", help="config.yaml (paths inside it are relative to its folder)")
    ap.add_argument("--video", help="video file, or 0 for webcam")
    ap.add_argument("--model", help="trained YOLO11-seg weights (.pt)")
    ap.add_argument("--output", help="annotated video, e.g. out.mp4")
    ap.add_argument("--json", help="latest-frame JSON, overwritten every frame")
    ap.add_argument("--history", help="JSONL with one line per frame")
    ap.add_argument("--roi", help="roi.json polygon")
    ap.add_argument("--no-roi", action="store_true", help="ignore any ROI, use the whole frame")
    ap.add_argument("--device", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--stride", type=int, help="process every Nth frame")
    ap.add_argument("--color", action="store_true",
                    help="feed colour frames (only if the model was trained on colour)")
    ap.add_argument("--preview", action="store_true", help="show a live window (q to stop)")
    ap.add_argument("--quiet", action="store_true", help="don't print every frame")
    a = ap.parse_args(argv)

    cfg = load_config(
        a.config,
        model_path=a.model, video=a.video, output_video=a.output,
        latest_json=a.json, history_jsonl=a.history, roi_json=a.roi,
        device=a.device, conf=a.conf, stride=a.stride,
        gray=False if a.color else None,
        show_preview=True if a.preview else None,
        print_frames=False if a.quiet else None,
    )
    if a.no_roi:
        cfg.roi_json = None

    n = sum(1 for _ in run(cfg))
    print(f"\n[done] {n} frames processed")
    for label, p in (("video", cfg.output_video), ("json", cfg.latest_json),
                     ("history", cfg.history_jsonl)):
        if p:
            print(f"  {label}: {p}")


if __name__ == "__main__":
    main()