r"""
video_to_dataset.py - take 1 frame per second from a video, run your weights on it
(run_video_frames.py + config.yaml) and save, for every sampled frame:

    images\     the clean frame (for training)
    labels\     a YOLO label .txt made from the model's predictions (same name as the image)
    annotated\  the frame with the analyzer's boxes / cards drawn on it (to look at only)
    results\    the analyzer's JSON result

Frames with exactly 2 heads + 2 tails go to  <out>\ok\...
every other frame goes to                    <out>\redo\...   (fix these labels)
report.csv in <out> lists every sampled frame.

    python video_to_dataset.py --config config.yaml --video "C:\videos\part_0.avi"
    python video_to_dataset.py --config config.yaml --video "C:\videos"            (every video in the folder)
    python video_to_dataset.py --config config.yaml --video part_0.avi --every 0.5  (2 frames per second)
    python video_to_dataset.py --config config.yaml --video part_0.avi --only-redo  (save only the redo frames)

File names: <video name>_frame<frame number>.jpg / .txt, e.g. part_0_frame00030.jpg
Labels:     segmentation model -> polygons  "class x1 y1 x2 y2 ..."  (normalised 0-1)
            detection model    -> boxes     "class cx cy w h"        (normalised 0-1)
            class numbers = the model's own class ids (same as your dataset)
            every head / tail in the WHOLE frame is labelled (not only inside the ROI);
            the ok / redo decision uses the ROI, like run_video_frames does.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from head_tail_analyzer.run_video_frames import (
        AnalyzerError, StreamState, analyze_frame, configure_for_model,
        load_model, load_stream_config)
    import head_tail_analyzer.run_video_frames as rvf
except ImportError:          # script placed next to run_video_frames.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_video_frames import (
        AnalyzerError, StreamState, analyze_frame, configure_for_model,
        load_model, load_stream_config)
    import run_video_frames as rvf

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".h264", ".h265", ".mjpeg", ".m4v", ".wmv"}
DEFAULT_OUT = r"C:\Users\EmageVision\Documents\video_frames"

# analyze_frame only returns the summary; keep the raw detections (with mask outlines)
# of the last frame so the labels can be written from them.
_last = {}
_orig_detect = rvf.detect


def _detect_and_keep(*args, **kwargs):
    dets = _orig_detect(*args, **kwargs)
    _last["dets"] = dets
    return dets


rvf.detect = _detect_and_keep


# ------------------------------------------------------------------ helpers
def write_image(path, img, ext):
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(str(path))


def list_videos(path):
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        vids = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        if not vids:
            raise SystemExit(f"no videos ({', '.join(sorted(VIDEO_EXT))}) in {p}")
        return vids
    if not p.exists():
        raise SystemExit(f"video not found: {p}")
    return [p]


def yolo_lines(dets, class_of, w, h, seg):
    """Detections -> YOLO label lines (normalised)."""
    lines = []
    for d in dets:
        cid = class_of[d["role"]]
        x1, y1, x2, y2 = d["xyxy"]
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if seg:
            c = d.get("outline")
            if c is not None and len(c) >= 3:
                c = cv2.approxPolyDP(c.astype(np.float32), 1.0, True).reshape(-1, 2)
            if c is None or len(c) < 3:          # no mask -> box as a 4-point polygon
                c = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            c = c.reshape(-1, 2).astype(np.float64)
            c[:, 0] = np.clip(c[:, 0] / w, 0, 1)
            c[:, 1] = np.clip(c[:, 1] / h, 0, 1)
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in c.flatten()))
        else:
            lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


# ------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="1 frame per second from a video -> images + YOLO labels + annotated frames.")
    ap.add_argument("--config", help="config.yaml used by run_video_frames / main.py")
    ap.add_argument("--model", help="model .pt (overrides config)")
    ap.add_argument("--video", required=True, help="a video file, or a folder of videos")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output folder (default {DEFAULT_OUT})")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between saved frames (default 1 = 1 frame/s)")
    ap.add_argument("--heads", type=int, default=2, help="expected heads per frame (default 2)")
    ap.add_argument("--tails", type=int, default=2, help="expected tails per frame (default 2)")
    ap.add_argument("--only-redo", action="store_true", help="save only the frames that are not 2 heads + 2 tails")
    ap.add_argument("--no-roi", action="store_true", help="ignore the config's ROI, count the whole frame")
    ap.add_argument("--ext", default=".jpg", choices=[".jpg", ".png"], help="image format (default .jpg)")
    ap.add_argument("--device", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--conf", type=float, help="confidence threshold (overrides config)")
    a = ap.parse_args(argv)

    videos = list_videos(a.video)

    # each sampled frame is 1 s away from the last one -> tracking cannot follow the
    # ends, so every frame is analysed on its own and an end counts on the first sighting
    overrides = dict(model_path=a.model, device=a.device, conf=a.conf, min_frames=1, min_frames_tail=1)
    try:
        cfg, roi = load_stream_config(a.config, use_roi=not a.no_roi, **overrides)
    except AnalyzerError as e:
        raise SystemExit(f"[config] {e}")
    cfg.print_frames = False
    cfg.latest_json = None
    cfg.history_jsonl = None

    entry = load_model(cfg.model_path, device=cfg.device, imgsz=int(cfg.imgsz or cfg.tile or 640))
    masks_ok = configure_for_model(cfg, entry)
    seg = entry.task == "segment"
    class_of = {role: cid for cid, role in entry.role_ids.items()}
    if entry.tubing_id is not None:
        print(f"[warn] the model also has a tubing class ({entry.tubing_id}); "
              f"the labels contain only HEAD / TAIL - add tubing yourself if your dataset uses it")

    out = Path(a.out)
    groups = ("ok", "redo")
    dirs = {(g, k): out / g / k for g in groups for k in ("images", "labels", "annotated", "results")}
    if a.only_redo:
        dirs = {gk: d for gk, d in dirs.items() if gk[0] == "redo"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"[run] {len(videos)} video(s), 1 frame every {a.every:g} s, "
          f"expected {a.heads} heads + {a.tails} tails, ROI={'on' if roi is not None else 'off'}, "
          f"labels={'polygons' if seg else 'boxes'} {class_of}")
    print(f"[run] output -> {out}\n")

    rows, n_ok, n_redo = [], 0, 0
    state = StreamState()
    for v in videos:
        try:
            cap, fps, total = rvf.open_video(str(v))
        except AnalyzerError as e:
            print(f"[skip] {e}")
            continue
        step = max(int(round(fps * a.every)), 1)
        digits = max(5, len(str(total or 0)))
        n_samples = (total + step - 1) // step if total else "?"
        print(f"[video] {v.name}: {fps:.1f} fps, {total or '?'} frames -> every {step}th frame ({n_samples} frames)")

        fidx = -1
        while True:
            fidx += 1
            if fidx % step:                      # skipped frame: grab only (fast, no decode)
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break

            state.reset()
            _last.clear()
            result, vis = analyze_frame(entry, frame, state, roi, cfg,
                                        frame_index=fidx, annotate=True, masks=masks_ok)
            heads, tails = result["heads_count"], result["tails_count"]
            redo = heads != a.heads or tails != a.tails
            group = "redo" if redo else "ok"
            n_redo += redo
            n_ok += not redo

            name = f"{v.stem}_frame{fidx:0{digits}d}"
            reasons = []
            if heads != a.heads:
                reasons.append(f"heads {heads}/{a.heads}")
            if tails != a.tails:
                reasons.append(f"tails {tails}/{a.tails}")
            saved = not (a.only_redo and not redo)

            if saved:
                h, w = frame.shape[:2]
                dets = _last.get("dets", [])
                write_image(dirs[(group, "images")] / (name + a.ext), frame, a.ext)
                (dirs[(group, "labels")] / (name + ".txt")).write_text(
                    "\n".join(yolo_lines(dets, class_of, w, h, seg)) + ("\n" if dets else ""))
                write_image(dirs[(group, "annotated")] / (name + ".jpg"), vis, ".jpg")
                (dirs[(group, "results")] / (name + ".json")).write_text(json.dumps(result, indent=2))

            rows.append({"video": v.name, "frame": fidx, "time_s": round(fidx / fps, 2),
                         "image": name + a.ext if saved else "", "status": group.upper(),
                         "heads_found": heads, "tails_found": tails,
                         "reason": ", ".join(reasons),
                         "bigger_tube": result["bigger_tube"]["status"]})
            print(f"  {group.upper():4}  {fidx / fps:7.1f}s  found {heads}H {tails}T  {name}")
        cap.release()

    out.mkdir(parents=True, exist_ok=True)
    report = out / "report.csv"
    fields = ["video", "frame", "time_s", "image", "status", "heads_found", "tails_found",
              "reason", "bigger_tube"]
    with open(report, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[done] {n_ok + n_redo} frames: {n_ok} OK, {n_redo} REDO")
    print(f"  output : {out}")
    print(f"  report : {report}")


if __name__ == "__main__":
    main()
