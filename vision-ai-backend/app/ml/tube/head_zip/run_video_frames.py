"""
run_video_frames.py - print HEADS, TAILS, widths, bounding boxes and the
bigger tube for EVERY FRAME of a video, live in the terminal - AND write the
same data as JSON so a frontend can pick it up. Nothing is passed on the
command line - edit CONFIG below and run:

    python run_video_frames.py

Unlike video_ends.py (which tracks ends across the whole video and only
prints one final summary), this analyses each frame ON ITS OWN, the exact
same way measure_from_masks.py analyses a single photo - same tested
functions (masks_from_model -> analyse), just looped over video frames.

Per frame you get, in the terminal AND in JSON:
  - how many heads, how many tails
  - every end's role, class, BOUNDING BOX [x, y, width, height] in pixels
    (top-left corner + size - draw it directly as a rectangle in a frontend),
    width, and BIGGER/SMALLER/SAME
  - which tube is bigger this frame (paired head+tail where possible, by
    matching width - see pair_ends() in measure_from_masks.py for why this
    is used instead of trying to trace pixels through the coil)

For the frontend: LATEST_JSON is OVERWRITTEN every frame with just that
frame's result - point a frontend at this single file and poll it (e.g. every
200 ms) to show "what the camera sees right now". HISTORY_JSONL, if set, gets
one JSON line appended per frame instead, for a full per-frame record you can
replay or chart afterwards - leave it as None if you don't need that.

Your model was trained on 640px TILES (prep_tiles.py), so inference tiles
the same way (TILE=640 below) - running full frames through it without
tiling under-sizes the tubes again, the same failure tiling was built to fix.
"""
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from measure_from_masks import analyse, build_class_map, masks_from_model, draw

# ============================== CONFIG — edit these ==============================
MODEL_PATH = r"C:\Users\EmageVision\Documents\head_tail_classification\runs\segment\train-2\weights\best.pt"
VIDEO_PATH = R"C:\Users\EmageVision\Documents\head_tail_classification\videos\testing_video 1.avi"

DEVICE = "0"          # "0" = first GPU, "cpu" = CPU
CONF = 0.25
TILE = 640            # must match training (0 = whole frame, NOT recommended for this model)
TILE_OVERLAP = 0.3
STRIDE = 1             # 1 = every frame; raise to skip frames if it runs too slowly
MM_PER_PX = None       # e.g. 0.08 once calibrated with a ruler / ArUco marker

OUTPUT_VIDEO = r"C:\Users\EmageVision\Documents\head_tail_classification\measured\testing_1_frames.mp4"
LATEST_JSON = r"C:\Users\EmageVision\Documents\head_tail_classification\measured\latest_frame.json"
HISTORY_JSONL = None   # e.g. r"...\measured\history.jsonl" to also keep every frame's result
SHOW_LIVE_PREVIEW = False
# ===================================================================================


def to_json_end(i, e):
    """Only JSON-safe, frontend-relevant fields - drops the internal '_axis'
    numpy data that analyse() uses purely for drawing the overlay."""
    return {
        "id": i,
        "role": e["role"],
        "class": e["class"],
        "status": e["status"],
        "bbox": e.get("bbox"),                 # [x, y, width, height] px, top-left origin
        "width_px": e.get("width_px"),
        "width_mm": e.get("width_mm"),
        "size": e.get("size"),                 # "BIGGER" / "SMALLER" / "SAME" / None
        "pair_id": e.get("pair_id"),           # ends sharing a pair_id = the same tube
        "opaque": e.get("opaque"),
        "colour": e.get("colour"),
    }


def build_frame_result(frame_idx, ends, last_known):
    """last_known is a small dict that PERSISTS across calls (create ONE per
    video run, e.g. {"bigger_tube": None, "smaller_tube": None}, and pass the
    SAME dict in every frame - do not recreate it inside the loop). If this
    frame has no BIGGER/SMALLER comparison (a momentary gap, or genuinely
    "SAME"), the last frame that DID have a real answer keeps being shown
    instead of the result going blank/"SAME" every time the sizes happen to
    land too close together for one frame."""
    detections = [to_json_end(i, e) for i, e in enumerate(ends)]
    # HEADS/TAILS should always match (each tube has exactly one of each) -
    # heads detect more reliably here than tails do, so if the two disagree
    # this frame, tails_count is forced to match heads_count. The raw
    # (uncorrected) tail count is still kept in the result as
    # "tails_count_raw" so you can see how often this correction fires.
    heads_count = sum(d["role"] == "HEAD" for d in detections)
    tails_count_raw = sum(d["role"] == "TAIL" for d in detections)
    tails_count = heads_count if heads_count != tails_count_raw else tails_count_raw
    bigger = next((d for d in detections if d["size"] == "BIGGER"), None)
    smaller = next((d for d in detections if d["size"] == "SMALLER"), None)
    used_last_known = False
    if bigger is not None:
        last_known["bigger_tube"] = bigger
        last_known["smaller_tube"] = smaller
    elif last_known["bigger_tube"] is not None:
        bigger = last_known["bigger_tube"]
        smaller = last_known["smaller_tube"]
        used_last_known = True
    return {
        "frame": frame_idx,
        "heads_count": heads_count,
        "tails_count": tails_count,
        "tails_count_raw": tails_count_raw,
        "detections": detections,
        "bigger_tube": bigger,
        "smaller_tube": smaller,
        "used_last_known_size": used_last_known,
    }


def frame_summary(result):
    print(f"\nFrame {result['frame']}: HEADS={result['heads_count']}  "
          f"TAILS={result['tails_count']}")
    for d in result["detections"]:
        w = f"{d['width_mm']:.2f}mm" if d["width_mm"] else (
            f"{d['width_px']:.1f}px" if d["width_px"] else "?")
        pair = f" tube#{d['pair_id']}" if d["pair_id"] else ""
        flag = "" if d["status"] == "OK" else f"  [{d['status']}]"
        print(f"  #{d['id']} {d['role']:4s} {d['class']:10s} bbox={d['bbox']} "
              f"width={w:>9s} {d['size'] or '':7s}{pair}{flag}")
    if result["bigger_tube"]:
        line = f"  -> BIGGER tube: {result['bigger_tube']['width_px']}px"
        if result["smaller_tube"]:
            line += f"   SMALLER tube: {result['smaller_tube']['width_px']}px"
        if result["used_last_known_size"]:
            line += "   (last known good comparison - none new this frame)"
        print(line)
    elif len(result["detections"]) >= 2:
        print("  -> tube sizes look the SAME so far")
    else:
        print("  -> not enough ends this frame to compare sizes")


def main():
    model = YOLO(MODEL_PATH)
    if DEVICE:
        model.to(f"cuda:{DEVICE}" if DEVICE.isdigit() else DEVICE)
    print(f"[device] model is running on: {model.device}")
    class_map = build_class_map(model.names)
    print(f"[classes] {model.names} -> tubing id {class_map[2]}, roles {class_map[1]}")

    if LATEST_JSON:
        Path(LATEST_JSON).parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_JSONL:
        Path(HISTORY_JSONL).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    preview = SHOW_LIVE_PREVIEW
    last_known = {"bigger_tube": None, "smaller_tube": None}
    idx, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % STRIDE:
                continue

            inst = masks_from_model(model, frame, CONF, True, TILE, TILE_OVERLAP)
            ends = analyse(frame, inst, MM_PER_PX, gap=3, class_map=class_map)
            result = build_frame_result(idx, ends, last_known)
            frame_summary(result)

            if LATEST_JSON:
                Path(LATEST_JSON).write_text(json.dumps(result, indent=2))
            if HISTORY_JSONL:
                with open(HISTORY_JSONL, "a") as f:
                    f.write(json.dumps(result) + "\n")

            if OUTPUT_VIDEO or preview:
                vis = draw(frame, inst, ends, tubing_id=class_map[2])
                if OUTPUT_VIDEO and writer is None:
                    h, w = vis.shape[:2]
                    Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / STRIDE, (w, h))
                    if not writer.isOpened():
                        raise SystemExit(f"cannot write {OUTPUT_VIDEO} - check the path "
                                         "is a file (e.g. ends in .mp4), not a folder")
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
    print("\n[done]")


if __name__ == "__main__":
    main()
