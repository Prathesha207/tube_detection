"""
run_video_frames.py - print HEADS, TAILS, widths, bounding boxes and the
bigger tube for EVERY FRAME of a video, live in the terminal - AND write the
same data as JSON so a frontend can pick it up. Nothing is passed on the
command line - edit CONFIG below and run:

    python run_video_frames.py

HOW THIS DIFFERS FROM A NAIVE PER-FRAME VERSION (why this design):
Analysing every frame from scratch, with no memory of earlier frames, has two
real problems in practice:
  1. A single frame where the tubing mask is briefly missed (motion blur, a
     hand passing over it, a bad tile seam) makes that end's width vanish for
     that frame - and with fewer than 2 valid widths to compare, you get a
     meaningless "SAME" or "not enough data" on a frame where a real bigger/
     smaller answer was obvious a moment ago.
  2. A stray false-positive that only appears for one or two frames counts
     exactly as much as a real, physically-present tube end - inflating
     HEADS/TAILS counts with noise.
So this script TRACKS each end across frames (the same matching used in
video_ends.py): an end must be seen for --MIN_FRAMES frames before it is
"confirmed" and included in the counts/output at all (filters flicker), and
once confirmed, its reported width is the MEDIAN of every good measurement
seen so far - so a single bad frame does not erase a well-established width;
the last known good value keeps being shown until a better one replaces it.
The bounding box shown each frame is still THIS frame's own detection (so it
tracks the object's real on-screen position), but width/size/pair_id come
from the track's stable history.

Per frame you get, in the terminal AND in JSON:
  - HEADS/TAILS = how many DISTINCT ends have been CONFIRMED so far (settles
    to the true count once your real ends are seen a few times, and does not
    fluctuate on temporary occlusion or a stray misdetection)
  - for each end currently visible in this exact frame: role, class,
    bounding box [x, y, width, height] in pixels (draw it directly as a
    rectangle in a frontend), its STABLE width, and BIGGER/SMALLER/SAME
  - which tube is bigger, from the confirmed tracks' stable widths - this
    keeps showing the last known answer through a momentary bad frame instead
    of resetting to "SAME"

Your model was trained on 640px TILES (prep_tiles.py), so inference tiles
the same way (TILE=640 below) - running full frames through it without
tiling under-sizes the tubes again, the same failure tiling was built to fix.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from measure_from_masks import analyse, build_class_map, masks_from_model, draw, pair_ends
from video_ends import Track, match, confirmed, size_labels

# ============================== CONFIG — edit these ==============================
MODEL_PATH = r"D:\tube_tracing_new\vision-monitor\vision-ai-backend\app\ml\tube\model\best.pt"
VIDEO_PATH = r"C:\Users\EmageVision\Downloads\testing_video_1.avi"
DEVICE = "0"          # "0" = first GPU, "cpu" = CPU
CONF = 0.35            # raised from 0.25 - tails were picking up more false
                       # positives than heads; combined with MIN_FRAMES below,
                       # a higher threshold cuts a lot of the weakest noise
                       # before it ever reaches the tracker
TILE = 640            # must match training (0 = whole frame, NOT recommended for this model)
TILE_OVERLAP = 0.3
STRIDE = 1             # 1 = every frame; raise to skip frames if it runs too slowly
MM_PER_PX = None       # e.g. 0.08 once calibrated with a ruler / ArUco marker

MIN_FRAMES = 5         # an end must be seen this many times before it counts at
                       # all - filters one-off flicker/false-positive detections
MAX_DIST = 60          # px a tip may move between frames and still be "the same end"

# A fixed region (drawn once with select_roi.py) - anything detected outside
# it is dropped before tracking even sees it. Leave as None to skip this.
ROI_JSON = r"C:\Users\EmageVision\Documents\head_tail_classification\roi.json"

# Your setup has TWO physical tubes in frame at any time. If the model ever
# confidently misreads something else in the scene (a watch, a screw) as a
# head/tail/tubing, it will usually sit far away from both real tubes' ends.
# Set to True to drop anything not part of the EXPECTED_TUBE_COUNT largest
# spatial clusters each frame - see keep_largest_clusters()'s docstring for
# the real limitation: this is a heuristic, not a guarantee.
FILTER_ISOLATED_DETECTIONS = True
EXPECTED_TUBE_COUNT = 2  # MUST match how many real tubes are ever in frame at once
CLUSTER_DIST = 400      # px - ends closer than this to each other are "the same tube"
MIN_RATIO = 1.15       # width ratio needed to call one tube BIGGER than another
SIZE_FROM = "tail"     # compare tubing behind "tail" ends only, or "all" ends

OUTPUT_VIDEO = r"C:\Users\EmageVision\Documents\head_tail_classification\measured\testing_1_frames.mp4"
LATEST_JSON = r"C:\Users\EmageVision\Documents\head_tail_classification\measured\latest_frame.json"
HISTORY_JSONL = None   # e.g. r"...\measured\history.jsonl" to also keep every frame's result
SHOW_LIVE_PREVIEW = False
# ===================================================================================


def load_roi(path):
    """Load a polygon saved by select_roi.py. Returns an (N,2) int32 array,
    or None if ROI_JSON is not set / the file doesn't exist yet."""
    if not path or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    return np.array(data["points"], dtype=np.int32)


def filter_by_roi(ends, roi):
    """Keep only ends whose tip falls inside the ROI polygon (or on its edge).
    A fixed region drawn once, based on where the real work actually happens,
    is more predictable than per-frame clustering (keep_largest_clusters):
    it doesn't depend on how many tubes happen to be visible this frame, so
    it has none of that filter's occlusion edge case. Use both together for
    layered protection - this one for things that never belong in the scene
    at all (a watch worn at the frame's edge), the cluster filter for
    anything that manages to appear inside the ROI itself."""
    if roi is None:
        return ends
    return [e for e in ends
           if cv2.pointPolygonTest(roi, tuple(float(v) for v in e["tip"]), False) >= 0]


def keep_largest_clusters(ends, cluster_dist, k):
    """Drop ends that are spatially ISOLATED from the main groups.

    This targets a different failure than tracking: a model that confidently
    (and wrongly) recognises something ELSE in the scene as a tube part - a
    watch buckle, a screw, anything with a vaguely tube-like shape. No shape
    or confidence check reliably tells these apart from a real end, because
    the false object can genuinely look elongated/tube-like in the right pose.

    What DOES reliably tell them apart here: you know exactly how many real
    tube systems (k) are ever in frame at once, and each one's ends sit near
    EACH OTHER and near their connecting tubing. Anything sitting far off on
    its own - like a watch on a wrist at the edge of the frame while the real
    tubes are in the middle - is not one of your k tubes, however confident
    the model is. This groups this frame's end tips by mutual proximity and
    keeps only the k LARGEST groups (by how many ends are in them), discarding
    smaller, isolated ones.

    k MUST match the true number of separate tube systems that can be in
    frame at once - set it too low and you will drop a real tube; there is no
    universally safe default, which is why it is a required argument, not a
    default. This is a size-based heuristic, not certain: if a false
    detection happens to produce as many nearby end-detections as a real
    tube that frame (e.g. a real tube is mostly occluded, showing only one
    end, while the false object's bogus tubing spawns two), ranking by count
    alone cannot always tell them apart - it works well because a real tube
    usually contributes more end-detections (head+tail) than an isolated
    false one (usually just the one misclassified object)."""
    if len(ends) <= k:
        return ends
    pts = [np.asarray(e["tip"], float) for e in ends]
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
    keep = sorted(i for group in top for i in group)
    return [ends[i] for i in keep]


def build_frame_result(frame_idx, ends, tracks, last_known):
    """Only CONFIRMED tracks are reported - a track needs MIN_FRAMES sightings
    before it counts, so a one-off false detection never appears here at all.

    Two extra safety nets on top of that:
    1. HEADS/TAILS should always match (each tube has exactly one of each) -
       if they disagree, the TAIL count is forced to match HEADS, since heads
       detect far more reliably here than tails do. The raw (uncorrected)
       tail count is still reported separately (tails_count_raw) so you can
       see how often this is triggering.
    2. If no BIGGER/SMALLER comparison is available this frame (a momentary
       gap, or genuinely "SAME"), the LAST frame that did have a real answer
       keeps being shown instead of the result going blank/"SAME" - `last_known`
       is a small dict that persists across calls (create ONE per video run,
       e.g. {"bigger_tube": None, "smaller_tube": None}, and pass the SAME
       dict in every frame - do not recreate it inside the loop)."""
    conf_tracks = confirmed(tracks, MIN_FRAMES)
    sizes = size_labels([t for t in conf_tracks if SIZE_FROM == "all" or t.role == "TAIL"],
                        MIN_RATIO)
    summaries = {t.id: t.summary(MM_PER_PX, sizes.get(t.id)) for t in conf_tracks}
    pair_ends(list(summaries.values()))     # tube# pairing from the STABLE widths

    detections = []
    for e in ends:
        tid = e.get("_track_id")
        s = summaries.get(tid)
        if s is None:          # this end's track is not yet confirmed - suppress it
            continue
        detections.append({
            "id": tid,
            "role": s["role"],
            "class": e["class"],
            "status": e["status"],                # this frame's own detection status
            "bbox": e.get("bbox"),                # this frame's own box position
            "width_px": s["width_px"],             # STABLE (median over confirmed history)
            "width_mm": s["width_mm"],
            "size": s["size"],
            "pair_id": s["pair_id"],
            "width_spread_px": s["width_spread_px"],
        })

    heads_count = sum(t.role == "HEAD" for t in conf_tracks)
    tails_count_raw = sum(t.role == "TAIL" for t in conf_tracks)
    tails_count = heads_count if heads_count != tails_count_raw else tails_count_raw

    bigger = next((s for s in summaries.values() if s["size"] == "BIGGER"), None)
    smaller = next((s for s in summaries.values() if s["size"] == "SMALLER"), None)
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
    tail_note = (f"  (raw tail detections: {result['tails_count_raw']} - "
                f"corrected to match heads)"
                if result["tails_count_raw"] != result["heads_count"] else "")
    print(f"\nFrame {result['frame']}: HEADS={result['heads_count']}  "
          f"TAILS={result['tails_count']}{tail_note}")
    for d in result["detections"]:
        w = f"{d['width_mm']:.2f}mm" if d["width_mm"] else (
            f"{d['width_px']:.1f}px" if d["width_px"] else "?")
        pair = f" tube#{d['pair_id']}" if d["pair_id"] else ""
        flag = "" if d["status"] == "OK" else f"  [this frame: {d['status']}]"
        print(f"  #{d['id']} {d['role']:4s} {d['class']:10s} bbox={d['bbox']} "
              f"width={w:>9s} {d['size'] or '':7s}{pair}{flag}")
    if result["bigger_tube"]:
        line = f"  -> BIGGER tube: {result['bigger_tube']['width_px']}px"
        if result["smaller_tube"]:
            line += f"   SMALLER tube: {result['smaller_tube']['width_px']}px"
        if result["used_last_known_size"]:
            line += "   (last known good comparison - none new this frame)"
        print(line)
    elif result["heads_count"] + result["tails_count"] >= 2:
        print("  -> tube sizes look the SAME so far")
    else:
        print("  -> not enough CONFIRMED ends yet to compare sizes")


def print_diagnosis(tracks, min_frames):
    """Printed once at the end. Tells apart the two usual causes of a wrong
    HEADS/TAILS count:
      - FRAGMENTATION: the same real end got counted as several different
        tracks because its tip moved further than MAX_DIST between frames
        (fast motion, camera shake, or too small a MAX_DIST) - look for many
        tracks of one role whose first_seen frames are spread across the
        whole video, each confirmed for only a modest number of frames.
      - GENUINE EXTRA DETECTIONS: a track that is confirmed, long-lived, and
        sits in one place for a long stretch is very unlikely to be
        fragmentation - it is either a real end, or something in the scene
        the model consistently (wrongly) recognises as one.
    Check the id numbers against OUTPUT_VIDEO: if you see track ids climbing
    into the double digits while only 2 real heads are ever on screen, that
    is fragmentation - raise MAX_DIST. If instead a couple of tracks are
    confirmed and clearly sit on the wrong object in the video, that is a
    detection quality issue - retraining or raising CONF is the fix, not MAX_DIST."""
    conf = confirmed(tracks, min_frames)
    print(f"\n[diagnosis] {len(tracks)} tracks were EVER created "
          f"({len(conf)} reached MIN_FRAMES={min_frames} and were confirmed)")
    for role in ("HEAD", "TAIL"):
        rows = [t for t in tracks if t.role == role]
        conf_rows = [t for t in rows if t in conf]
        print(f"  {role}: {len(rows)} tracks ever created, {len(conf_rows)} confirmed")
        for t in sorted(rows, key=lambda t: t.first_seen):
            tag = "CONFIRMED" if t in conf else "discarded (flicker)"
            print(f"    id={t.id:<3} first_seen=frame{t.first_seen:<5} "
                 f"last_seen=frame{t.last_seen:<5} seen {t.frames} time(s)  "
                 f"first tip={tuple(round(v) for v in t.first_tip)}  [{tag}]")
    print("  -> if one role shows MANY short-lived confirmed tracks spread "
         "across the video, that is track FRAGMENTATION (same real end, "
         "counted repeatedly): try raising MAX_DIST.")
    print("  -> if instead a role shows a small number of confirmed tracks "
         "each seen for a long, continuous stretch, those are genuine "
         "detections - check in OUTPUT_VIDEO what they actually sit on.")


def main():
    model = YOLO(MODEL_PATH)
    roi = load_roi(ROI_JSON)
    print(f"[roi] {'using ' + str(len(roi)) + '-point ROI from ' + ROI_JSON if roi is not None else 'none set - all detections considered'}")
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

    tracks = []
    last_known = {"bigger_tube": None, "smaller_tube": None}
    writer = None
    preview = SHOW_LIVE_PREVIEW
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
            ends = [e for e in analyse(frame, inst, MM_PER_PX, gap=3, class_map=class_map)
                    if "tip" in e]
            ends = filter_by_roi(ends, roi)
            if FILTER_ISOLATED_DETECTIONS:
                ends = keep_largest_clusters(ends, CLUSTER_DIST, EXPECTED_TUBE_COUNT)
            match(tracks, ends, idx, MAX_DIST)
            result = build_frame_result(idx, ends, tracks, last_known)
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
    print_diagnosis(tracks, MIN_FRAMES)
    print("\n[done]")


if __name__ == "__main__":
    main()
