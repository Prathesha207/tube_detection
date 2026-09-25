# """
# video_ends.py - count heads and tails in a VIDEO and measure the tubes.

# For every frame:
#     YOLO11-seg masks -> ends attached to their tubing -> tip + width
# Across frames:
#     each end is followed by its tip position, so it is counted ONCE, and its
#     width is the MEDIAN over every frame it was seen in (far steadier than one
#     frame). An end must be seen in --min-frames frames before it counts, which
#     drops flicker detections.
# Result:
#     how many heads, how many tails, the width of the tube behind each tail,
#     and which of those tubes is the bigger one.

#   python video_ends.py --video clip.mp4 --model best-seg.pt --output out.mp4 --report out.json
#   add --mm-per-px 0.08 for millimetres (ruler or ArUco in the frame)
#   add --size-from all to compare the tubing behind heads as well as tails
# """
# import argparse
# import json
# import time
# from collections import Counter, deque
# from pathlib import Path

# import cv2
# import numpy as np

# from measure_from_masks import NAMES, ROLE, analyse, masks_from_model, build_class_map, pair_ends

# PANEL_BG = (22, 22, 28)
# COL = {"HEAD": (255, 160, 0), "TAIL": (0, 200, 255)}
# COL_BIG, COL_SMALL, COL_WARN = (80, 220, 120), (200, 200, 200), (60, 160, 255)


# class Track:
#     """One physical end, followed across frames by the position of its tip."""

#     def __init__(self, tid, role, smooth=0.4):
#         self.id = tid
#         self.role = role
#         self.smooth = smooth
#         self.tip = None
#         self.cls_votes = Counter()
#         self.widths = deque(maxlen=400)
#         self.trans = deque(maxlen=400)
#         self.frames = 0
#         self.last_seen = -1

#     def update(self, frame_idx, end):
#         tip = np.asarray(end["tip"], float)
#         self.tip = tip if self.tip is None else self.smooth * tip + (1 - self.smooth) * self.tip
#         self.cls_votes[end["class"]] += 1
#         if end.get("width_px") and end["status"] == "OK":
#             self.widths.append(end["width_px"])
#         if end.get("transparency") is not None:
#             self.trans.append(end["transparency"])
#         self.frames += 1
#         self.last_seen = frame_idx

#     @property
#     def cls(self):
#         return self.cls_votes.most_common(1)[0][0] if self.cls_votes else "?"

#     @property
#     def width(self):
#         return float(np.median(self.widths)) if self.widths else None

#     def summary(self, mm_per_px, size):
#         w = self.width
#         return {
#             "id": self.id, "role": self.role, "class": self.cls, "frames": self.frames,
#             "width_px": round(w, 2) if w else None,
#             "width_mm": round(w * mm_per_px, 2) if (w and mm_per_px) else None,
#             "width_spread_px": round(float(np.std(self.widths)), 2) if self.widths else None,
#             "measured_frames": len(self.widths),
#             "transparency": round(float(np.median(self.trans)), 2) if self.trans else None,
#             "size": size,
#         }


# def match(tracks, ends, frame_idx, max_dist):
#     """Greedy nearest-tip matching; a detection too far from every track starts a new one."""
#     free = [t for t in tracks if t.tip is not None]
#     pairs = []
#     for ei, e in enumerate(ends):
#         for t in free:
#             d = float(np.linalg.norm(np.asarray(e["tip"], float) - t.tip))
#             if d <= max_dist and t.role == e["role"] and t.last_seen != frame_idx:
#                 pairs.append((d, ei, t))
#     pairs.sort(key=lambda p: p[0])
#     used_e, used_t = set(), set()
#     for d, ei, t in pairs:
#         if ei in used_e or id(t) in used_t:
#             continue
#         t.update(frame_idx, ends[ei])
#         used_e.add(ei)
#         used_t.add(id(t))
#     for ei, e in enumerate(ends):
#         if ei not in used_e:
#             t = Track(len(tracks) + 1, e["role"])
#             t.update(frame_idx, e)
#             tracks.append(t)


# def size_labels(tracks, min_ratio):
#     """Split the tube widths at their largest gap -> BIGGER / SMALLER / SAME."""
#     ok = [t for t in tracks if t.width]
#     if len(ok) < 2:
#         return {t.id: "SAME" for t in ok}
#     ws = np.sort([t.width for t in ok])
#     ratios = ws[1:] / ws[:-1]
#     if ratios.max() < min_ratio:
#         return {t.id: "SAME" for t in ok}
#     i = int(ratios.argmax())
#     cut = (ws[i] + ws[i + 1]) / 2
#     return {t.id: ("BIGGER" if t.width > cut else "SMALLER") for t in ok}


# def confirmed(tracks, min_frames):
#     return [t for t in tracks if t.frames >= min_frames]


# def fmt_w(w, mm_per_px):
#     if w is None:
#         return "-"
#     return f"{w * mm_per_px:.2f} mm" if mm_per_px else f"{w:.1f} px"


# def draw_panel(frame, tracks, sizes, mm_per_px, min_frames, size_from, frame_idx, total):
#     conf = confirmed(tracks, min_frames)
#     heads = [t for t in conf if t.role == "HEAD"]
#     tails = [t for t in conf if t.role == "TAIL"]
#     tubes = sorted([t for t in conf if t.width and (size_from == "all" or t.role == "TAIL")],
#                    key=lambda t: -t.width)
#     rows = 3 + max(len(tubes), 1)
#     h = 30 + (rows + 1) * 26        # +1 row for the frame counter, kept inside the panel
#     w = 330
#     panel = frame[10:10 + h, 10:10 + w]
#     if panel.shape[:2] == (h, w):
#         frame[10:10 + h, 10:10 + w] = cv2.addWeighted(
#             np.full_like(panel, PANEL_BG), 0.72, panel, 0.28, 0)
#     cv2.rectangle(frame, (10, 10), (10 + w, 10 + h), (90, 90, 100), 1)

#     def line(i, text, colour, scale=0.55, thick=1):
#         cv2.putText(frame, text, (24, 40 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
#                     scale, colour, thick, cv2.LINE_AA)

#     line(0, f"HEADS: {len(heads)}", COL["HEAD"], 0.62)
#     line(1, f"TAILS: {len(tails)}", COL["TAIL"], 0.62)
#     line(2, "TUBE WIDTHS" + ("" if size_from == "all" else " (behind tails)"), (170, 170, 175), 0.5)
#     if not tubes:
#         line(3, "measuring...", (150, 150, 150), 0.5)
#     for i, t in enumerate(tubes):
#         s = sizes.get(t.id, "")
#         c = COL_BIG if s == "BIGGER" else COL_SMALL
#         spread = float(np.std(t.widths)) if len(t.widths) > 2 else 0.0
#         warn = "  ?" if (t.width and spread > 0.15 * t.width) else ""
#         line(3 + i, f"{t.role} #{t.id}: {fmt_w(t.width, mm_per_px)}  {s}{warn}",
#              COL_WARN if warn else c, 0.55)
#     line(rows, f"frame {frame_idx}" if total <= 0 else f"frame {frame_idx}/{total}",
#         (140, 140, 145), 0.45)


# PALETTE = [(255, 120, 0), (230, 230, 230), (0, 200, 255), (0, 180, 0),
#           (200, 120, 255), (120, 200, 255)]


# def draw_ends(frame, instances, ends, tracks, sizes, mm_per_px, frame_idx, tubing_id=None):
#     overlay = frame.copy()
#     fills = {} if tubing_id is None else {tubing_id: (0, 180, 0)}
#     for c, m in instances:
#         col = fills.setdefault(c, PALETTE[len(fills) % len(PALETTE)])
#         overlay[m] = col
#     cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
#     for t in tracks:
#         if t.last_seen != frame_idx or t.tip is None:
#             continue
#         p = tuple(int(v) for v in t.tip)
#         c = COL[t.role]
#         cv2.circle(frame, p, 6, (0, 0, 0), -1)
#         cv2.circle(frame, p, 4, c, -1)
#         txt = f"#{t.id} {t.role} {fmt_w(t.width, mm_per_px)} {sizes.get(t.id, '')}".strip()
#         cv2.putText(frame, txt, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 0, 0), 3, cv2.LINE_AA)
#         cv2.putText(frame, txt, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, c, 1, cv2.LINE_AA)


# def run(model, a, class_map=None):
#     src = int(a.video) if str(a.video).isdigit() else a.video   # "0" -> webcam 0
#     cap = cv2.VideoCapture(src)
#     if not cap.isOpened():
#         raise SystemExit(f"cannot open {a.video}")
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     tubing_id = class_map[2] if class_map else None
#     tracks, writer, idx, sizes = [], None, 0, {}
#     preview = [not a.no_preview]
#     processed, t0 = 0, time.time()
#     try:
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             idx += 1
#             if idx % a.stride:
#                 continue
#             inst = masks_from_model(model, frame, a.conf, not a.no_gray, a.tile, a.tile_overlap,
#                                     batch_tiles=not a.no_batch_tiles)
#             ends = [e for e in analyse(frame, inst, a.mm_per_px, a.max_disagree, a.attach_gap, class_map)
#                     if "tip" in e]
#             match(tracks, ends, idx, a.max_dist)
#             sizes = size_labels([t for t in confirmed(tracks, a.min_frames)
#                                  if a.size_from == "all" or t.role == "TAIL"], a.min_ratio)
#             draw_ends(frame, inst, ends, tracks, sizes, a.mm_per_px, idx, tubing_id)
#             draw_panel(frame, tracks, sizes, a.mm_per_px, a.min_frames, a.size_from, idx, total)
#             processed += 1
#             if processed % max(a.progress_every, 1) == 0:
#                 elapsed = time.time() - t0
#                 rate = processed / elapsed if elapsed else 0
#                 remaining = (total // a.stride - processed) / rate if (rate and total > 0) else None
#                 eta = f"~{remaining/60:.1f} min left" if remaining is not None else "live"
#                 print(f"[progress] frame {idx}{'' if total <= 0 else f'/{total}'}  "
#                       f"{processed} processed  {rate:.2f} frames/s  {eta}", flush=True)
#             if writer is None and a.output:
#                 h, w = frame.shape[:2]          # from the real frame: phones rotate video
#                 Path(a.output).parent.mkdir(parents=True, exist_ok=True)
#                 writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"),
#                                          fps / a.stride, (w, h))
#                 if not writer.isOpened():
#                     raise SystemExit(f"cannot write {a.output} (give a file path, e.g. out.mp4)")
#             if writer:
#                 writer.write(frame)
#             if preview[0]:
#                 try:
#                     s = 900 / frame.shape[0]
#                     cv2.imshow("tube ends", cv2.resize(frame, None, fx=s, fy=s))
#                     if cv2.waitKey(1) & 0xFF == ord("q"):
#                         break
#                 except cv2.error:      # headless OpenCV build: carry on without a window
#                     preview[0] = False
#                     print("[info] no GUI support in this OpenCV build - preview disabled")
#     finally:
#         cap.release()
#         if writer:
#             writer.release()
#         if preview[0]:
#             try:
#                 cv2.destroyAllWindows()
#             except cv2.error:
#                 pass

#     conf = confirmed(tracks, a.min_frames)
#     end_summaries = [t.summary(a.mm_per_px, sizes.get(t.id)) for t in conf]
#     pair_ends(end_summaries)     # which head belongs to which tail, by matching width
#     return {
#         "video": a.video,
#         "frames_processed": idx // a.stride,
#         "heads": sum(t.role == "HEAD" for t in conf),
#         "tails": sum(t.role == "TAIL" for t in conf),
#         "ends": end_summaries,
#     }


# def main():
#     ap = argparse.ArgumentParser(description=__doc__,
#                                  formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--video", required=True)
#     ap.add_argument("--model", required=True, help="trained YOLO11-seg weights")
#     ap.add_argument("--output", help="annotated video, e.g. measured\\clip.mp4")
#     ap.add_argument("--report", help="JSON report path")
#     ap.add_argument("--conf", type=float, default=0.35)
#     ap.add_argument("--device", default=None,
#                     help="0 for first GPU, cpu for CPU, blank = let Ultralytics decide")
#     ap.add_argument("--tile", type=int, default=0, help="tile size for inference, e.g. 640; 0 = whole frame")
#     ap.add_argument("--tile-overlap", type=float, default=0.3)
#     ap.add_argument("--no-batch-tiles", action="store_true",
#                     help="send tiles to the model one at a time instead of as one batch")
#     ap.add_argument("--no-gray", action="store_true", help="model was trained on colour images")
#     ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
#     ap.add_argument("--progress-every", type=int, default=10,
#                     help="print a progress line every N processed frames")
#     ap.add_argument("--min-frames", type=int, default=5,
#                     help="frames an end must appear in before it is counted")
#     ap.add_argument("--max-dist", type=float, default=60,
#                     help="px a tip may move between processed frames and still be the same end")
#     ap.add_argument("--min-ratio", type=float, default=1.15,
#                     help="width ratio needed to call one tube bigger")
#     ap.add_argument("--size-from", choices=["tail", "all"], default="tail",
#                     help="compare the tubing behind tails only (default) or behind every end")
#     ap.add_argument("--attach-gap", type=int, default=3)
#     ap.add_argument("--max-disagree", type=float, default=0.2)
#     ap.add_argument("--mm-per-px", type=float, default=None)
#     ap.add_argument("--no-preview", action="store_true")
#     a = ap.parse_args()

#     from ultralytics import YOLO
#     model = YOLO(a.model)
#     if a.device is not None:
#         model.to(f"cuda:{a.device}" if a.device.isdigit() else a.device)
#     class_map = build_class_map(model.names)
#     print(f"[classes] {model.names} -> tubing id {class_map[2]}, roles {class_map[1]}")
#     rep = run(model, a, class_map)
#     print(f"\nHEADS: {rep['heads']}   TAILS: {rep['tails']}")
#     for e in rep["ends"]:
#         pair = f"  tube #{e['pair_id']}" if e.get("pair_id") else "  (unpaired)"
#         print(f"  #{e['id']} {e['role']:4s} {e['class']:10s} "
#               f"width={e['width_mm'] or e['width_px']}"
#               f"{'mm' if e['width_mm'] else 'px'} "
#               f"+-{e['width_spread_px']} over {e['measured_frames']} frames  {e['size'] or ''}{pair}")
#     if a.report:
#         Path(a.report).write_text(json.dumps(rep, indent=2))
#         print(f"[report] {a.report}")


# if __name__ == "__main__":
#     main()

"""
video_ends.py - count heads and tails in a VIDEO and measure the tubes.

For every frame:
    YOLO11-seg masks -> ends attached to their tubing -> tip + width
Across frames:
    each end is followed by its tip position, so it is counted ONCE, and its
    width is the MEDIAN over every frame it was seen in (far steadier than one
    frame). An end must be seen in --min-frames frames before it counts, which
    drops flicker detections.
Result:
    how many heads, how many tails, the width of the tube behind each tail,
    and which of those tubes is the bigger one.

  python video_ends.py --video clip.mp4 --model best-seg.pt --output out.mp4 --report out.json
  add --mm-per-px 0.08 for millimetres (ruler or ArUco in the frame)
  add --size-from all to compare the tubing behind heads as well as tails
"""
import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from measure_from_masks import NAMES, ROLE, analyse, masks_from_model, build_class_map, pair_ends

PANEL_BG = (22, 22, 28)
COL = {"HEAD": (255, 160, 0), "TAIL": (0, 200, 255)}
COL_BIG, COL_SMALL, COL_WARN = (80, 220, 120), (200, 200, 200), (60, 160, 255)


class Track:
    """One physical end, followed across frames by the position of its tip."""

    def __init__(self, tid, role, smooth=0.4):
        self.id = tid
        self.role = role
        self.smooth = smooth
        self.tip = None
        self.first_tip = None
        self.cls_votes = Counter()
        self.widths = deque(maxlen=400)
        self.trans = deque(maxlen=400)
        self.frames = 0
        self.first_seen = None
        self.last_seen = -1

    def update(self, frame_idx, end):
        tip = np.asarray(end["tip"], float)
        if self.tip is None:
            self.first_tip = tip.copy()
            self.first_seen = frame_idx
        self.tip = tip if self.tip is None else self.smooth * tip + (1 - self.smooth) * self.tip
        self.cls_votes[end["class"]] += 1
        if end.get("width_px") and end["status"] == "OK":
            self.widths.append(end["width_px"])
        if end.get("transparency") is not None:
            self.trans.append(end["transparency"])
        self.frames += 1
        self.last_seen = frame_idx

    @property
    def cls(self):
        return self.cls_votes.most_common(1)[0][0] if self.cls_votes else "?"

    @property
    def width(self):
        return float(np.median(self.widths)) if self.widths else None

    def summary(self, mm_per_px, size):
        w = self.width
        return {
            "id": self.id, "role": self.role, "class": self.cls, "frames": self.frames,
            "width_px": round(w, 2) if w else None,
            "width_mm": round(w * mm_per_px, 2) if (w and mm_per_px) else None,
            "width_spread_px": round(float(np.std(self.widths)), 2) if self.widths else None,
            "measured_frames": len(self.widths),
            "transparency": round(float(np.median(self.trans)), 2) if self.trans else None,
            "size": size,
        }



def match(tracks, ends, frame_idx, max_dist, width_reid_frac=0.15, width_reid_floor=15.0):
    """Match each detection to a track in two passes.

    Pass 1 - spatial: nearest tip within max_dist, same role. This is the
    normal case: an end sitting roughly where it was last frame.

    Pass 2 - width re-identification, for anything pass 1 left unmatched: an
    object that gets PICKED UP and carried somewhere else can jump far more
    than any reasonable max_dist in a single frame - no distance threshold
    survives that. But its measured tube width doesn't change when it moves,
    so an unmatched detection is re-attached to an existing, currently-idle
    track of the same role whose established width is within
    width_reid_frac of this detection's width. This is the same idea as
    pair_ends() (identity by diameter, since position can't be trusted), just
    applied over TIME instead of across two roles in one frame. A track with
    no confirmed width yet (not enough good measurements so far) is skipped
    for re-id - too little evidence to trust its width - and tracks already
    matched this frame are excluded, so this cannot merge two genuinely
    different objects that are both visible at the same time.

    Tags each end dict with '_track_id' = the track it was assigned to, so a
    caller can join this frame's raw detection (e.g. its bbox) back to that
    track's stable, multi-frame history (its median width, confirmation status)."""
    free = [t for t in tracks if t.tip is not None]
    pairs = []
    for ei, e in enumerate(ends):
        for t in free:
            d = float(np.linalg.norm(np.asarray(e["tip"], float) - t.tip))
            if d <= max_dist and t.role == e["role"] and t.last_seen != frame_idx:
                pairs.append((d, ei, t))
    pairs.sort(key=lambda p: p[0])
    used_e, used_t = set(), set()
    for d, ei, t in pairs:
        if ei in used_e or id(t) in used_t:
            continue
        t.update(frame_idx, ends[ei])
        ends[ei]["_track_id"] = t.id
        used_e.add(ei)
        used_t.add(id(t))

    # pass 2: width re-id for whatever pass 1 couldn't place
    for ei, e in enumerate(ends):
        if ei in used_e or not e.get("width_px"):
            continue
        best, best_diff = None, None
        for t in tracks:
            if id(t) in used_t or t.role != e["role"] or t.last_seen == frame_idx or not t.width:
                continue
            # diff = abs(e["width_px"] - t.width) / min(e["width_px"], t.width)
            denom = max(min(e["width_px"], t.width), width_reid_floor)
            diff = abs(e["width_px"] - t.width) / denom
            if diff <= width_reid_frac and (best_diff is None or diff < best_diff):
                best, best_diff = t, diff
        if best is not None:
            best.update(frame_idx, e)
            e["_track_id"] = best.id
            used_e.add(ei)
            used_t.add(id(best))

    for ei, e in enumerate(ends):
        if ei not in used_e:
            t = Track(len(tracks) + 1, e["role"])
            t.update(frame_idx, e)
            e["_track_id"] = t.id
            tracks.append(t)


def size_labels(tracks, min_ratio):
    """Split the tube widths at their largest gap -> BIGGER / SMALLER / SAME."""
    ok = [t for t in tracks if t.width]
    if len(ok) < 2:
        return {t.id: "SAME" for t in ok}
    ws = np.sort([t.width for t in ok])
    ratios = ws[1:] / ws[:-1]
    if ratios.max() < min_ratio:
        return {t.id: "SAME" for t in ok}
    i = int(ratios.argmax())
    cut = (ws[i] + ws[i + 1]) / 2
    return {t.id: ("BIGGER" if t.width > cut else "SMALLER") for t in ok}


def confirmed(tracks, min_frames):
    return [t for t in tracks if t.frames >= min_frames]


def fmt_w(w, mm_per_px):
    if w is None:
        return "-"
    return f"{w * mm_per_px:.2f} mm" if mm_per_px else f"{w:.1f} px"


def draw_panel(frame, tracks, sizes, mm_per_px, min_frames, size_from, frame_idx, total):
    conf = confirmed(tracks, min_frames)
    heads = [t for t in conf if t.role == "HEAD"]
    tails = [t for t in conf if t.role == "TAIL"]
    tubes = sorted([t for t in conf if t.width and (size_from == "all" or t.role == "TAIL")],
                   key=lambda t: -t.width)
    rows = 3 + max(len(tubes), 1)
    h = 30 + (rows + 1) * 26        # +1 row for the frame counter, kept inside the panel
    w = 330
    panel = frame[10:10 + h, 10:10 + w]
    if panel.shape[:2] == (h, w):
        frame[10:10 + h, 10:10 + w] = cv2.addWeighted(
            np.full_like(panel, PANEL_BG), 0.72, panel, 0.28, 0)
    cv2.rectangle(frame, (10, 10), (10 + w, 10 + h), (90, 90, 100), 1)

    def line(i, text, colour, scale=0.55, thick=1):
        cv2.putText(frame, text, (24, 40 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thick, cv2.LINE_AA)

    line(0, f"HEADS: {len(heads)}", COL["HEAD"], 0.62)
    line(1, f"TAILS: {len(tails)}", COL["TAIL"], 0.62)
    line(2, "TUBE WIDTHS" + ("" if size_from == "all" else " (behind tails)"), (170, 170, 175), 0.5)
    if not tubes:
        line(3, "measuring...", (150, 150, 150), 0.5)
    for i, t in enumerate(tubes):
        s = sizes.get(t.id, "")
        c = COL_BIG if s == "BIGGER" else COL_SMALL
        spread = float(np.std(t.widths)) if len(t.widths) > 2 else 0.0
        warn = "  ?" if (t.width and spread > 0.15 * t.width) else ""
        line(3 + i, f"{t.role} #{t.id}: {fmt_w(t.width, mm_per_px)}  {s}{warn}",
             COL_WARN if warn else c, 0.55)
    line(rows, f"frame {frame_idx}" if total <= 0 else f"frame {frame_idx}/{total}",
        (140, 140, 145), 0.45)


PALETTE = [(255, 120, 0), (230, 230, 230), (0, 200, 255), (0, 180, 0),
          (200, 120, 255), (120, 200, 255)]


def draw_ends(frame, instances, ends, tracks, sizes, mm_per_px, frame_idx, tubing_id=None):
    overlay = frame.copy()
    fills = {} if tubing_id is None else {tubing_id: (0, 180, 0)}
    for c, m in instances:
        col = fills.setdefault(c, PALETTE[len(fills) % len(PALETTE)])
        overlay[m] = col
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    for t in tracks:
        if t.last_seen != frame_idx or t.tip is None:
            continue
        p = tuple(int(v) for v in t.tip)
        c = COL[t.role]
        cv2.circle(frame, p, 6, (0, 0, 0), -1)
        cv2.circle(frame, p, 4, c, -1)
        txt = f"#{t.id} {t.role} {fmt_w(t.width, mm_per_px)} {sizes.get(t.id, '')}".strip()
        cv2.putText(frame, txt, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1, cv2.LINE_AA)


def run(model, a, class_map=None):
    src = int(a.video) if str(a.video).isdigit() else a.video   # "0" -> webcam 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tubing_id = class_map[2] if class_map else None
    tracks, writer, idx, sizes = [], None, 0, {}
    preview = [not a.no_preview]
    processed, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % a.stride:
                continue
            inst = masks_from_model(model, frame, a.conf, not a.no_gray, a.tile, a.tile_overlap,
                                    batch_tiles=not a.no_batch_tiles)
            ends = [e for e in analyse(frame, inst, a.mm_per_px, a.max_disagree, a.attach_gap, class_map)
                    if "tip" in e]
            match(tracks, ends, idx, a.max_dist)
            sizes = size_labels([t for t in confirmed(tracks, a.min_frames)
                                 if a.size_from == "all" or t.role == "TAIL"], a.min_ratio)
            draw_ends(frame, inst, ends, tracks, sizes, a.mm_per_px, idx, tubing_id)
            draw_panel(frame, tracks, sizes, a.mm_per_px, a.min_frames, a.size_from, idx, total)
            processed += 1
            if processed % max(a.progress_every, 1) == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed else 0
                remaining = (total // a.stride - processed) / rate if (rate and total > 0) else None
                eta = f"~{remaining/60:.1f} min left" if remaining is not None else "live"
                print(f"[progress] frame {idx}{'' if total <= 0 else f'/{total}'}  "
                      f"{processed} processed  {rate:.2f} frames/s  {eta}", flush=True)
            if writer is None and a.output:
                h, w = frame.shape[:2]          # from the real frame: phones rotate video
                Path(a.output).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps / a.stride, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"cannot write {a.output} (give a file path, e.g. out.mp4)")
            if writer:
                writer.write(frame)
            if preview[0]:
                try:
                    s = 900 / frame.shape[0]
                    cv2.imshow("tube ends", cv2.resize(frame, None, fx=s, fy=s))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:      # headless OpenCV build: carry on without a window
                    preview[0] = False
                    print("[info] no GUI support in this OpenCV build - preview disabled")
    finally:
        cap.release()
        if writer:
            writer.release()
        if preview[0]:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    conf = confirmed(tracks, a.min_frames)
    end_summaries = [t.summary(a.mm_per_px, sizes.get(t.id)) for t in conf]
    pair_ends(end_summaries)     # which head belongs to which tail, by matching width
    return {
        "video": a.video,
        "frames_processed": idx // a.stride,
        "heads": sum(t.role == "HEAD" for t in conf),
        "tails": sum(t.role == "TAIL" for t in conf),
        "ends": end_summaries,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="trained YOLO11-seg weights")
    ap.add_argument("--output", help="annotated video, e.g. measured\\clip.mp4")
    ap.add_argument("--report", help="JSON report path")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None,
                    help="0 for first GPU, cpu for CPU, blank = let Ultralytics decide")
    ap.add_argument("--tile", type=int, default=0, help="tile size for inference, e.g. 640; 0 = whole frame")
    ap.add_argument("--tile-overlap", type=float, default=0.3)
    ap.add_argument("--no-batch-tiles", action="store_true",
                    help="send tiles to the model one at a time instead of as one batch")
    ap.add_argument("--no-gray", action="store_true", help="model was trained on colour images")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="print a progress line every N processed frames")
    ap.add_argument("--min-frames", type=int, default=5,
                    help="frames an end must appear in before it is counted")
    ap.add_argument("--max-dist", type=float, default=60,
                    help="px a tip may move between processed frames and still be the same end")
    ap.add_argument("--min-ratio", type=float, default=1.15,
                    help="width ratio needed to call one tube bigger")
    ap.add_argument("--size-from", choices=["tail", "all"], default="tail",
                    help="compare the tubing behind tails only (default) or behind every end")
    ap.add_argument("--attach-gap", type=int, default=3)
    ap.add_argument("--max-disagree", type=float, default=0.2)
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    if a.device is not None:
        model.to(f"cuda:{a.device}" if a.device.isdigit() else a.device)
    class_map = build_class_map(model.names)
    print(f"[classes] {model.names} -> tubing id {class_map[2]}, roles {class_map[1]}")
    rep = run(model, a, class_map)
    print(f"\nHEADS: {rep['heads']}   TAILS: {rep['tails']}")
    for e in rep["ends"]:
        pair = f"  tube #{e['pair_id']}" if e.get("pair_id") else "  (unpaired)"
        print(f"  #{e['id']} {e['role']:4s} {e['class']:10s} "
              f"width={e['width_mm'] or e['width_px']}"
              f"{'mm' if e['width_mm'] else 'px'} "
              f"+-{e['width_spread_px']} over {e['measured_frames']} frames  {e['size'] or ''}{pair}")
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2))
        print(f"[report] {a.report}")


if __name__ == "__main__":
    main()