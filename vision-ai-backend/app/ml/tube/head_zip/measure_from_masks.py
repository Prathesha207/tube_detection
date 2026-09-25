# # """
# # measure_from_masks.py - head/tail, transparency and width from SEGMENTATION MASKS.
# #                                             (later, --model)

# # Classes: however many your model has, matched by NAME not by position:
# #   any class whose name contains "head"   -> role HEAD
# #   any class whose name contains "tail"   -> role TAIL
# #   the class named exactly "tubing"       -> the tubing mask
# # This matters because retraining can change how many classes exist and in what
# # order (e.g. merging head_blue/head_black into one "head" class shifts every
# # later index) - a hardcoded index like "tubing = class 3" silently breaks the
# # moment the class count changes: every end then finds NO tubing mask and no
# # width is ever computed. Matching by name survives that.

# #   python measure_from_masks.py --images photos/ --labels cvat_export/labels/ --output measured/
# #   python measure_from_masks.py --images photos/ --model best-seg.pt --output measured/
# # """
# # import argparse
# # import json
# # from pathlib import Path

# # import cv2
# # import numpy as np
# # from skimage.morphology import skeletonize

# # from tube_ends import measure_width, transparency_score, colour_score

# # # Fallback class list, used ONLY for --labels (hand-drawn CVAT .txt export),
# # # which carries no embedded names - it must match your data.yaml order exactly.
# # # --model needs no such list: it reads names straight from the checkpoint.
# # FALLBACK_NAMES = ["head_blue", "head_black", "tail_open", "tubing"]


# # def build_class_map(names):
# #     """names: {id: name} (from a YOLO model) or a list in id order (CVAT fallback).
# #     Returns (names_by_id, role_by_id, tubing_id) built by matching NAME, not position."""
# #     if isinstance(names, (list, tuple)):
# #         names = {i: n for i, n in enumerate(names)}
# #     role, tubing_id = {}, None
# #     for cid, name in names.items():
# #         n = name.lower()
# #         if "tubing" == n or "tube" in n:
# #             tubing_id = cid
# #         elif "head" in n:
# #             role[cid] = "HEAD"
# #         elif "tail" in n:
# #             role[cid] = "TAIL"
# #     if tubing_id is None:
# #         raise ValueError(f"no class named 'tubing' found among {names} - "
# #                          "check the model's / data.yaml's class names")
# #     return names, role, tubing_id


# # # Fallback map for the --labels path (see FALLBACK_NAMES above).
# # NAMES, ROLE, TUBING = build_class_map(FALLBACK_NAMES)


# # # ---------------------------------------------------------------- inputs
# # def masks_from_yolo_txt(txt, h, w):
# #     """YOLO-seg label file -> list of (class_id, bool mask)."""
# #     out = []
# #     if not Path(txt).exists():
# #         return out
# #     for line in Path(txt).read_text().splitlines():
# #         v = line.split()
# #         if len(v) < 7:
# #             continue
# #         cls = int(v[0])
# #         pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
# #         m = np.zeros((h, w), np.uint8)
# #         cv2.fillPoly(m, [pts], 1)
# #         out.append((cls, m.astype(bool)))
# #     return out


# # def _predict(model, img, conf, gray):
# #     inp = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else img
# #     r = model(inp, conf=conf, retina_masks=True, verbose=False)[0]
# #     if r.masks is None:
# #         return []
# #     ms = r.masks.data.cpu().numpy() > 0.5
# #     cls = r.boxes.cls.cpu().tolist()
# #     cfs = r.boxes.conf.cpu().tolist()
# #     return list(zip([int(c) for c in cls], ms, cfs))


# # def _predict_batch(model, imgs, conf, gray):
# #     """Same as _predict but for a LIST of equally-sized crops in one model call -
# #     the point of tiling on GPU: one batched forward pass instead of one per tile."""
# #     if gray:
# #         imgs = [cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for im in imgs]
# #     results = model(imgs, conf=conf, retina_masks=True, verbose=False)
# #     out = []
# #     for r in results:
# #         if r.masks is None:
# #             out.append([])
# #             continue
# #         ms = r.masks.data.cpu().numpy() > 0.5
# #         cls = r.boxes.cls.cpu().tolist()
# #         cfs = r.boxes.conf.cpu().tolist()
# #         out.append(list(zip([int(c) for c in cls], ms, cfs)))
# #     return out


# # def masks_from_model(model, bgr, conf, gray=True, tile=0, overlap=0.3, dedupe_iou=0.4, batch_tiles=True):
# #     """Masks for the whole photo, or - with tile>0 - from overlapping tiles.

# #     Tiling matters when the ends are small: a 640 px tile of a 1920 px photo
# #     gives the model 3x the pixels on the same object. On GPU the tiles are sent
# #     as ONE batched call (batch_tiles=True) so the speed hit from tiling is small;
# #     on CPU a batch is processed sequentially anyway, so either mode works.
# #     Duplicates from the overlap are removed by IoU, keeping the more confident one."""
# #     if not tile:
# #         return [(c, m) for c, m, _ in _predict(model, bgr, conf, gray)]
# #     H, W = bgr.shape[:2]
# #     step = max(int(tile * (1 - overlap)), 32)
# #     xs = sorted({*range(0, max(W - tile, 0) + 1, step), max(W - tile, 0)})
# #     ys = sorted({*range(0, max(H - tile, 0) + 1, step), max(H - tile, 0)})
# #     origins = [(x, y) for y in ys for x in xs]
# #     crops = [bgr[y:y + tile, x:x + tile] for x, y in origins]
# #     # pad any short edge tile to a uniform size so they can be batched together
# #     crops = [c if c.shape[:2] == (tile, tile) else
# #              cv2.copyMakeBorder(c, 0, tile - c.shape[0], 0, tile - c.shape[1],
# #                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
# #              for c in crops]
# #     per_tile = (_predict_batch(model, crops, conf, gray) if batch_tiles
# #                 else [_predict(model, c, conf, gray) for c in crops])
# #     dets = []
# #     for (x, y), tile_dets in zip(origins, per_tile):
# #         for c, m, cf in tile_dets:
# #             full = np.zeros((H, W), bool)
# #             mh, mw = m.shape
# #             th, tw = min(mh, H - y), min(mw, W - x)
# #             full[y:y + th, x:x + tw] = m[:th, :tw]
# #             if full.any():
# #                 dets.append((c, full, cf))
# #     kept = []
# #     for c, m, cf in sorted(dets, key=lambda d: -d[2]):
# #         if any(kc == c and (km & m).sum() > dedupe_iou * min(km.sum(), m.sum())
# #                for kc, km, _ in kept):
# #             continue
# #         kept.append((c, m, cf))
# #     return [(c, m) for c, m, _ in kept]


# # # ---------------------------------------------------------------- geometry
# # def attach(end_mask, tubings, gap=3):
# #     """Index of the tubing mask that touches this end (largest contact).
# #     `gap` px of separation between the two polygons is tolerated."""
# #     k = 2 * int(max(gap, 0)) + 1
# #     grown = cv2.dilate(end_mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
# #     best, best_n = None, 0
# #     for i, t in enumerate(tubings):
# #         n = int((grown & t).sum())
# #         if n > best_n:
# #             best, best_n = i, n
# #     if best is None:
# #         return None, None
# #     ys, xs = np.nonzero(grown & tubings[best])
# #     return best, np.array([xs.mean(), ys.mean()], np.float32)


# # def local_axis(tube_mask, contact, radius):
# #     """Unit direction of the tube near `contact`, pointing away from the end,
# #     plus the skeleton points used and the mask width there."""
# #     skel = skeletonize(tube_mask)
# #     dt = cv2.distanceTransform(tube_mask.astype(np.uint8), cv2.DIST_L2, 5)
# #     # drop the short corner branches a skeleton grows at polygon ends: they sit
# #     # close to the border (small distance value) and would tilt the axis
# #     skel &= dt >= 0.7 * dt[skel].max() if skel.any() else skel
# #     ys, xs = np.nonzero(skel)
# #     if len(xs) < 3:
# #         return None, None, None
# #     pts = np.stack([xs, ys], 1).astype(np.float32)
# #     near = pts[np.linalg.norm(pts - contact, axis=1) < radius]
# #     if len(near) < 3:
# #         near = pts[np.argsort(np.linalg.norm(pts - contact, axis=1))[:10]]
# #     c = near.mean(0)
# #     _, _, vt = np.linalg.svd(near - c)
# #     u = vt[0]
# #     if np.dot(c - contact, u) < 0:        # point from the end into the tubing
# #         u = -u
# #     mask_width = 2.0 * float(np.median(dt[near[:, 1].astype(int), near[:, 0].astype(int)]))
# #     return u.astype(np.float32), near, mask_width


# # def tip_point(end_mask, contact, u):
# #     """Tip ON the tube axis: the end region's farthest extent from the tubing."""
# #     ys, xs = np.nonzero(end_mask)
# #     pts = np.stack([xs, ys], 1).astype(np.float32)
# #     reach = float(np.max((contact - pts) @ u))
# #     return contact - u * max(reach, 1.0)


# # # ---------------------------------------------------------------- per image
# # def analyse(bgr, instances, mm_per_px=None, max_disagree=0.2, gap=3, class_map=None):
# #     """class_map = (names, role, tubing_id) from build_class_map(); defaults to the
# #     hand-label fallback. Pass model.names via build_class_map() when using --model,
# #     so class ids are matched by their real name, not by an assumed position."""
# #     names, role, tubing_id = class_map or (NAMES, ROLE, TUBING)
# #     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
# #     tubings = [m for c, m in instances if c == tubing_id]
# #     ends = []
# #     for cls, m in instances:
# #         if cls == tubing_id or cls not in role:
# #             continue   # tubing itself, or a class that is neither head- nor tail-like
# #         e = {"class": names.get(cls, str(cls)), "role": role[cls], "status": "OK"}
# #         ti, contact = attach(m, tubings, gap)
# #         if ti is None:
# #             e["status"] = "NO_TUBING_MASK"
# #             ends.append(e)
# #             continue
# #         tube = tubings[ti]
# #         area_w = np.sqrt(tube.sum() / max(skeletonize(tube).sum(), 1))   # rough width for radius
# #         u, near, mask_w = local_axis(tube, contact, radius=4 * area_w + 5)
# #         if u is None:
# #             e["status"] = "BAD_TUBING_MASK"
# #             ends.append(e)
# #             continue
# #         tip = tip_point(m, contact, u)
# #         start = contact + u * max(0.5 * mask_w, 2)             # a little into bare tubing
# #         edge_w, q = measure_width(gray, start - u * 10, start,
# #                                   length=int(max(3 * mask_w, 15)),
# #                                   search_px=int(0.5 * mask_w * 1.4) + 3,
# #                                   refine=True, max_angle_deg=6, max_shift=3)
# #         width = edge_w if edge_w else mask_w
# #         if edge_w is None:
# #             e["status"] = "CHECK (no clear walls, using mask width)"
# #         elif abs(edge_w - mask_w) / mask_w > max_disagree:
# #             e["status"] = "CHECK (edge and mask widths disagree)"
# #         trans = transparency_score(gray, tip, contact, width)
# #         x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
# #         e.update({
# #             "tubing_id": ti,
# #             "tip": [round(float(v), 1) for v in tip],
# #             "width_edge_px": round(edge_w, 2) if edge_w else None,
# #             "width_mask_px": round(mask_w, 2),
# #             "width_px": round(width, 2),
# #             "width_mm": round(width * mm_per_px, 2) if mm_per_px else None,
# #             "edge_quality": round(q, 2),
# #             "transparency": round(trans, 2) if trans is not None else None,
# #             "opaque": (trans < 0.5) if trans is not None else None,
# #             "colour": colour_score(bgr, (x, y, x + w, y + h)),
# #             "_axis": (contact, u, near),
# #         })
# #         ends.append(e)
# #     compare(ends)
# #     pair_ends(ends)
# #     return ends


# # def pair_ends(ends, max_width_diff=0.15):
# #     """Pair each HEAD to the TAIL it most likely belongs to, by width similarity.

# #     This is the practical stand-in for "tracing": actually following a tube's
# #     pixels through a coil is unreliable the moment two tubes touch, cross, or
# #     overlap in the image, which they do constantly here. A tube's own head and
# #     tail were cut from the same stock, so they carry the same diameter; two
# #     different tube sizes almost never measure alike by chance. Matching on
# #     width sidesteps the coil entirely and works in a single still frame.

# #     Mutual best match: a head and a tail are paired only if each is the other's
# #     closest width match, within max_width_diff (fraction of the smaller width).
# #     Ends with no acceptable match are left unpaired (pair_id stays None) rather
# #     than being forced together - a wrong forced pair is worse than no pair.
# #     """
# #     heads = [e for e in ends if e.get("role") == "HEAD" and e.get("width_px")]
# #     tails = [e for e in ends if e.get("role") == "TAIL" and e.get("width_px")]
# #     for e in ends:
# #         e["pair_id"] = None
# #     if not heads or not tails:
# #         return

# #     def closest(a, options):
# #         best, best_diff = None, None
# #         for b in options:
# #             diff = abs(a["width_px"] - b["width_px"]) / min(a["width_px"], b["width_px"])
# #             if diff <= max_width_diff and (best_diff is None or diff < best_diff):
# #                 best, best_diff = b, diff
# #         return best

# #     used_tails = set()
# #     pair_id = 0
# #     for h in sorted(heads, key=lambda e: e["width_px"]):
# #         available = [t for t in tails if id(t) not in used_tails]
# #         t = closest(h, available)
# #         if t is not None and closest(t, heads) is h:      # mutual best match only
# #             pair_id += 1
# #             h["pair_id"] = t["pair_id"] = pair_id
# #             used_tails.add(id(t))


# # def compare(ends, min_ratio=1.15):
# #     ok = [e for e in ends if e.get("width_px")]
# #     if len(ok) < 2:
# #         return
# #     ws = np.array([e["width_px"] for e in ok])
# #     order = np.sort(ws)
# #     ratios = order[1:] / order[:-1]
# #     if ratios.max() < min_ratio:
# #         for e in ok:
# #             e["size"] = "SAME"
# #         return
# #     cut = (order[ratios.argmax()] + order[ratios.argmax() + 1]) / 2
# #     for e in ok:
# #         e["size"] = "BIGGER" if e["width_px"] > cut else "SMALLER"


# # PALETTE = [(255, 120, 0), (255, 255, 255), (0, 200, 255), (0, 180, 0),
# #           (200, 120, 255), (120, 200, 255), (255, 200, 120)]


# # def draw(bgr, instances, ends, tubing_id=None):
# #     tubing_id = TUBING if tubing_id is None else tubing_id
# #     vis = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
# #     overlay = vis.copy()
# #     colours = {tubing_id: (0, 180, 0)}
# #     for c, m in instances:
# #         col = colours.setdefault(c, PALETTE[len(colours) % len(PALETTE)])
# #         overlay[m] = col
# #     vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
# #     for i, e in enumerate(ends):
# #         if "_axis" not in e:
# #             continue
# #         contact, u, near = e["_axis"]
# #         for p in near.astype(int):
# #             vis[p[1], p[0]] = (0, 0, 255)
# #         tip = tuple(int(v) for v in e["tip"])
# #         cv2.circle(vis, tip, 4, (0, 0, 255), -1)
# #         w = e["width_mm"] or e["width_px"]
# #         unit = "mm" if e["width_mm"] else "px"
# #         txt = f"#{i} {e['role']} {w:.1f}{unit} {e.get('size', '')}"
# #         if e["status"] != "OK":
# #             txt += " CHECK"
# #         cv2.putText(vis, txt, (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
# #                     0.45, (0, 0, 0), 3, cv2.LINE_AA)
# #         cv2.putText(vis, txt, (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
# #                     0.45, (0, 255, 255), 1, cv2.LINE_AA)
# #     return vis


# # def main():
# #     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
# #     ap.add_argument("--images", required=True)
# #     src = ap.add_mutually_exclusive_group(required=True)
# #     src.add_argument("--labels", help="folder of YOLO-seg .txt labels (CVAT export)")
# #     src.add_argument("--model", help="trained YOLO11-seg weights")
# #     ap.add_argument("--output", default="measured")
# #     ap.add_argument("--conf", type=float, default=0.35)
# #     ap.add_argument("--device", default=None,
# #                     help="0 for first GPU, cpu for CPU, blank = let Ultralytics decide")
# #     ap.add_argument("--tile", type=int, default=0,
# #                     help="run the model on overlapping tiles of this size (e.g. 640); 0 = whole photo")
# #     ap.add_argument("--tile-overlap", type=float, default=0.3)
# #     ap.add_argument("--no-batch-tiles", action="store_true",
# #                     help="send tiles to the model one at a time instead of as one batch")
# #     ap.add_argument("--no-gray", action="store_true",
# #                     help="feed the model colour images (use ONLY if it was trained on colour)")
# #     ap.add_argument("--mm-per-px", type=float, default=None)
# #     ap.add_argument("--max-disagree", type=float, default=0.2)
# #     ap.add_argument("--attach-gap", type=int, default=3,
# #                     help="px of gap tolerated between an end polygon and its tubing polygon")
# #     a = ap.parse_args()

# #     model = None
# #     class_map = None
# #     if a.model:
# #         from ultralytics import YOLO
# #         model = YOLO(a.model)
# #         if a.device is not None:
# #             model.to(f"cuda:{a.device}" if a.device.isdigit() else a.device)
# #         class_map = build_class_map(model.names)   # names come from THIS checkpoint, not a guess
# #         print(f"[classes] {model.names} -> tubing id {class_map[2]}, roles {class_map[1]}")
# #     out = Path(a.output)
# #     out.mkdir(parents=True, exist_ok=True)
# #     report = []
# #     for p in sorted(Path(a.images).iterdir()):
# #         if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
# #             continue
# #         bgr = cv2.imread(str(p))
# #         h, w = bgr.shape[:2]
# #         inst = (masks_from_model(model, bgr, a.conf, not a.no_gray, a.tile, a.tile_overlap,
# #                                  batch_tiles=not a.no_batch_tiles) if model
# #                 else masks_from_yolo_txt(Path(a.labels) / f"{p.stem}.txt", h, w))
# #         ends = analyse(bgr, inst, a.mm_per_px, a.max_disagree, a.attach_gap, class_map)
# #         tubing_id = class_map[2] if class_map else None
# #         cv2.imwrite(str(out / f"{p.stem}_measured.jpg"), draw(bgr, inst, ends, tubing_id))
# #         for e in ends:
# #             e.pop("_axis", None)
# #         report.append({"image": p.name, "ends": ends})
# #         print(p.name)
# #         for i, e in enumerate(ends):
# #             print(f"  #{i} {e['role']:4s} {e['class']:10s} width={e.get('width_px')}px "
# #                   f"(edge {e.get('width_edge_px')}, mask {e.get('width_mask_px')}) "
# #                   f"{'opaque' if e.get('opaque') else 'transparent'} {e.get('size', '')} [{e['status']}]")
# #     (out / "report.json").write_text(json.dumps(report, indent=2))
# #     print(f"[done] {out}")


# # if __name__ == "__main__":
# #     main()



# """
# measure_from_masks.py - head/tail, transparency and width from SEGMENTATION MASKS.

# Works on:
#   * hand-drawn masks exported from CVAT as "Ultralytics YOLO Segmentation"  (now, 4 photos)
#   * YOLO11-seg predictions                                             (later, --model)

# Classes: however many your model has, matched by NAME not by position:
#   any class whose name contains "head"   -> role HEAD
#   any class whose name contains "tail"   -> role TAIL
#   the class named exactly "tubing"       -> the tubing mask
# This matters because retraining can change how many classes exist and in what
# order (e.g. merging head_blue/head_black into one "head" class shifts every
# later index) - a hardcoded index like "tubing = class 3" silently breaks the
# moment the class count changes: every end then finds NO tubing mask and no
# width is ever computed. Matching by name survives that.

#   python measure_from_masks.py --images photos/ --labels cvat_export/labels/ --output measured/
#   python measure_from_masks.py --images photos/ --model best-seg.pt --output measured/
# """
# import argparse
# import json
# from pathlib import Path

# import cv2
# import numpy as np
# from skimage.morphology import skeletonize

# from tube_ends import measure_width, transparency_score, colour_score

# # Fallback class list, used ONLY for --labels (hand-drawn CVAT .txt export),
# # which carries no embedded names - it must match your data.yaml order exactly.
# # --model needs no such list: it reads names straight from the checkpoint.
# FALLBACK_NAMES = ["head_blue", "head_black", "tail_open", "tubing"]


# def build_class_map(names):
#     """names: {id: name} (from a YOLO model) or a list in id order (CVAT fallback).
#     Returns (names_by_id, role_by_id, tubing_id) built by matching NAME, not position."""
#     if isinstance(names, (list, tuple)):
#         names = {i: n for i, n in enumerate(names)}
#     role, tubing_id = {}, None
#     for cid, name in names.items():
#         n = name.lower()
#         if "tubing" == n or "tube" in n:
#             tubing_id = cid
#         elif "head" in n:
#             role[cid] = "HEAD"
#         elif "tail" in n:
#             role[cid] = "TAIL"
#     if tubing_id is None:
#         raise ValueError(f"no class named 'tubing' found among {names} - "
#                          "check the model's / data.yaml's class names")
#     return names, role, tubing_id


# # Fallback map for the --labels path (see FALLBACK_NAMES above).
# NAMES, ROLE, TUBING = build_class_map(FALLBACK_NAMES)


# # ---------------------------------------------------------------- inputs
# def masks_from_yolo_txt(txt, h, w):
#     """YOLO-seg label file -> list of (class_id, bool mask)."""
#     out = []
#     if not Path(txt).exists():
#         return out
#     for line in Path(txt).read_text().splitlines():
#         v = line.split()
#         if len(v) < 7:
#             continue
#         cls = int(v[0])
#         pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
#         m = np.zeros((h, w), np.uint8)
#         cv2.fillPoly(m, [pts], 1)
#         out.append((cls, m.astype(bool)))
#     return out


# def _predict(model, img, conf, gray):
#     inp = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else img
#     r = model(inp, conf=conf, retina_masks=True, verbose=False)[0]
#     if r.masks is None:
#         return []
#     ms = r.masks.data.cpu().numpy() > 0.5
#     cls = r.boxes.cls.cpu().tolist()
#     cfs = r.boxes.conf.cpu().tolist()
#     return list(zip([int(c) for c in cls], ms, cfs))


# def _predict_batch(model, imgs, conf, gray):
#     """Same as _predict but for a LIST of equally-sized crops in one model call -
#     the point of tiling on GPU: one batched forward pass instead of one per tile."""
#     if gray:
#         imgs = [cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for im in imgs]
#     results = model(imgs, conf=conf, retina_masks=True, verbose=False)
#     out = []
#     for r in results:
#         if r.masks is None:
#             out.append([])
#             continue
#         ms = r.masks.data.cpu().numpy() > 0.5
#         cls = r.boxes.cls.cpu().tolist()
#         cfs = r.boxes.conf.cpu().tolist()
#         out.append(list(zip([int(c) for c in cls], ms, cfs)))
#     return out


# def masks_from_model(model, bgr, conf, gray=True, tile=0, overlap=0.3, dedupe_iou=0.4, batch_tiles=True):
#     """Masks for the whole photo, or - with tile>0 - from overlapping tiles.

#     Tiling matters when the ends are small: a 640 px tile of a 1920 px photo
#     gives the model 3x the pixels on the same object. On GPU the tiles are sent
#     as ONE batched call (batch_tiles=True) so the speed hit from tiling is small;
#     on CPU a batch is processed sequentially anyway, so either mode works.
#     Duplicates from the overlap are removed by IoU, keeping the more confident one."""
#     if not tile:
#         return [(c, m) for c, m, _ in _predict(model, bgr, conf, gray)]
#     H, W = bgr.shape[:2]
#     step = max(int(tile * (1 - overlap)), 32)
#     xs = sorted({*range(0, max(W - tile, 0) + 1, step), max(W - tile, 0)})
#     ys = sorted({*range(0, max(H - tile, 0) + 1, step), max(H - tile, 0)})
#     origins = [(x, y) for y in ys for x in xs]
#     crops = [bgr[y:y + tile, x:x + tile] for x, y in origins]
#     # pad any short edge tile to a uniform size so they can be batched together
#     crops = [c if c.shape[:2] == (tile, tile) else
#              cv2.copyMakeBorder(c, 0, tile - c.shape[0], 0, tile - c.shape[1],
#                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
#              for c in crops]
#     per_tile = (_predict_batch(model, crops, conf, gray) if batch_tiles
#                 else [_predict(model, c, conf, gray) for c in crops])
#     dets = []
#     for (x, y), tile_dets in zip(origins, per_tile):
#         for c, m, cf in tile_dets:
#             full = np.zeros((H, W), bool)
#             mh, mw = m.shape
#             th, tw = min(mh, H - y), min(mw, W - x)
#             full[y:y + th, x:x + tw] = m[:th, :tw]
#             if full.any():
#                 dets.append((c, full, cf))
#     kept = []
#     for c, m, cf in sorted(dets, key=lambda d: -d[2]):
#         if any(kc == c and (km & m).sum() > dedupe_iou * min(km.sum(), m.sum())
#                for kc, km, _ in kept):
#             continue
#         kept.append((c, m, cf))
#     return [(c, m) for c, m, _ in kept]


# # ---------------------------------------------------------------- geometry
# def attach(end_mask, tubings, gap=3):
#     """Index of the tubing mask that touches this end (largest contact).
#     `gap` px of separation between the two polygons is tolerated."""
#     k = 2 * int(max(gap, 0)) + 1
#     grown = cv2.dilate(end_mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
#     best, best_n = None, 0
#     for i, t in enumerate(tubings):
#         n = int((grown & t).sum())
#         if n > best_n:
#             best, best_n = i, n
#     if best is None:
#         return None, None
#     ys, xs = np.nonzero(grown & tubings[best])
#     return best, np.array([xs.mean(), ys.mean()], np.float32)


# def local_axis(tube_mask, contact, radius):
#     """Unit direction of the tube near `contact`, pointing away from the end,
#     plus the skeleton points used and the mask width there."""
#     skel = skeletonize(tube_mask)
#     dt = cv2.distanceTransform(tube_mask.astype(np.uint8), cv2.DIST_L2, 5)
#     # drop the short corner branches a skeleton grows at polygon ends: they sit
#     # close to the border (small distance value) and would tilt the axis
#     skel &= dt >= 0.7 * dt[skel].max() if skel.any() else skel
#     ys, xs = np.nonzero(skel)
#     if len(xs) < 3:
#         return None, None, None
#     pts = np.stack([xs, ys], 1).astype(np.float32)
#     near = pts[np.linalg.norm(pts - contact, axis=1) < radius]
#     if len(near) < 3:
#         near = pts[np.argsort(np.linalg.norm(pts - contact, axis=1))[:10]]
#     c = near.mean(0)
#     _, _, vt = np.linalg.svd(near - c)
#     u = vt[0]
#     if np.dot(c - contact, u) < 0:        # point from the end into the tubing
#         u = -u
#     mask_width = 2.0 * float(np.median(dt[near[:, 1].astype(int), near[:, 0].astype(int)]))
#     return u.astype(np.float32), near, mask_width


# def tip_point(end_mask, contact, u):
#     """Tip ON the tube axis: the end region's farthest extent from the tubing."""
#     ys, xs = np.nonzero(end_mask)
#     pts = np.stack([xs, ys], 1).astype(np.float32)
#     reach = float(np.max((contact - pts) @ u))
#     return contact - u * max(reach, 1.0)


# # ---------------------------------------------------------------- per image
# def analyse(bgr, instances, mm_per_px=None, max_disagree=0.2, gap=3, class_map=None):
#     """class_map = (names, role, tubing_id) from build_class_map(); defaults to the
#     hand-label fallback. Pass model.names via build_class_map() when using --model,
#     so class ids are matched by their real name, not by an assumed position."""
#     names, role, tubing_id = class_map or (NAMES, ROLE, TUBING)
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     tubings = [m for c, m in instances if c == tubing_id]
#     ends = []
#     for cls, m in instances:
#         if cls == tubing_id or cls not in role:
#             continue   # tubing itself, or a class that is neither head- nor tail-like
#         x, y, bw, bh = cv2.boundingRect(m.astype(np.uint8))
#         e = {"class": names.get(cls, str(cls)), "role": role[cls], "status": "OK",
#              "bbox": [int(x), int(y), int(bw), int(bh)]}   # [x, y, width, height] in pixels
#         ti, contact = attach(m, tubings, gap)
#         if ti is None:
#             e["status"] = "NO_TUBING_MASK"
#             ends.append(e)
#             continue
#         tube = tubings[ti]
#         area_w = np.sqrt(tube.sum() / max(skeletonize(tube).sum(), 1))   # rough width for radius
#         u, near, mask_w = local_axis(tube, contact, radius=4 * area_w + 5)
#         if u is None:
#             e["status"] = "BAD_TUBING_MASK"
#             ends.append(e)
#             continue
#         tip = tip_point(m, contact, u)
#         start = contact + u * max(0.5 * mask_w, 2)             # a little into bare tubing
#         edge_w, q = measure_width(gray, start - u * 10, start,
#                                   length=int(max(3 * mask_w, 15)),
#                                   search_px=int(0.5 * mask_w * 1.4) + 3,
#                                   refine=True, max_angle_deg=6, max_shift=3)
#         width = edge_w if edge_w else mask_w
#         if edge_w is None:
#             e["status"] = "CHECK (no clear walls, using mask width)"
#         elif abs(edge_w - mask_w) / mask_w > max_disagree:
#             e["status"] = "CHECK (edge and mask widths disagree)"
#         trans = transparency_score(gray, tip, contact, width)
#         x, y, w, h = e["bbox"]
#         e.update({
#             "tubing_id": ti,
#             "tip": [round(float(v), 1) for v in tip],
#             "width_edge_px": round(edge_w, 2) if edge_w else None,
#             "width_mask_px": round(mask_w, 2),
#             "width_px": round(width, 2),
#             "width_mm": round(width * mm_per_px, 2) if mm_per_px else None,
#             "edge_quality": round(q, 2),
#             "transparency": round(trans, 2) if trans is not None else None,
#             "opaque": (trans < 0.5) if trans is not None else None,
#             "colour": colour_score(bgr, (x, y, x + w, y + h)),
#             "_axis": (contact, u, near),
#         })
#         ends.append(e)
#     compare(ends)
#     pair_ends(ends)
#     return ends


# def pair_ends(ends, max_width_diff=0.15):
#     """Pair each HEAD to the TAIL it most likely belongs to, by width similarity.

#     This is the practical stand-in for "tracing": actually following a tube's
#     pixels through a coil is unreliable the moment two tubes touch, cross, or
#     overlap in the image, which they do constantly here. A tube's own head and
#     tail were cut from the same stock, so they carry the same diameter; two
#     different tube sizes almost never measure alike by chance. Matching on
#     width sidesteps the coil entirely and works in a single still frame.

#     Mutual best match: a head and a tail are paired only if each is the other's
#     closest width match, within max_width_diff (fraction of the smaller width).
#     Ends with no acceptable match are left unpaired (pair_id stays None) rather
#     than being forced together - a wrong forced pair is worse than no pair.
#     """
#     heads = [e for e in ends if e.get("role") == "HEAD" and e.get("width_px")]
#     tails = [e for e in ends if e.get("role") == "TAIL" and e.get("width_px")]
#     for e in ends:
#         e["pair_id"] = None
#     if not heads or not tails:
#         return

#     def closest(a, options):
#         best, best_diff = None, None
#         for b in options:
#             diff = abs(a["width_px"] - b["width_px"]) / min(a["width_px"], b["width_px"])
#             if diff <= max_width_diff and (best_diff is None or diff < best_diff):
#                 best, best_diff = b, diff
#         return best

#     used_tails = set()
#     pair_id = 0
#     for h in sorted(heads, key=lambda e: e["width_px"]):
#         available = [t for t in tails if id(t) not in used_tails]
#         t = closest(h, available)
#         if t is not None and closest(t, heads) is h:      # mutual best match only
#             pair_id += 1
#             h["pair_id"] = t["pair_id"] = pair_id
#             used_tails.add(id(t))


# def compare(ends, min_ratio=1.15):
#     ok = [e for e in ends if e.get("width_px")]
#     if len(ok) < 2:
#         return
#     ws = np.array([e["width_px"] for e in ok])
#     order = np.sort(ws)
#     ratios = order[1:] / order[:-1]
#     if ratios.max() < min_ratio:
#         for e in ok:
#             e["size"] = "SAME"
#         return
#     cut = (order[ratios.argmax()] + order[ratios.argmax() + 1]) / 2
#     for e in ok:
#         e["size"] = "BIGGER" if e["width_px"] > cut else "SMALLER"


# PALETTE = [(255, 120, 0), (255, 255, 255), (0, 200, 255), (0, 180, 0),
#           (200, 120, 255), (120, 200, 255), (255, 200, 120)]


# def draw(bgr, instances, ends, tubing_id=None):
#     tubing_id = TUBING if tubing_id is None else tubing_id
#     vis = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
#     overlay = vis.copy()
#     colours = {tubing_id: (0, 180, 0)}
#     for c, m in instances:
#         col = colours.setdefault(c, PALETTE[len(colours) % len(PALETTE)])
#         overlay[m] = col
#     vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
#     for i, e in enumerate(ends):
#         if "_axis" not in e:
#             continue
#         contact, u, near = e["_axis"]
#         for p in near.astype(int):
#             vis[p[1], p[0]] = (0, 0, 255)
#         tip = tuple(int(v) for v in e["tip"])
#         cv2.circle(vis, tip, 4, (0, 0, 255), -1)
#         w = e["width_mm"] or e["width_px"]
#         unit = "mm" if e["width_mm"] else "px"
#         txt = f"#{i} {e['role']} {w:.1f}{unit} {e.get('size', '')}"
#         if e["status"] != "OK":
#             txt += " CHECK"
#         cv2.putText(vis, txt, (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.45, (0, 0, 0), 3, cv2.LINE_AA)
#         cv2.putText(vis, txt, (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.45, (0, 255, 255), 1, cv2.LINE_AA)
#     return vis


# def main():
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--images", required=True)
#     src = ap.add_mutually_exclusive_group(required=True)
#     src.add_argument("--labels", help="folder of YOLO-seg .txt labels (CVAT export)")
#     src.add_argument("--model", help="trained YOLO11-seg weights")
#     ap.add_argument("--output", default="measured")
#     ap.add_argument("--conf", type=float, default=0.35)
#     ap.add_argument("--device", default=None,
#                     help="0 for first GPU, cpu for CPU, blank = let Ultralytics decide")
#     ap.add_argument("--tile", type=int, default=0,
#                     help="run the model on overlapping tiles of this size (e.g. 640); 0 = whole photo")
#     ap.add_argument("--tile-overlap", type=float, default=0.3)
#     ap.add_argument("--no-batch-tiles", action="store_true",
#                     help="send tiles to the model one at a time instead of as one batch")
#     ap.add_argument("--no-gray", action="store_true",
#                     help="feed the model colour images (use ONLY if it was trained on colour)")
#     ap.add_argument("--mm-per-px", type=float, default=None)
#     ap.add_argument("--max-disagree", type=float, default=0.2)
#     ap.add_argument("--attach-gap", type=int, default=3,
#                     help="px of gap tolerated between an end polygon and its tubing polygon")
#     a = ap.parse_args()

#     model = None
#     class_map = None
#     if a.model:
#         from ultralytics import YOLO
#         model = YOLO(a.model)
#         if a.device is not None:
#             model.to(f"cuda:{a.device}" if a.device.isdigit() else a.device)
#         class_map = build_class_map(model.names)   # names come from THIS checkpoint, not a guess
#         print(f"[classes] {model.names} -> tubing id {class_map[2]}, roles {class_map[1]}")
#     out = Path(a.output)
#     out.mkdir(parents=True, exist_ok=True)
#     report = []
#     for p in sorted(Path(a.images).iterdir()):
#         if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
#             continue
#         bgr = cv2.imread(str(p))
#         h, w = bgr.shape[:2]
#         inst = (masks_from_model(model, bgr, a.conf, not a.no_gray, a.tile, a.tile_overlap,
#                                  batch_tiles=not a.no_batch_tiles) if model
#                 else masks_from_yolo_txt(Path(a.labels) / f"{p.stem}.txt", h, w))
#         ends = analyse(bgr, inst, a.mm_per_px, a.max_disagree, a.attach_gap, class_map)
#         tubing_id = class_map[2] if class_map else None
#         cv2.imwrite(str(out / f"{p.stem}_measured.jpg"), draw(bgr, inst, ends, tubing_id))
#         for e in ends:
#             e.pop("_axis", None)
#         report.append({"image": p.name, "ends": ends})
#         print(p.name)
#         for i, e in enumerate(ends):
#             print(f"  #{i} {e['role']:4s} {e['class']:10s} width={e.get('width_px')}px "
#                   f"(edge {e.get('width_edge_px')}, mask {e.get('width_mask_px')}) "
#                   f"{'opaque' if e.get('opaque') else 'transparent'} {e.get('size', '')} [{e['status']}]")
#     (out / "report.json").write_text(json.dumps(report, indent=2))
#     print(f"[done] {out}")


# if __name__
# 
#  == "__main__":
#     main()




"""
measure_from_masks.py - head/tail, transparency and width from SEGMENTATION MASKS.

Works on:
  * hand-drawn masks exported from CVAT as "Ultralytics YOLO Segmentation"  (now, 4 photos)
  * YOLO11-seg predictions                                             (later, --model)

Classes: however many your model has, matched by NAME not by position:
  any class whose name contains "head"   -> role HEAD
  any class whose name contains "tail"   -> role TAIL
  the class named exactly "tubing"       -> the tubing mask
This matters because retraining can change how many classes exist and in what
order (e.g. merging head_blue/head_black into one "head" class shifts every
later index) - a hardcoded index like "tubing = class 3" silently breaks the
moment the class count changes: every end then finds NO tubing mask and no
width is ever computed. Matching by name survives that.

  python measure_from_masks.py --images photos/ --labels cvat_export/labels/ --output measured/
  python measure_from_masks.py --images photos/ --model best-seg.pt --output measured/
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from tube_ends import measure_width, transparency_score, colour_score

# Fallback class list, used ONLY for --labels (hand-drawn CVAT .txt export),
# which carries no embedded names - it must match your data.yaml order exactly.
# --model needs no such list: it reads names straight from the checkpoint.
FALLBACK_NAMES = ["head_blue", "head_black", "tail_open", "tubing"]


def build_class_map(names):
    """names: {id: name} (from a YOLO model) or a list in id order (CVAT fallback).
    Returns (names_by_id, role_by_id, tubing_id) built by matching NAME, not position."""
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    role, tubing_id = {}, None
    for cid, name in names.items():
        n = name.lower()
        if "tubing" == n or "tube" in n:
            tubing_id = cid
        elif "head" in n:
            role[cid] = "HEAD"
        elif "tail" in n:
            role[cid] = "TAIL"
    if tubing_id is None:
        raise ValueError(f"no class named 'tubing' found among {names} - "
                         "check the model's / data.yaml's class names")
    return names, role, tubing_id


# Fallback map for the --labels path (see FALLBACK_NAMES above).
NAMES, ROLE, TUBING = build_class_map(FALLBACK_NAMES)


# ---------------------------------------------------------------- inputs
def masks_from_yolo_txt(txt, h, w):
    """YOLO-seg label file -> list of (class_id, bool mask)."""
    out = []
    if not Path(txt).exists():
        return out
    for line in Path(txt).read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        cls = int(v[0])
        pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        out.append((cls, m.astype(bool)))
    return out


def _predict(model, img, conf, gray):
    inp = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else img
    r = model(inp, conf=conf, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    ms = r.masks.data.cpu().numpy() > 0.5
    cls = r.boxes.cls.cpu().tolist()
    cfs = r.boxes.conf.cpu().tolist()
    return list(zip([int(c) for c in cls], ms, cfs))


def _predict_batch(model, imgs, conf, gray):
    """Same as _predict but for a LIST of equally-sized crops in one model call -
    the point of tiling on GPU: one batched forward pass instead of one per tile."""
    if gray:
        imgs = [cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for im in imgs]
    results = model(imgs, conf=conf, retina_masks=True, verbose=False)
    out = []
    for r in results:
        if r.masks is None:
            out.append([])
            continue
        ms = r.masks.data.cpu().numpy() > 0.5
        cls = r.boxes.cls.cpu().tolist()
        cfs = r.boxes.conf.cpu().tolist()
        out.append(list(zip([int(c) for c in cls], ms, cfs)))
    return out


def masks_from_model(model, bgr, conf, gray=True, tile=0, overlap=0.3, dedupe_iou=0.4,
                     dedupe_center_frac=0.6, batch_tiles=True, with_conf=False):
    """Masks for the whole photo, or - with tile>0 - from overlapping tiles.

    Tiling matters when the ends are small: a 640 px tile of a 1920 px photo
    gives the model 3x the pixels on the same object. On GPU the tiles are sent
    as ONE batched call (batch_tiles=True) so the speed hit from tiling is small;
    on CPU a batch is processed sequentially anyway, so either mode works.

    Duplicates from the tile overlap are removed two ways:
      1. mask overlap ratio > dedupe_iou (the classic case: same object, same
         shape, just detected twice)
      2. box-CENTER distance < dedupe_center_frac of the boxes' diagonal (the
         tile-SEAM case: the same real object gets a slightly different
         segmentation boundary in each tile crop, so its mask shape and even
         its size can differ enough that overlap alone misses it - but the
         two boxes are still clearly centred on the same object). Without
         this second check, an object sitting near a tile seam is easily
         double-counted, inflating HEADS/TAILS counts on real video.
    In both cases the higher-confidence detection is kept.

    with_conf=True returns (class_id, mask, confidence) triples instead of
    (class_id, mask) pairs - the model's own confidence for that detection,
    kept all the way to the JSON output so a real CONF cutoff can be picked
    from actual numbers instead of guessed."""
    if not tile:
        raw = _predict(model, bgr, conf, gray)
        return [(c, m, cf) for c, m, cf in raw] if with_conf else [(c, m) for c, m, _ in raw]
    H, W = bgr.shape[:2]
    step = max(int(tile * (1 - overlap)), 32)
    xs = sorted({*range(0, max(W - tile, 0) + 1, step), max(W - tile, 0)})
    ys = sorted({*range(0, max(H - tile, 0) + 1, step), max(H - tile, 0)})
    origins = [(x, y) for y in ys for x in xs]
    crops = [bgr[y:y + tile, x:x + tile] for x, y in origins]
    # pad any short edge tile to a uniform size so they can be batched together
    crops = [c if c.shape[:2] == (tile, tile) else
             cv2.copyMakeBorder(c, 0, tile - c.shape[0], 0, tile - c.shape[1],
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
             for c in crops]
    per_tile = (_predict_batch(model, crops, conf, gray) if batch_tiles
                else [_predict(model, c, conf, gray) for c in crops])
    dets = []
    for (x, y), tile_dets in zip(origins, per_tile):
        for c, m, cf in tile_dets:
            full = np.zeros((H, W), bool)
            mh, mw = m.shape
            th, tw = min(mh, H - y), min(mw, W - x)
            full[y:y + th, x:x + tw] = m[:th, :tw]
            if full.any():
                bx, by, bw, bh = cv2.boundingRect(full.astype(np.uint8))
                centre = (bx + bw / 2, by + bh / 2)
                diag = float(np.hypot(bw, bh))
                dets.append((c, full, cf, centre, diag))
    kept = []
    for c, m, cf, centre, diag in sorted(dets, key=lambda d: -d[2]):
        def is_dup(kc, km, kcentre, kdiag):
            if kc != c:
                return False
            overlap_ratio = (km & m).sum() / min(km.sum(), m.sum())
            if overlap_ratio > dedupe_iou:
                return True
            dist = float(np.hypot(centre[0] - kcentre[0], centre[1] - kcentre[1]))
            return dist < dedupe_center_frac * max(diag, kdiag)

        if any(is_dup(kc, km, kcentre, kdiag) for kc, km, _, kcentre, kdiag in kept):
            continue
        kept.append((c, m, cf, centre, diag))
    if with_conf:
        return [(c, m, cf) for c, m, cf, _, _ in kept]
    return [(c, m) for c, m, _, _, _ in kept]


# ---------------------------------------------------------------- geometry
def attach(end_mask, tubings, gap=3, max_gap=None):
    """Index of the tubing mask that touches this end (largest contact).
    `gap` px of separation between the two polygons is tolerated.

    Segmentation masks are jagged frame to frame - the SAME real end that
    attaches fine one frame can end up a few px short of its tubing mask the
    next, purely from mask-boundary noise. Giving up at the first miss turns
    a real, working end into a flickering NO_TUBING_MASK. So if nothing
    touches within `gap`, the search retries once with a wider gap (capped
    at `max_gap`, default 4x gap or gap+8, whichever is larger) before
    truly giving up - close misses get rescued, but a detection that is
    genuinely nowhere near any tubing still correctly returns None."""
    if max_gap is None:
        max_gap = max(gap * 4, gap + 8)
    g = max(gap, 0)
    while True:
        k = 2 * int(g) + 1
        grown = cv2.dilate(end_mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        best, best_n = None, 0
        for i, t in enumerate(tubings):
            n = int((grown & t).sum())
            if n > best_n:
                best, best_n = i, n
        if best is not None:
            ys, xs = np.nonzero(grown & tubings[best])
            return best, np.array([xs.mean(), ys.mean()], np.float32)
        if g >= max_gap:
            return None, None
        g = min(g * 2, max_gap)


def local_axis(tube_mask, contact, radius):
    """Unit direction of the tube near `contact`, pointing away from the end,
    plus the skeleton points used and the mask width there."""
    skel = skeletonize(tube_mask)
    dt = cv2.distanceTransform(tube_mask.astype(np.uint8), cv2.DIST_L2, 5)
    # drop the short corner branches a skeleton grows at polygon ends: they sit
    # close to the border (small distance value) and would tilt the axis
    skel &= dt >= 0.7 * dt[skel].max() if skel.any() else skel
    ys, xs = np.nonzero(skel)
    if len(xs) < 3:
        return None, None, None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    near = pts[np.linalg.norm(pts - contact, axis=1) < radius]
    if len(near) < 3:
        near = pts[np.argsort(np.linalg.norm(pts - contact, axis=1))[:10]]
    c = near.mean(0)
    _, _, vt = np.linalg.svd(near - c)
    u = vt[0]
    if np.dot(c - contact, u) < 0:        # point from the end into the tubing
        u = -u
    mask_width = 2.0 * float(np.median(dt[near[:, 1].astype(int), near[:, 0].astype(int)]))
    return u.astype(np.float32), near, mask_width


def tip_point(end_mask, contact, u):
    """Tip ON the tube axis: the end region's farthest extent from the tubing."""
    ys, xs = np.nonzero(end_mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    reach = float(np.max((contact - pts) @ u))
    return contact - u * max(reach, 1.0)


# ---------------------------------------------------------------- per image
def analyse(bgr, instances, mm_per_px=None, max_disagree=0.2, gap=3, class_map=None):
    """class_map = (names, role, tubing_id) from build_class_map(); defaults to the
    hand-label fallback. Pass model.names via build_class_map() when using --model,
    so class ids are matched by their real name, not by an assumed position.

    `instances` accepts either (class_id, mask) pairs (the default from
    masks_from_model) or (class_id, mask, confidence) triples (from
    masks_from_model(..., with_conf=True)) - confidence, when present, is
    carried onto each end as e["confidence"] so a real threshold can be
    picked from actual numbers instead of guessed."""
    names, role, tubing_id = class_map or (NAMES, ROLE, TUBING)
    norm = [(it[0], it[1], (it[2] if len(it) > 2 else None)) for it in instances]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tubings = [m for c, m, _ in norm if c == tubing_id]
    ends = []
    for cls, m, det_conf in norm:
        if cls == tubing_id or cls not in role:
            continue   # tubing itself, or a class that is neither head- nor tail-like
        x, y, bw, bh = cv2.boundingRect(m.astype(np.uint8))
        e = {"class": names.get(cls, str(cls)), "role": role[cls], "status": "OK",
             "bbox": [int(x), int(y), int(bw), int(bh)],   # [x, y, width, height] in pixels
             "confidence": round(float(det_conf), 3) if det_conf is not None else None}
        ti, contact = attach(m, tubings, gap)
        if ti is None:
            e["status"] = "NO_TUBING_MASK"
            ends.append(e)
            continue
        tube = tubings[ti]
        area_w = np.sqrt(tube.sum() / max(skeletonize(tube).sum(), 1))   # rough width for radius
        u, near, mask_w = local_axis(tube, contact, radius=4 * area_w + 5)
        if u is None:
            e["status"] = "BAD_TUBING_MASK"
            ends.append(e)
            continue
        tip = tip_point(m, contact, u)
        start = contact + u * max(0.5 * mask_w, 2)             # a little into bare tubing
        edge_w, q = measure_width(gray, start - u * 10, start,
                                  length=int(max(3 * mask_w, 15)),
                                  search_px=int(0.5 * mask_w * 1.4) + 3,
                                  refine=True, max_angle_deg=6, max_shift=3)
        width = edge_w if edge_w else mask_w
        if edge_w is None:
            e["status"] = "CHECK (no clear walls, using mask width)"
        elif abs(edge_w - mask_w) / mask_w > max_disagree:
            e["status"] = "CHECK (edge and mask widths disagree)"
        trans = transparency_score(gray, tip, contact, width)
        x, y, w, h = e["bbox"]
        e.update({
            "tubing_id": ti,
            "tip": [round(float(v), 1) for v in tip],
            "width_edge_px": round(edge_w, 2) if edge_w else None,
            "width_mask_px": round(mask_w, 2),
            "width_px": round(width, 2),
            "width_mm": round(width * mm_per_px, 2) if mm_per_px else None,
            "edge_quality": round(q, 2),
            "transparency": round(trans, 2) if trans is not None else None,
            "opaque": (trans < 0.5) if trans is not None else None,
            "colour": colour_score(bgr, (x, y, x + w, y + h)),
            "_axis": (contact, u, near),
        })
        ends.append(e)
    compare(ends)
    pair_ends(ends)
    return ends


def pair_ends(ends, max_width_diff=0.15):
    """Pair each HEAD to the TAIL it most likely belongs to, by width similarity.

    This is the practical stand-in for "tracing": actually following a tube's
    pixels through a coil is unreliable the moment two tubes touch, cross, or
    overlap in the image, which they do constantly here. A tube's own head and
    tail were cut from the same stock, so they carry the same diameter; two
    different tube sizes almost never measure alike by chance. Matching on
    width sidesteps the coil entirely and works in a single still frame.

    Mutual best match: a head and a tail are paired only if each is the other's
    closest width match, within max_width_diff (fraction of the smaller width).
    Ends with no acceptable match are left unpaired (pair_id stays None) rather
    than being forced together - a wrong forced pair is worse than no pair.
    """
    heads = [e for e in ends if e.get("role") == "HEAD" and e.get("width_px")]
    tails = [e for e in ends if e.get("role") == "TAIL" and e.get("width_px")]
    for e in ends:
        e["pair_id"] = None
    if not heads or not tails:
        return

    def closest(a, options):
        best, best_diff = None, None
        for b in options:
            diff = abs(a["width_px"] - b["width_px"]) / min(a["width_px"], b["width_px"])
            if diff <= max_width_diff and (best_diff is None or diff < best_diff):
                best, best_diff = b, diff
        return best

    used_tails, used_heads = set(), set()
    pair_id = 0
    for h in sorted(heads, key=lambda e: e["width_px"]):
        available_tails = [t for t in tails if id(t) not in used_tails]
        t = closest(h, available_tails)
        if t is None:
            continue
        # Mutual best match only - but a head that already has a pair must not
        # be allowed to "win" a tail away from the head that's actually still
        # looking for one, or a valid remaining pair gets silently rejected.
        available_heads = [x for x in heads if id(x) not in used_heads]
        if closest(t, available_heads) is h:
            pair_id += 1
            h["pair_id"] = t["pair_id"] = pair_id
            used_tails.add(id(t))
            used_heads.add(id(h))


def compare(ends, min_ratio=1.15):
    ok = [e for e in ends if e.get("width_px")]
    if len(ok) < 2:
        return
    ws = np.array([e["width_px"] for e in ok])
    order = np.sort(ws)
    ratios = order[1:] / order[:-1]
    if ratios.max() < min_ratio:
        for e in ok:
            e["size"] = "SAME"
        return
    cut = (order[ratios.argmax()] + order[ratios.argmax() + 1]) / 2
    for e in ok:
        e["size"] = "BIGGER" if e["width_px"] > cut else "SMALLER"


PALETTE = [(255, 120, 0), (255, 255, 255), (0, 200, 255), (0, 180, 0),
          (200, 120, 255), (120, 200, 255), (255, 200, 120)]


def draw(bgr, instances, ends, tubing_id=None):
    tubing_id = TUBING if tubing_id is None else tubing_id
    vis = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    colours = {tubing_id: (0, 180, 0)}
    for item in instances:
        c, m = item[0], item[1]      # accepts (c, m) or (c, m, confidence) alike
        col = colours.setdefault(c, PALETTE[len(colours) % len(PALETTE)])
        overlay[m] = col
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    for i, e in enumerate(ends):
        if "_axis" not in e:
            continue
        contact, u, near = e["_axis"]
        for p in near.astype(int):
            vis[p[1], p[0]] = (0, 0, 255)
        tip = tuple(int(v) for v in e["tip"])
        cv2.circle(vis, tip, 4, (0, 0, 255), -1)
        w = e["width_mm"] or e["width_px"]
        unit = "mm" if e["width_mm"] else "px"
        txt = f"#{i} {e['role']} {w:.1f}{unit} {e.get('size', '')}"
        if e["status"] != "OK":
            txt += " CHECK"
        cv2.putText(vis, txt, (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="folder of YOLO-seg .txt labels (CVAT export)")
    src.add_argument("--model", help="trained YOLO11-seg weights")
    ap.add_argument("--output", default="measured")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None,
                    help="0 for first GPU, cpu for CPU, blank = let Ultralytics decide")
    ap.add_argument("--tile", type=int, default=0,
                    help="run the model on overlapping tiles of this size (e.g. 640); 0 = whole photo")
    ap.add_argument("--tile-overlap", type=float, default=0.3)
    ap.add_argument("--no-batch-tiles", action="store_true",
                    help="send tiles to the model one at a time instead of as one batch")
    ap.add_argument("--no-gray", action="store_true",
                    help="feed the model colour images (use ONLY if it was trained on colour)")
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--max-disagree", type=float, default=0.2)
    ap.add_argument("--attach-gap", type=int, default=3,
                    help="px of gap tolerated between an end polygon and its tubing polygon")
    a = ap.parse_args()

    model = None
    class_map = None
    if a.model:
        from ultralytics import YOLO
        model = YOLO(a.model)
        if a.device is not None:
            model.to(f"cuda:{a.device}" if a.device.isdigit() else a.device)
        class_map = build_class_map(model.names)   # names come from THIS checkpoint, not a guess
        print(f"[classes] {model.names} -> tubing id {class_map[2]}, roles {class_map[1]}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for p in sorted(Path(a.images).iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        inst = (masks_from_model(model, bgr, a.conf, not a.no_gray, a.tile, a.tile_overlap,
                                 batch_tiles=not a.no_batch_tiles) if model
                else masks_from_yolo_txt(Path(a.labels) / f"{p.stem}.txt", h, w))
        ends = analyse(bgr, inst, a.mm_per_px, a.max_disagree, a.attach_gap, class_map)
        tubing_id = class_map[2] if class_map else None
        cv2.imwrite(str(out / f"{p.stem}_measured.jpg"), draw(bgr, inst, ends, tubing_id))
        for e in ends:
            e.pop("_axis", None)
        report.append({"image": p.name, "ends": ends})
        print(p.name)
        for i, e in enumerate(ends):
            print(f"  #{i} {e['role']:4s} {e['class']:10s} width={e.get('width_px')}px "
                  f"(edge {e.get('width_edge_px')}, mask {e.get('width_mask_px')}) "
                  f"{'opaque' if e.get('opaque') else 'transparent'} {e.get('size', '')} [{e['status']}]")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()