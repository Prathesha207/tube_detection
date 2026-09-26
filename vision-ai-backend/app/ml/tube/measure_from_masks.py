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
                     dedupe_center_frac=0.6, batch_tiles=True):
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
    In both cases the higher-confidence detection is kept."""
    if not tile:
        return [(c, m) for c, m, _ in _predict(model, bgr, conf, gray)]
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
                centre = (bx + bw / 2.0, by + bh / 2.0)
                diag = float(np.hypot(bw, bh))
                dets.append((c, full, cf, centre, diag, (bx, by, bw, bh)))
    kept = []
    for c, m, cf, centre, diag, b in sorted(dets, key=lambda d: -d[2]):
        def is_dup(kc, km, kcentre, kdiag, kb):
            if kc != c:
                return False
            inter = (km & m).sum()
            if inter > 0:
                union = (km | m).sum()
                if union > 0 and (inter / union) > dedupe_iou:
                    return True
                if (inter / min(km.sum(), m.sum())) > 0.40:
                    return True
            # Box IoU check for tile seam boundary splits
            bx1, by1, bw1, bh1 = b
            bx2, by2, bw2, bh2 = kb
            ix1, iy1 = max(bx1, bx2), max(by1, by2)
            ix2, iy2 = min(bx1 + bw1, bx2 + bw2), min(by1 + bh1, by2 + bh2)
            if ix2 > ix1 and iy2 > iy1:
                b_inter = (ix2 - ix1) * (iy2 - iy1)
                b_union = bw1 * bh1 + bw2 * bh2 - b_inter
                if b_union > 0 and (b_inter / b_union) > 0.45:
                    return True
            return False

        if any(is_dup(kc, km, kcentre, kdiag, kb) for kc, km, _, kcentre, kdiag, kb in kept):
            continue
        kept.append((c, m, cf, centre, diag, b))
    return [(c, m) for c, m, _, _, _, _ in kept]


# ---------------------------------------------------------------- geometry
def attach(end_mask, tubings, gap=25):
    """Index of the tubing mask that touches or is closest to this end.
    `gap` px of separation between the two polygons is tolerated.

    Performance: all dilation and DT work is done on a small padded crop
    around the end bounding box rather than the full 1920x1080 frame.
    For a 150x150 end with gap=25 the crop is ~200x200 px vs 2 MP — ~45x
    faster on the critical dilation call.
    """
    if not tubings:
        return None, None

    H, W = end_mask.shape
    k  = 2 * int(max(gap, 0)) + 1
    pad = int(max(gap, 0)) + 4   # extra margin so grown mask is fully inside crop

    # Bounding box of the end mask
    ex, ey, ew, eh = cv2.boundingRect(end_mask.astype(np.uint8))
    x0 = max(ex - pad, 0);  y0 = max(ey - pad, 0)
    x1 = min(ex + ew + pad, W); y1 = min(ey + eh + pad, H)

    end_crop = end_mask[y0:y1, x0:x1]
    grown_crop = cv2.dilate(end_crop.astype(np.uint8),
                             np.ones((k, k), np.uint8)).astype(bool)

    best, best_n = None, 0
    for i, t in enumerate(tubings):
        t_crop = t[y0:y1, x0:x1]
        n = int((grown_crop & t_crop).sum())
        if n > best_n:
            best, best_n = i, n

    if best is not None and best_n > 0:
        t_crop = tubings[best][y0:y1, x0:x1]
        ys_c, xs_c = np.nonzero(grown_crop & t_crop)
        if len(xs_c) > 0:
            return best, np.array([xs_c.mean() + x0, ys_c.mean() + y0], np.float32)

    # Distance transform fallback — still in the local crop (much faster)
    # Expand the crop a little more so the DT gradient is meaningful
    pad2 = int(35) + pad
    x0b = max(ex - pad2, 0);  y0b = max(ey - pad2, 0)
    x1b = min(ex + ew + pad2, W); y1b = min(ey + eh + pad2, H)
    end_crop_b = end_mask[y0b:y1b, x0b:x1b]

    best_i = None
    min_d  = 999999.0
    for i, t in enumerate(tubings):
        if not t.any():
            continue
        t_crop_b = t[y0b:y1b, x0b:x1b]
        if not t_crop_b.any():
            continue
        dt = cv2.distanceTransform((~t_crop_b).astype(np.uint8), cv2.DIST_L2, 5)
        if not end_crop_b.any():
            continue
        d = float(dt[end_crop_b].min())
        if d < min_d:
            min_d = d
            best_i = i

    if best_i is not None and min_d <= 35.0:
        t_ys, t_xs = np.nonzero(tubings[best_i][y0b:y1b, x0b:x1b])
        e_ys, e_xs = np.nonzero(end_crop_b)
        if len(e_xs) == 0 or len(t_xs) == 0:
            return None, None
        e_c = np.array([e_xs.mean(), e_ys.mean()])
        dists = np.hypot(t_xs - e_c[0], t_ys - e_c[1])
        min_idx = np.argmin(dists)
        return best_i, np.array([t_xs[min_idx] + x0b, t_ys[min_idx] + y0b], np.float32)

    return None, None




def local_axis(tube_mask, contact, radius, tube_dt=None):
    """Unit direction of the tube near `contact`, via skeletonize (accurate, slower).
    Used by the photo-analysis CLI where accuracy > speed.
    For real-time video inference use local_axis_fast() instead."""
    H, W = tube_mask.shape
    cx, cy = int(round(float(contact[0]))), int(round(float(contact[1])))
    pad = int(radius) + 64
    x0, y0 = max(cx - pad, 0), max(cy - pad, 0)
    x1, y1 = min(cx + pad, W),  min(cy + pad, H)
    crop = tube_mask[y0:y1, x0:x1]
    if not crop.any():
        return None, None, None

    skel_crop = skeletonize(crop)
    if tube_dt is not None:
        dt_crop = tube_dt[y0:y1, x0:x1].copy()
    else:
        dt_crop = cv2.distanceTransform(crop.astype(np.uint8), cv2.DIST_L2, 5)

    if skel_crop.any():
        skel_crop &= dt_crop >= 0.7 * dt_crop[skel_crop].max()

    ys_c, xs_c = np.nonzero(skel_crop)
    if len(xs_c) < 3:
        return None, None, None

    xs_g = xs_c + x0
    ys_g = ys_c + y0
    pts  = np.stack([xs_g, ys_g], 1).astype(np.float32)

    near = pts[np.linalg.norm(pts - contact, axis=1) < radius]
    if len(near) < 3:
        near = pts[np.argsort(np.linalg.norm(pts - contact, axis=1))[:10]]
    c = near.mean(0)
    _, _, vt = np.linalg.svd(near - c)
    u = vt[0]
    if np.dot(c - contact, u) < 0:
        u = -u

    lx = (near[:, 0] - x0).astype(int).clip(0, dt_crop.shape[1] - 1)
    ly = (near[:, 1] - y0).astype(int).clip(0, dt_crop.shape[0] - 1)
    mask_width = 2.0 * float(np.median(dt_crop[ly, lx]))
    return u.astype(np.float32), near, mask_width


def local_axis_fast(tube_mask, tube_dt, contact, radius):
    """Fast tube axis direction using DT-weighted PCA - NO skeletonize.

    Skeletonize (Zhang-Suen) takes 50-200ms per coil crop even at 250x250px.
    Instead, weight all tubing pixels by their DT value so high-DT pixels
    (near the centerline) dominate the PCA. The first principal component of
    this weighted point cloud equals the tube axis. Typically <2ms total.

    Returns the same (u, near, mask_width) tuple as local_axis().
    """
    H, W = tube_mask.shape
    cx, cy = int(round(float(contact[0]))), int(round(float(contact[1])))
    pad = int(radius) + 48
    x0, y0 = max(cx - pad, 0), max(cy - pad, 0)
    x1, y1 = min(cx + pad, W),  min(cy + pad, H)

    crop_mask = tube_mask[y0:y1, x0:x1]
    crop_dt   = tube_dt[y0:y1, x0:x1]

    if not crop_mask.any():
        return None, None, None

    ys_c, xs_c = np.nonzero(crop_mask)
    if len(xs_c) < 3:
        return None, None, None

    xs_g = (xs_c + x0).astype(np.float32)
    ys_g = (ys_c + y0).astype(np.float32)
    pts     = np.stack([xs_g, ys_g], 1)
    weights = crop_dt[ys_c, xs_c].astype(np.float32)

    # Keep only pixels within radius of contact
    dists = np.linalg.norm(pts - contact, axis=1)
    sel   = dists < radius
    if sel.sum() < 3:
        sel = np.zeros(len(dists), bool)
        sel[np.argsort(dists)[:20]] = True

    pts_n = pts[sel]
    w_n   = weights[sel]
    w_sum = w_n.sum()
    if len(pts_n) < 3 or w_sum < 1e-6:
        return None, None, None

    # Weighted PCA — eigh faster than SVD for symmetric 2x2
    centroid = (pts_n * w_n[:, None]).sum(0) / w_sum
    centered = pts_n - centroid
    cov = (centered * w_n[:, None]).T @ centered / w_sum
    _, vecs = np.linalg.eigh(cov)   # ascending order; last = largest eigenvector = axis
    u = vecs[:, -1].astype(np.float32)
    if np.dot(centroid - contact, u) < 0:
        u = -u

    # Width: 2x DT at the contact pixel
    cy_cl = max(0, min(cy, H - 1))
    cx_cl = max(0, min(cx, W - 1))
    mask_width = 2.0 * max(float(tube_dt[cy_cl, cx_cl]), 1.0)

    # Highest-DT pixels near contact for visualisation
    top_k = min(10, len(pts_n))
    near  = pts_n[np.argsort(w_n)[-top_k:]]
    return u, near, mask_width


def tip_point(end_mask, contact, u):
    """Tip ON the tube axis: the end region's farthest extent from the tubing."""
    ys, xs = np.nonzero(end_mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    reach = float(np.max((contact - pts) @ u))
    return contact - u * max(reach, 1.0)


# ---------------------------------------------------------------- per image
def analyse(bgr, instances, mm_per_px=None, max_disagree=0.2, gap=25, class_map=None):
    """class_map = (names, role, tubing_id) from build_class_map(); defaults to the
    hand-label fallback. Pass model.names via build_class_map() when using --model,
    so class ids are matched by their real name, not by an assumed position."""
    names, role, tubing_id = class_map or (NAMES, ROLE, TUBING)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tubings = [m for c, m in instances if c == tubing_id]
    # Cache per-tubing distance transform (computed at most once per tube per frame)
    # area_w = DT max = inscribed-circle radius = actual tube cross-section radius (~30-50 px).
    # This keeps the local_axis crop small (radius~130 px) instead of using the coil
    # bbox which can be 600+ px and defeats the whole purpose of cropping.
    tube_dt_cache = {}   # tubing index -> (dt_array, area_w)
    def get_tube_dt(ti):
        if ti not in tube_dt_cache:
            dt = cv2.distanceTransform(tubings[ti].astype(np.uint8), cv2.DIST_L2, 5)
            tube_dt_cache[ti] = (dt, max(float(dt.max()), 4.0))
        return tube_dt_cache[ti]
    ends = []
    for cls, m in instances:
        if cls == tubing_id or cls not in role:
            continue   # tubing itself, or a class that is neither head- nor tail-like
        x, y, bw, bh = cv2.boundingRect(m.astype(np.uint8))
        # Crop to bounding box for nonzero scan and self DT — same result, ~88x fewer pixels
        m_crop = m[y:y + bh, x:x + bw]
        ys_c, xs_c = np.nonzero(m_crop)
        center_x = float(xs_c.mean() + x) if len(xs_c) > 0 else float(x + bw / 2.0)
        center_y = float(ys_c.mean() + y) if len(ys_c) > 0 else float(y + bh / 2.0)

        # Self mask width from bbox-cropped DT (max is identical to full-frame DT max)
        dt_self_crop = cv2.distanceTransform(m_crop.astype(np.uint8), cv2.DIST_L2, 5)
        self_mask_w = 2.0 * float(dt_self_crop.max()) if dt_self_crop.any() else float(min(bw, bh))

        e = {
            "class": names.get(cls, str(cls)),
            "role": role[cls],
            "status": "OK",
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "bbox_width": round(float(bw), 2),
            "bbox_height": round(float(bh), 2),
            "tip": [round(center_x, 1), round(center_y, 1)],
            "width_mask_px": round(self_mask_w, 2),
            "width_px": round(self_mask_w, 2),
            "final_width": round(self_mask_w, 2),
            "width_edge_px": None,
            "width_mm": round(self_mask_w * mm_per_px, 2) if mm_per_px else None,
        }

        ti, contact = attach(m, tubings, gap)
        if ti is not None:
            tube = tubings[ti]
            dt_tube, _ = get_tube_dt(ti)
            # Use DT VALUE AT CONTACT POINT = actual local tube radius (~30-50 px).
            # DT max = coil blob half-width (~200-300 px) which makes the crop = full frame.
            cx_c = max(0, min(int(round(contact[0])), dt_tube.shape[1] - 1))
            cy_c = max(0, min(int(round(contact[1])), dt_tube.shape[0] - 1))
            area_w = max(float(dt_tube[cy_c, cx_c]), 4.0)
            # Use fast DT-weighted PCA (no skeletonize) for video inference speed
            u, near, mask_w = local_axis_fast(tube, dt_tube, contact, radius=4 * area_w + 5)
            if u is not None:
                tip = tip_point(m, contact, u)
                start = contact + u * max(0.5 * mask_w, 2)             # a little into bare tubing
                edge_w, q = measure_width(gray, start - u * 10, start,
                                          length=int(max(3 * mask_w, 15)),
                                          search_px=int(0.5 * mask_w * 1.4) + 3,
                                          refine=True, max_angle_deg=6, max_shift=3)
                # Primary width source is YOLO mask:
                e["width_mask_px"] = round(mask_w, 2)
                e["final_width"] = round(mask_w, 2)
                e["width_px"] = round(mask_w, 2)
                e["tip"] = [round(float(tip[0]), 1), round(float(tip[1]), 1)]
                if edge_w is not None:
                    e["width_edge_px"] = round(edge_w, 2)
                    e["edge_quality"] = round(q, 2)
                trans = transparency_score(gray, tip, contact, mask_w)
                e.update({
                    "tubing_id": ti,
                    "transparency": round(trans, 2) if trans is not None else None,
                    "opaque": (trans < 0.5) if trans is not None else None,
                    "colour": colour_score(bgr, (x, y, x + bw, y + bh)),
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

    used_tails = set()
    pair_id = 0
    for h in sorted(heads, key=lambda e: e["width_px"]):

        available = [t for t in tails if id(t) not in used_tails]
        t = closest(h, available)
        if t is not None and closest(t, heads) is h:      # mutual best match only
            pair_id += 1
            h["pair_id"] = t["pair_id"] = pair_id
            used_tails.add(id(t))


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
    for c, m in instances:
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
