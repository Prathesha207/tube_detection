"""
wrapper_trace.py - follow every tube OUT of the wrapper for `trace_px` pixels and give
the X, Y, Z (metres, camera frame) of where that trace ends.

    trace_px: 150          # config.yaml, ORIGINAL frame pixels (null / 0 = off)
    analyzer.trace_px = 150             # or change it at run time from the backend
    analyzer.process_frame(frame, depth=d, trace_px=220)   # or for one frame only

How one trace is made (per tube mask, per frame):
  1. wrapper = union of the model's `wrapper` masks, grown by trace_wrapper_gap px.
  2. tube mask minus the grown wrapper -> pieces. A piece that touches the wrapper's
     border ring is a tube LEAVING the wrapper; the touching pixels = the exit point.
     (A piece touching the wrapper in two places -> two exits.)
  3. From the exit, a minimum-cost path runs through the tube mask to the far end of
     that piece. Cost is lowest on the tube's centre line, so the path follows the
     middle of the tube, around bends, never jumping to a neighbouring tube.
  4. The path is smoothed and walked for trace_px pixels of ARC LENGTH (along the
     tube, not a straight line) -> end point. Piece shorter than trace_px (tube hidden
     by a hand / another tube / the image edge) -> end at the last visible pixel,
     status "SHORT", reached_px says how far it got.
  5. X, Y, Z of the start and end: median of the depth pixels inside the tube mask
     within trace_depth_radius px of the point, back-projected with camera_matrix.

Side: the exits are split into two sides of the wrapper along the direction the
tubes cross it ("left"/"right" when that is mostly horizontal in the image,
"up"/"down" when mostly vertical).

Transparent tubing: stereo depth often sees THROUGH clear tube to the surface under
it. The tube lies on that surface, so Z is then off by about one tube radius (a few
mm) - fine for position, not for measuring the tube's own height.
"""
import cv2
import numpy as np
from skimage.graph import MCP_Geometric

try:
    from . import depth3d
except ImportError:
    import depth3d


def _bbox(mask):
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def prepare_wrapper(wrapper_masks, gap):
    """Union of the wrapper masks -> dict with the grown wrapper and its border ring,
    both FULL-frame bool (inference coordinates), or None when there is no wrapper."""
    masks = [m for m in wrapper_masks if m is not None and m.any()]
    if not masks:
        return None
    union = masks[0].copy()
    for m in masks[1:]:
        union |= m
    H, W = union.shape
    x0, y0, x1, y1 = _bbox(union)
    g = max(int(round(gap)), 0)
    pad = g + 6
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    sub = union[Y0:Y1, X0:X1].astype(np.uint8)
    grown = cv2.dilate(sub, np.ones((2 * g + 1, 2 * g + 1), np.uint8)) if g else sub
    ring = cv2.dilate(sub, np.ones((2 * g + 7, 2 * g + 7), np.uint8)) & (1 - grown)
    grown_f = np.zeros((H, W), bool)
    ring_f = np.zeros((H, W), bool)
    grown_f[Y0:Y1, X0:X1] = grown.astype(bool)
    ring_f[Y0:Y1, X0:X1] = ring.astype(bool)
    cs, _ = cv2.findContours(union[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea) + np.array([x0, y0])
    rect = cv2.minAreaRect(c.astype(np.float32))
    ys, xs = np.nonzero(sub.astype(bool))
    return {"grown": grown_f, "ring": ring_f, "box": (X0, Y0, X1, Y1),
            "center": np.array([xs.mean() + X0, ys.mean() + Y0], np.float32),
            "rect": rect, "contour": c}


def _smooth_path(pts, win):
    """Moving average along the path (ends kept fixed-ish by edge padding)."""
    if win <= 1 or len(pts) < 3:
        return pts
    win = min(int(win) | 1, (len(pts) // 2) * 2 + 1)
    h = win // 2
    padded = np.pad(pts, ((h, h), (0, 0)), mode="edge")
    ker = np.ones(win, np.float32) / win
    out = np.stack([np.convolve(padded[:, i], ker, mode="valid") for i in range(2)], 1)
    out[0] = pts[0]
    return out


def _walk(pts, length):
    """Point at `length` of arc along pts, the index range used, and the total length."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if length >= total:
        return pts[-1].copy(), len(pts), total
    i = int(np.searchsorted(cum, length))           # cum[i-1] < length <= cum[i]
    t = (length - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    return pts[i - 1] + t * (pts[i] - pts[i - 1]), i, total


def trace_piece(piece, start_rc, length, center_weight=4.0, smooth=9):
    """Centre-line path through `piece` (bool crop) from start_rc (row, col) to its
    far end, cut at `length` px of arc. -> (path xy float array, end xy, reached px,
    local half-width at the end) or None."""
    dt = cv2.distanceTransform(piece.astype(np.uint8), cv2.DIST_L2, 3)
    dmax = float(dt.max())
    if dmax <= 0:
        return None
    cost = np.where(piece, 1.0 + center_weight * (1.0 - dt / dmax), np.inf)
    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs([start_rc])
    cum = np.where(np.isfinite(cum), cum, -1)
    far = np.unravel_index(int(np.argmax(cum)), cum.shape)
    if cum[far] <= 0:
        return None
    rc = np.array(mcp.traceback(far), np.float32)
    pts = _smooth_path(rc[:, ::-1], smooth)              # (row, col) -> (x, y)
    end, i_end, total = _walk(pts, length)
    path = np.vstack([pts[:i_end], end[None]]) if i_end < len(pts) else pts
    ex, ey = int(round(end[0])), int(round(end[1]))
    ey, ex = min(max(ey, 0), dt.shape[0] - 1), min(max(ex, 0), dt.shape[1] - 1)
    return path, end, min(length, total), float(dt[ey, ex])


def point_xyz(depth, cam, mask, pt, radius, k, cfg):
    """Median X, Y, Z of the valid depth pixels of `mask` within `radius` of pt
    (pt / mask / radius in inference coordinates; depth in ORIGINAL pixels).
    Falls back to the disk without the mask. -> ((X, Y, Z), source) or (None, None)."""
    if depth is None or cam is None:
        return None, None
    H, W = depth.shape[:2]
    r = max(int(np.ceil(radius)), 1)
    x, y = float(pt[0]), float(pt[1])
    mh, mw = mask.shape
    x0, x1 = max(int(x) - r - 3, 0), min(int(x) + r + 4, mw)
    y0, y1 = max(int(y) - r - 3, 0), min(int(y) + r + 4, mh)
    if x0 >= x1 or y0 >= y1:
        return None, None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    for src, sel in (("mask", (d2 <= r * r) & mask[y0:y1, x0:x1]),
                     ("disk", d2 <= (r + 3) ** 2)):
        if sel.sum() < 3:
            continue
        uu = np.clip(np.round(xx[sel] / k).astype(int), 0, W - 1)
        vv = np.clip(np.round(yy[sel] / k).astype(int), 0, H - 1)
        raw = depth[vv, uu]
        ok = (raw >= cfg.depth_min_mm) & (raw <= cfg.depth_max_mm)
        if ok.sum() >= 3:
            z = raw[ok].astype(np.float64) * cfg.depth_scale
            X, Y = cam.backproject(uu[ok], vv[ok], z)
            return (float(np.median(X)), float(np.median(Y)), float(np.median(z))), src
    return None, None


def trace_tube(tube_mask, wrapper, cfg, k=1.0, depth=None, cam=None, tube_box=None):
    """Every exit of one tube mask out of the wrapper, traced for cfg.trace_px.
    tube_mask: FULL-frame bool (inference coordinates). Returns a list of dicts in
    ORIGINAL pixels (no 'side' yet - set_sides() adds it for the whole frame)."""
    length_px = float(getattr(cfg, "trace_px", 0) or 0)
    if wrapper is None or length_px <= 0 or tube_mask is None:
        return []
    H, W = tube_mask.shape
    b = tube_box or _bbox(tube_mask)
    if b is None:
        return []
    x0, y0, x1, y1 = max(int(b[0]) - 2, 0), max(int(b[1]) - 2, 0), min(int(b[2]) + 3, W), min(int(b[3]) + 3, H)
    wx0, wy0, wx1, wy1 = wrapper["box"]
    if x1 <= wx0 or wx1 <= x0 or y1 <= wy0 or wy1 <= y0:
        return []                                         # nowhere near the wrapper
    tube = tube_mask[y0:y1, x0:x1]
    outside = tube & ~wrapper["grown"][y0:y1, x0:x1]
    ring = wrapper["ring"][y0:y1, x0:x1]
    min_area = int(getattr(cfg, "trace_min_area", 40) or 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8), connectivity=8)
    L = length_px * k                                     # inference pixels
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        px, py, pw, ph = stats[i, :4]
        piece = lab[py:py + ph, px:px + pw] == i
        touch = piece & ring[py:py + ph, px:px + pw]
        if touch.sum() < 2:
            continue
        nt, tl = cv2.connectedComponents(cv2.dilate(touch.astype(np.uint8), np.ones((5, 5), np.uint8)))
        for j in range(1, nt):
            cyx = np.argwhere((tl == j) & touch)
            if len(cyx) < 2:
                continue
            contact = cyx.mean(0)                          # (row, col) in the piece crop
            pr = np.argwhere(piece)
            s = pr[int(np.argmin(((pr - contact) ** 2).sum(1)))]
            # a path of arc length L never leaves a square of half-size L around its
            # start: trace only inside it (a big coil piece would otherwise cost a lot)
            R = int(L * 1.15) + 20
            wy0, wx0 = max(int(s[0]) - R, 0), max(int(s[1]) - R, 0)
            win = piece[wy0:int(s[0]) + R + 1, wx0:int(s[1]) + R + 1]
            tr = trace_piece(win, (int(s[0]) - wy0, int(s[1]) - wx0), L,
                             float(getattr(cfg, "trace_center_weight", 4.0)),
                             int(getattr(cfg, "trace_smooth", 9) or 1))
            if tr is None:
                continue
            path, end, reached, half_w = tr
            off = np.array([x0 + px + wx0, y0 + py + wy0], np.float32)
            path_f = path + off                            # full inference frame
            start_f, end_f = path_f[0], end + off
            rad = getattr(cfg, "trace_depth_radius", None) or max(3.0, 0.6 * half_w)
            xyz_s, src_s = point_xyz(depth, cam, tube_mask, start_f, rad, k, cfg)
            xyz_e, src_e = point_xyz(depth, cam, tube_mask, end_f, rad, k, cfg)
            samples = []
            step = float(getattr(cfg, "trace_path_step_px", 0) or 0) * k
            if step > 0 and depth is not None:
                for d_ in np.arange(step, reached + 1e-6, step):
                    p, _, _ = _walk(path_f, d_)
                    xyz, _ = point_xyz(depth, cam, tube_mask, p, rad, k, cfg)
                    samples.append({"along_px": round(d_ / k, 1),
                                    "px": [round(float(p[0] / k), 1), round(float(p[1] / k), 1)],
                                    "xyz_m": depth3d.r3(xyz)})
            q, _, _ = _walk(path_f, min(30.0 * k, max(reached, 1.0)))
            dirv = q - start_f
            dirv = dirv / (np.linalg.norm(dirv) + 1e-9)    # direction the tube leaves in
            simple = cv2.approxPolyDP((path_f / k).astype(np.float32).reshape(-1, 1, 2), 1.5, False)
            out.append({
                "status": "OK" if reached >= L - 1e-6 else "SHORT",
                "trace_px": round(length_px, 1),
                "reached_px": round(reached / k, 1),
                "start_px": [round(float(start_f[0] / k), 1), round(float(start_f[1] / k), 1)],
                "end_px": [round(float(end_f[0] / k), 1), round(float(end_f[1] / k), 1)],
                "start_xyz_m": depth3d.r3(xyz_s), "start_depth_source": src_s,
                "end_xyz_now_m": depth3d.r3(xyz_e), "end_depth_source": src_e,
                "path": np.round(simple.reshape(-1, 2), 1).tolist(),
                "path_xyz": samples or None,
                "_out": dirv,
            })
    return out


def set_sides(traces, wrapper=None, k=1.0):
    """Label every exit with the wrapper side it leaves from. Exits on opposite sides
    leave in opposite directions, so the main axis of the leaving directions is the
    crossing direction; the sign along it picks the side."""
    if not traces:
        return
    v = np.array([t["_out"] for t in traces], np.float32)
    a = np.linalg.svd(v)[2][0] if len(v) >= 2 else v[0]
    horiz = abs(a[0]) >= abs(a[1])
    if (horiz and a[0] < 0) or (not horiz and a[1] < 0):
        a = -a
    names = ("right", "left") if horiz else ("down", "up")
    for t in traces:
        t["side"] = names[0] if float(np.dot(t["_out"], a)) >= 0 else names[1]


def wrapper_record(wrapper, k, conf=None):
    """Wrapper for the JSON: bbox [x, y, w, h], polygon, centre (ORIGINAL pixels)."""
    if wrapper is None:
        return None
    c = wrapper["contour"].reshape(-1, 2).astype(np.float32)
    poly = cv2.approxPolyDP(c.reshape(-1, 1, 2), 2.0, True).reshape(-1, 2) / k
    x, y, w, h = cv2.boundingRect(c)
    return {"conf": round(float(conf), 3) if conf is not None else None,
            "bbox": [int(round(x / k)), int(round(y / k)), int(round(w / k)), int(round(h / k))],
            "center": [round(float(wrapper["center"][0] / k), 1), round(float(wrapper["center"][1] / k), 1)],
            "polygon": np.round(poly).astype(int).tolist()}
