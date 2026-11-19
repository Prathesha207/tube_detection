"""
depth3d.py - X, Y, Z (metres, camera frame) for every head / tail, from a depth
video that is ALIGNED to the RGB video (OAK: depth aligned to CAM_A, uint16 mm).

Used by run_video_frames.py and main.py. Config keys (config.yaml):

    depth_video:   path to the *_depth.mkv recorded together with `video`
    camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  of the RGB camera
    calib_width / calib_height: image size the camera_matrix belongs to
    dist_coeffs:   optional lens distortion list (k1 k2 p1 p2 k3 ...); null = none
    depth_scale:   raw depth unit -> metres (OAK: 0.001 = millimetres)
    depth_min_mm / depth_max_mm: raw values outside this range count as "no depth"

Coordinate frame: origin at the RGB camera, X right, Y down, Z forward (metres).

How Z is taken for one end:
  * the end's segmentation mask, shrunk by 2 px (edges of objects have bad depth),
  * every mask pixel with a valid depth -> back-projected to 3D,
  * the MEDIAN X, Y, Z of those points (robust to holes and stray background pixels).
  * no mask / too few valid pixels -> median of a 7x7 patch at the box centre.
"""
import re
import subprocess

import cv2
import numpy as np

DEPTH_PIX_FMTS = ("gray16", "y16", "b16g", "gray12", "gray10")
MAX_SAMPLES = 4000          # mask pixels used per end (subsampled for speed)


# ------------------------------------------------------------------ video in
def probe_video(path):
    """(width, height, fps, stream line) using `ffmpeg -i` (ffprobe not needed)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path)],
                             capture_output=True, text=True).stderr
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found - install it and put it on PATH") from None
    lines = [l.strip() for l in out.splitlines() if "Video:" in l]
    if not lines:
        raise RuntimeError(f"no video stream in {path}\n{out[-800:]}")
    line = lines[0]
    w, h = map(int, re.search(r"\b(\d{2,5})x(\d{2,5})\b", line).groups())
    m = re.search(r"([\d.]+) fps", line)
    return w, h, (float(m.group(1)) if m else None), line


class DepthReader:
    """Reads a 16-bit depth video frame by frame through ffmpeg (OpenCV cannot
    read 16-bit video reliably). read() -> uint16 array (H, W) or None at the end."""

    def __init__(self, path):
        self.path = str(path)
        self.w, self.h, self.fps, self.stream = probe_video(self.path)
        self.is_16bit = any(f in self.stream for f in DEPTH_PIX_FMTS)
        if not self.is_16bit:
            print(f"[depth] WARNING: {self.path} is not 16-bit grayscale ({self.stream}). "
                  "If it is h264/hevc/yuv420p the real distances are lost - Z will be wrong.")
        self.nbytes = self.w * self.h * 2
        self.frames_read = 0
        self.ended = False
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", self.path, "-map", "0:v:0",
             "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        if self.ended:
            return None
        buf = self.proc.stdout.read(self.nbytes)
        if len(buf) < self.nbytes:
            self.ended = True
            return None
        self.frames_read += 1
        return np.frombuffer(buf, np.uint16).reshape(self.h, self.w)

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def find_depth_pair(video):
    """OAK recorder naming: <name>_video.mp4 -> <name>_depth.mkv in the same folder."""
    from pathlib import Path
    p = Path(str(video))
    base = p.stem[:-len("_video")] if p.stem.endswith("_video") else p.stem
    names = [f"{base}_depth.mkv", f"{p.stem}_depth.mkv"]
    if p.stem.endswith("_temp"):              # clip001_temp.mp4 -> clip001_depth_raw_temp.mkv
        b = p.stem[:-len("_temp")]
        names += [f"{b}_depth_raw_temp.mkv", f"{b}_depth_temp.mkv", f"{b}_depth_raw.mkv"]
    for name in names:
        c = p.with_name(name)
        if c.exists():
            return str(c)
    clip = p.stem.split("_")[0]               # last resort: ONE clip001_depth*.mkv in the folder
    hits = sorted(p.parent.glob(f"{clip}_depth*.mkv"))
    return str(hits[0]) if len(hits) == 1 else None


def open_depth(cfg, rgb_fps=None, video=None):
    """DepthReader for cfg.depth_video, or None when depth is not configured.
    depth_video: auto -> the *_depth.mkv next to the RGB video (see find_depth_pair)."""
    path = getattr(cfg, "depth_video", None)
    if not path:
        return None
    if not getattr(cfg, "camera_matrix", None):
        raise RuntimeError("depth_video is set but camera_matrix is missing in the config")
    if str(path).lower() == "auto":
        path = find_depth_pair(video or cfg.video)
        if path is None:
            print(f"[depth] WARNING: depth_video=auto but no *_depth.mkv found next to "
                  f"{video or cfg.video} - running without 3D")
            return None
    r = DepthReader(path)
    print(f"[depth] {path}\n        {r.stream}")
    if rgb_fps and r.fps and abs(rgb_fps - r.fps) > 0.5:
        print(f"[depth] WARNING: fps differ (RGB {rgb_fps:.2f}, depth {r.fps:.2f}) - "
              "frames may drift out of sync")
    return r


# -------------------------------------------------------------------- camera
class Camera:
    """Pinhole camera scaled to the frame size, optional lens distortion."""

    def __init__(self, camera_matrix, calib_size, frame_size, dist_coeffs=None):
        K = np.array(camera_matrix, np.float64).reshape(3, 3)
        cw, ch = calib_size
        fw, fh = frame_size
        sx, sy = fw / cw, fh / ch
        self.fx, self.fy = K[0, 0] * sx, K[1, 1] * sy
        self.cx, self.cy = K[0, 2] * sx, K[1, 2] * sy
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.dist = np.array(dist_coeffs, np.float64).ravel() if dist_coeffs else None
        self.scaled = (fw, fh) != (cw, ch)

    def backproject(self, u, v, z):
        """Pixels (u, v) + depth z (metres) -> X, Y (metres). Arrays in, arrays out."""
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        if self.dist is not None and self.dist.any():
            pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float32)
            n = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
            xn, yn = n[:, 0].astype(np.float64), n[:, 1].astype(np.float64)
        else:
            xn, yn = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        return xn * z, yn * z


_CAMERAS = {}


def camera_for(cfg, frame_w, frame_h):
    """Camera for this frame size (built once, then cached)."""
    cw = int(cfg.calib_width or frame_w)
    ch = int(cfg.calib_height or frame_h)
    key = (str(cfg.camera_matrix), str(cfg.dist_coeffs), cw, ch, frame_w, frame_h)
    cam = _CAMERAS.get(key)
    if cam is None:
        cam = Camera(cfg.camera_matrix, (cw, ch), (frame_w, frame_h), cfg.dist_coeffs)
        _CAMERAS[key] = cam
        print(f"[depth] intrinsics for {frame_w}x{frame_h}: fx={cam.fx:.2f} fy={cam.fy:.2f} "
              f"cx={cam.cx:.2f} cy={cam.cy:.2f}"
              + (f"  (scaled from {cw}x{ch})" if cam.scaled else "")
              + ("  + distortion" if cam.dist is not None else ""))
    return cam


_WARNED = set()


def align_to_frame(depth, frame_w, frame_h):
    """Depth must have the RGB frame's size. Different size -> nearest-neighbour
    resize (only correct if the depth was already aligned to the RGB camera)."""
    dh, dw = depth.shape[:2]
    if (dw, dh) == (frame_w, frame_h):
        return depth
    if (dw, dh, frame_w, frame_h) not in _WARNED:
        _WARNED.add((dw, dh, frame_w, frame_h))
        print(f"[depth] NOTE: depth is {dw}x{dh}, RGB is {frame_w}x{frame_h} - resizing depth. "
              "Only correct if the depth was recorded aligned to the RGB camera.")
    return cv2.resize(depth, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ per end
def end_xyz(depth, cam, mask, k, center, cfg, box=None):
    """3D position of one head/tail.
    depth  uint16 depth frame, same size as the ORIGINAL RGB frame
    mask   bool mask in the (possibly scaled, factor k) inference frame, or None
    center (x, y) box centre in ORIGINAL pixels
    box    optional (x0, y0, x1, y1) of the mask's pixels (inference frame): the work is
           then done on that crop (+3 px) instead of the whole frame - same result, faster
    -> {"xyz": (X, Y, Z) metres, "src": "mask"/"center", "valid": n_pixels} or None"""
    H, W = depth.shape[:2]
    lo, hi = cfg.depth_min_mm, cfg.depth_max_mm
    u = v = raw = None
    src = None

    ox = oy = 0
    if mask is not None and box is not None:
        mh, mw = mask.shape
        ox, oy = max(box[0] - 3, 0), max(box[1] - 3, 0)
        mask = mask[oy:min(box[3] + 3, mh), ox:min(box[2] + 3, mw)]
    if mask is not None and mask.any():
        m = mask
        er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        if er.sum() >= 20:                         # keep the shrunk mask unless it vanished
            m = er
        ys, xs = np.nonzero(m)
        ys, xs = ys + oy, xs + ox
        if len(xs) > MAX_SAMPLES:
            sel = np.linspace(0, len(xs) - 1, MAX_SAMPLES).astype(int)
            ys, xs = ys[sel], xs[sel]
        uu = np.clip(np.round(xs / k).astype(int), 0, W - 1)      # back to original pixels
        vv = np.clip(np.round(ys / k).astype(int), 0, H - 1)
        rr = depth[vv, uu]
        ok = (rr >= lo) & (rr <= hi)
        if ok.sum() >= 10:
            u, v, raw, src = uu[ok], vv[ok], rr[ok], "mask"

    if src is None:                                # fallback: small patch at the box centre
        cx = int(min(max(round(center[0]), 0), W - 1))
        cy = int(min(max(round(center[1]), 0), H - 1))
        y0, y1, x0, x1 = max(cy - 3, 0), min(cy + 4, H), max(cx - 3, 0), min(cx + 4, W)
        vv, uu = np.mgrid[y0:y1, x0:x1]
        rr = depth[y0:y1, x0:x1]
        ok = (rr >= lo) & (rr <= hi)
        if ok.sum() >= 3:
            u, v, raw, src = uu[ok], vv[ok], rr[ok], "center"

    if src is None:
        return None
    z = raw.astype(np.float64) * cfg.depth_scale
    X, Y = cam.backproject(u, v, z)
    return {"xyz": (float(np.median(X)), float(np.median(Y)), float(np.median(z))),
            "src": src, "valid": int(len(z))}


def width_mm(width_px, z_m, fx):
    """Pixel width -> millimetres at distance z (tube roughly facing the camera)."""
    if not width_px or not z_m or not fx:
        return None
    return width_px * z_m / fx * 1000.0


def r3(xyz):
    return [round(float(c), 4) for c in xyz] if xyz is not None else None