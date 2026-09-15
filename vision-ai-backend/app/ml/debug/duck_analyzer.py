"""
duck_analyzer.py
Streaming, frame-by-frame duck ID + anomaly detection (HYBRID: tracker + anchors).

Frontend pushes ONE frame at a time; each call returns THAT frame's result now.

    from duck_analyzer import DuckAnalyzer
    analyzer = DuckAnalyzer("config.yaml", expected_duck_count=18)
    result = analyzer.process_frame(frame)

IMPORTANT -- load the model ONCE, reuse for every frame
-------------------------------------------------------
The YOLO model + MediaPipe are created a single time in __init__ and then
reused by every process_frame() call -- the model is NOT reloaded per frame.
So for a stream of 50 frames through ONE analyzer, the weights load once.
If your backend instead constructs a NEW DuckAnalyzer() for every request,
that reloads the model each time; construct ONE analyzer at startup and reuse
it (see get_shared_analyzer() at the bottom for a ready-made singleton).

Design (handles MOVING ducks)
-----------------------------
- A real tracker (BoT-SORT via ultralytics model.track) follows each duck as it
  MOVES, giving each a persistent tracker-id that survives motion.
- At anchor-lock (a clean expected-count frame), each tracker-id is bound to a
  stable DISPLAY id 1..N, numbered by position. We then always follow the
  tracker-id but SHOW your 1..N number.
- "Missing" = the tracker lost that duck (truly gone/occluded), NOT "it moved".
- New tracker-ids after lock -> "added" ducks (display ids N+1, ...), gated by
  persistence so brief detector glitches don't create IDs.
- Shake stabilization: anomaly decided on the majority count over
  `anomaly_smoothing_frames`.

Deep re-ID embedder (optional)
------------------------------
- If `use_embedder: true`, a DeepEmbedder (ResNet50 or SigLIP, switchable via
  `embedder_backend`) produces one L2-normalized feature vector per duck crop.
- That vector is blended with the histogram signature in the rebind matcher
  (weight `embed_weight`), which is the fix for similar-coloured duck ID swaps.
- The embedder is OPTIONAL and best-effort: if it's off or a crop fails, the
  matcher transparently falls back to the histogram signature. Nothing else in
  the pipeline changes.
- Each slot keeps its latest embedding under display_info[did]["emb"], which is
  also what a FrameStore logger reads to persist per-frame embeddings.

GPU acceleration
----------------
- Model is explicitly moved onto the configured device (.to(device)).
- One dummy warmup inference runs at load, so the FIRST real frame isn't slow
  (CUDA kernels/memory get initialized on the dummy frame instead).
- Optional FP16 (half precision) via `use_half: true` -- ~2x faster on GPU,
  applied to both detection and model.track(). Slightly changes confidences.
- An optional one-time GPU sanity benchmark prints FP16 TFLOP/s at startup
  (enable with `gpu_benchmark: true`).

Hand short-circuit
------------------
- MediaPipe Hands (default) runs FIRST on every frame. If a hand is detected,
  the whole duck pipeline is SKIPPED for that frame -> {"status":"HAND", ...}.

Thumbnail logic
---------------
- confirmed  : INITIAL anchor ducks, once each at lock.
- added      : a genuinely-new duck beyond anchor count, once, new id.
- other_toys : non-duck object, once when first detected, separate id space.
- missing    : NO thumbnail; only the id is reported (missing_ids).

Per-frame result: see _finish() at the bottom.
- Color coding: whole frame's boxes GREEN if NORMAL, RED if ANOMALY.
- Optional local saving via `save_local` (off by default).
"""

import os
import sys
# --- silence noisy native (C++) logs from MediaPipe / TFLite / absl ---
# These MUST be set before mediapipe / tensorflow-lite get imported, so they
# live at the very top, above the other imports. 3 = errors only.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import time
import base64
from collections import deque, Counter

import warnings
import logging
# --- silence Python-level deprecation chatter (protobuf, ultralytics) ---
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.*")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import cv2
import numpy as np
import yaml
import torch

GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)


def _gpu_sanity_benchmark(device):
    """One-time FP16 matmul benchmark to verify GPU compute capability.
    Purely diagnostic -- prints TFLOP/s. Non-fatal on any error."""
    try:
        torch.cuda.synchronize()
        n = 4096
        a = torch.randn(n, n, device=device, dtype=torch.float16)
        b = torch.randn(n, n, device=device, dtype=torch.float16)
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        dt_per_iter = (time.perf_counter() - t0) / iters
        tflops = (2 * n ** 3) / dt_per_iter / 1e12
        print(f"  [GPU] FP16 matmul sanity check: {n}x{n} in "
              f"{dt_per_iter * 1000:.2f} ms/iter (~{tflops:.1f} TFLOP/s)")
        del a, b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [GPU] [WARN] sanity benchmark failed (non-fatal): {e}")


def get_crop(frame, xyxy, pad_frac=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_frac)); y1 = max(0, int(y1 - bh * pad_frac))
    x2 = min(w, int(x2 + bw * pad_frac)); y2 = min(h, int(y2 + bh * pad_frac))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def color_hist(crop):
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def appearance_sim(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(h1.astype("float32"), h2.astype("float32"),
                                 cv2.HISTCMP_CORREL))


# ---------------------------------------------------------------------- #
#  Richer appearance signature for occlusion re-identification.          #
#  A single HS histogram is weak between similarly-coloured rubber       #
#  ducks, so a reappearing duck often matches several slots almost       #
#  equally and position becomes the (unreliable) tiebreaker. We add two  #
#  cheap extra cues -- mean Lab colour and aspect ratio -- and combine   #
#  them into one similarity score. Purely internal; no schema change.    #
# ---------------------------------------------------------------------- #
def appearance_signature(crop):
    """Return a dict signature {hist, lab, ar} for a crop, or None."""
    if crop is None or crop.size == 0:
        return None
    hist = color_hist(crop)
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    except Exception:
        lab = None
    h, w = crop.shape[:2]
    ar = (w / h) if h > 0 else 0.0
    return {"hist": hist, "lab": lab, "ar": float(ar)}


def signature_sim(s1, s2):
    """Blended appearance similarity in [0,1] from two signatures.
    Weights: histogram 0.6, Lab colour 0.3, aspect ratio 0.1."""
    if s1 is None or s2 is None:
        return 0.0
    # histogram correlation -> map [-1,1] to [0,1]
    hs = appearance_sim(s1.get("hist"), s2.get("hist"))
    hs = max(0.0, hs)
    # Lab distance -> similarity (Lab spans ~0..255; 60 is a soft scale)
    if s1.get("lab") is not None and s2.get("lab") is not None:
        d = float(np.linalg.norm(s1["lab"] - s2["lab"]))
        ls = max(0.0, 1.0 - d / 60.0)
    else:
        ls = 0.0
    # aspect-ratio similarity
    a1, a2 = s1.get("ar", 0.0), s2.get("ar", 0.0)
    if a1 > 0 and a2 > 0:
        rs = max(0.0, 1.0 - abs(a1 - a2) / max(a1, a2))
    else:
        rs = 0.0
    return 0.6 * hs + 0.3 * ls + 0.1 * rs


def _hungarian(cost):
    """Optimal assignment minimizing total cost for a rectangular matrix.
    Uses scipy if available, otherwise a self-contained O(n^3) solver so we
    add NO hard dependency. Returns (row_idx, col_idx) like scipy."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except Exception:
        pass
    # --- pure-numpy Hungarian (square-padded) ---
    n_r, n_c = cost.shape
    n = max(n_r, n_c)
    big = cost.max() + 1.0 if cost.size else 1.0
    C = np.full((n, n), big, dtype=float)
    C[:n_r, :n_c] = cost
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = np.inf; j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_r and j <= n_c:
            rows.append(i - 1); cols.append(j - 1)
    order = np.argsort(rows)
    return np.array(rows)[order], np.array(cols)[order]


def position_sim(c1, c2, diag):
    return max(0.0, 1.0 - np.linalg.norm(c1 - c2) / diag)


def center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def crop_to_base64(crop, quality):
    if crop is None or crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def _rounded_rect(frame, p1, p2, color, radius=6):
    x1, y1 = p1; x2, y2 = p2
    radius = int(max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(frame, (cx, cy), radius, color, -1)


def draw_box(frame, xyxy, label, color, dashed=False):
    x1, y1, x2, y2 = map(int, xyxy)
    bw, bh = x2 - x1, y2 - y1
    cl = int(max(8, min(bw, bh) * 0.25))
    t = 2
    if dashed:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    else:
        for (ax, ay, bx, by) in [
            (x1, y1, x1 + cl, y1), (x1, y1, x1, y1 + cl),
            (x2, y1, x2 - cl, y1), (x2, y1, x2, y1 + cl),
            (x1, y2, x1 + cl, y2), (x1, y2, x1, y2 - cl),
            (x2, y2, x2 - cl, y2), (x2, y2, x2, y2 - cl)]:
            cv2.line(frame, (ax, ay), (bx, by), color, t, cv2.LINE_AA)
    scale = max(0.4, frame.shape[1] / 1800.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 5
    ty2 = y1; ty1 = y1 - th - 2 * pad; tx1 = x1; tx2 = x1 + tw + 2 * pad
    if ty1 < 0:
        ty1, ty2 = y1, y1 + th + 2 * pad
    _rounded_rect(frame, (tx1, ty1), (tx2, ty2), color, radius=6)
    cv2.putText(frame, label, (tx1 + pad, ty2 - pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def draw_banner(frame, text, color):
    w = frame.shape[1]
    scale = max(0.7, w / 1280.0)
    thick = max(2, int(w / 640))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(10 * scale)
    cv2.rectangle(frame, (0, 0), (tw + 2 * pad, th + 2 * pad), color, -1)
    cv2.putText(frame, text, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thick, cv2.LINE_AA)


class DuckAnalyzer:
    def __init__(self, config_path, expected_duck_count=None):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        # Robustly resolve model_path across any working directory / caller
        configured_model = cfg.get("model_path", "best.pt")
        candidates = []
        if configured_model:
            if os.path.isabs(configured_model) and os.path.exists(configured_model):
                candidates.append(configured_model)
            else:
                cfg_dir = os.path.dirname(os.path.abspath(config_path))
                candidates.append(os.path.join(cfg_dir, configured_model))
                this_dir = os.path.dirname(os.path.abspath(__file__))
                ml_dir = os.path.dirname(this_dir)
                candidates.append(os.path.join(ml_dir, configured_model))
                candidates.append(os.path.join(this_dir, configured_model))
                candidates.append(os.path.join(os.getcwd(), configured_model))
        this_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.dirname(this_dir)
        candidates.extend([
            os.path.join(this_dir, "best.pt"),
            os.path.join(ml_dir, "model", "best.pt"),
            os.path.join(ml_dir, "models", "best.pt"),
            os.path.join(ml_dir, "best.pt"),
            os.path.join(os.getcwd(), "models", "best.pt"),
            os.path.join(os.getcwd(), "best.pt"),
        ])
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            candidates.insert(0, os.path.join(sys._MEIPASS, "app", "ml", "model", "best.pt"))
            candidates.insert(1, os.path.join(sys._MEIPASS, "app", "ml", "models", "best.pt"))

        resolved_model = next((c for c in candidates if c and os.path.exists(c)), None)
        self.model_path = resolved_model if resolved_model else configured_model
        self.thumbnail_dir = cfg.get("thumbnail_dir")
        self.annotated_dir = cfg.get("annotated_dir")

        if expected_duck_count is not None:
            self.expected = int(expected_duck_count)
        elif cfg.get("expected_duck_count") is not None:
            self.expected = int(cfg["expected_duck_count"])
        else:
            raise ValueError("expected_duck_count not provided")

        self.duck_cls = int(cfg["duck_class_id"])
        self.other_cls = int(cfg["other_class_id"])
        # -------- tray gating (optional) --------
        # If the model has a trained "tray" class, only ducks/others whose box
        # CENTER falls inside the detected tray are counted; everything outside
        # (a hand reaching in, clutter on the desk, a duck off the tray) is
        # dropped BEFORE any anchor / matching / count logic. Set tray_class_id
        # to -1 (default) to disable gating entirely -> behaves exactly as before.
        # Exactly one tray is assumed: the highest-confidence tray box is used.
        self.tray_cls = int(cfg.get("tray_class_id", -1))
        # pad the tray box outward by this fraction of its size, so a duck right
        # on the rim isn't dropped by a slightly-tight tray box. 0 = exact box.
        self.tray_pad_frac = float(cfg.get("tray_pad_frac", 0.03))
        # if the tray isn't detected on a frame, reuse the last known tray box
        # for up to this many frames (tray briefly occluded by a hand/motion),
        # instead of suddenly counting zero ducks. 0 = no coasting.
        self.tray_coast_frames = int(cfg.get("tray_coast_frames", 15))
        self._last_tray_box = None      # (x1,y1,x2,y2) or None
        self._tray_gone = 0             # frames since tray last seen
        self.conf = float(cfg["conf"])
        self.iou = float(cfg.get("iou", 0.7))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = cfg.get("device", 0)
        self.warmup_frames = int(cfg.get("warmup_frames", 10))
        self.warmup_require_exact = bool(cfg.get("warmup_require_exact", True))
        self.warmup_max_frames = int(cfg.get("warmup_max_frames", 300))
        self.row_tol_frac = float(cfg["row_tolerance_frac"])
        self.jpeg_quality = int(cfg.get("jpeg_quality", 80))
        self.smooth_n = int(cfg.get("anomaly_smoothing_frames", 5))
        self.save_local = bool(cfg.get("save_local", False))
        self.tracker = cfg.get("tracker", "botsort.yaml")
        self.new_id_patience = int(cfg.get("new_id_patience", 20))
        self.missing_patience = int(cfg.get("missing_patience", 5))
        self.rebind_pos_weight = float(cfg.get("rebind_pos_weight", 0.5))
        self.rebind_app_weight = float(cfg.get("rebind_app_weight", 0.5))
        # When ducks are CLOSE together, position can't tell them apart, so IDs
        # swap. In that case lean on appearance (colour) instead. This is the
        # appearance weight used specifically for a crowded match; the normal
        # weights above are used when the candidate slot is well separated.
        # crowd_dist_frac: how close (as a fraction of the frame diagonal) two
        # candidates must be to count as "crowded".
        self.rebind_app_weight_crowded = float(
            cfg.get("rebind_app_weight_crowded", 0.8))
        self.crowd_dist_frac = float(cfg.get("crowd_dist_frac", 0.06))
        self.debug = bool(cfg.get("debug", False))
        self.rebind_threshold = float(cfg.get("rebind_threshold", 0.2))
        self.reappear_patience = int(cfg.get("reappear_patience", 5))
        self.strict_count = bool(cfg.get("strict_count", True))
        # A rebind must ALSO clear a minimum APPEARANCE similarity, not just the
        # blended score. Without this a wrong-colour duck that merely drifts
        # over a missing duck's position reclaims the slot on position alone
        # (0.5*pos easily beats rebind_threshold), masking the loss and forcing
        # a false NORMAL. Set 0 to disable. ~0.35 rejects clearly-different ducks.
        self.rebind_min_app = float(cfg.get("rebind_min_app", 0.35))

        # -------- GPU / precision --------
        # use_half -> FP16 inference (faster on GPU; only applies when running
        # on a CUDA device, ignored on CPU). gpu_benchmark -> one-time TFLOP/s
        # sanity print at startup.
        self.use_half = bool(cfg.get("use_half", False))
        self.gpu_benchmark = bool(cfg.get("gpu_benchmark", False))
        # resolve a device string for torch/ultralytics (e.g. "cuda:0" / "cpu")
        self._device_str = self._resolve_device(self.device)
        # FP16 only makes sense on CUDA -- force it off on CPU.
        if self.use_half and not self._device_str.startswith("cuda"):
            print("[GPU] use_half requested but device is CPU -> disabling half.")
            self.use_half = False

        # -------- hand detection (short-circuits a frame) --------
        self.hand_backend = str(cfg.get("hand_backend", "mediapipe")).lower()
        self.hand_model_path = cfg.get("hand_model_path")   # only used by "yolo"
        self.hand_conf = float(cfg.get("hand_conf", 0.5))
        self.hand_track_conf = float(cfg.get("hand_track_conf", 0.5))
        self.hand_device = cfg.get("hand_device", self.device)
        self.hand_model = None
        self._mp_hands = None
        self.hand_hold_frames = int(cfg.get("hand_hold_frames", 5))
        self._hand_hold = 0

        # -------- hand ROI (draw once, then fixed) --------
        # If a ROI is set, the hand short-circuit fires ONLY when a detected
        # hand overlaps this rectangle -- a hand elsewhere in the frame is
        # ignored. The ROI is stored as [x, y, w, h] in a small JSON file
        # (roi_path) so it persists across runs: draw it one time (see
        # calibrate_roi / draw_roi.py) and every later run reuses it. With no
        # ROI file present, behaviour is unchanged (whole frame counts).
        self.roi_path = cfg.get("roi_path", "hand_roi.json")
        if self.roi_path and not os.path.isabs(self.roi_path):
            cfg_dir = os.path.dirname(os.path.abspath(config_path))
            this_dir = os.path.dirname(os.path.abspath(__file__))
            for r_cand in [
                os.path.join(cfg_dir, self.roi_path),
                os.path.join(this_dir, self.roi_path),
                os.path.join(os.path.dirname(this_dir), self.roi_path),
            ]:
                if os.path.exists(r_cand):
                    self.roi_path = r_cand
                    break
        self.roi = None  # (x, y, w, h) or None
        self._load_roi()

        # other_toys persistence gate
        self.other_id_patience = int(cfg.get("other_id_patience", 8))
        self._prov_other = {}

        if self.save_local:
            if self.thumbnail_dir:
                os.makedirs(self.thumbnail_dir, exist_ok=True)
            if self.annotated_dir:
                os.makedirs(self.annotated_dir, exist_ok=True)

        # ---- load models ONCE, on device, with a warmup pass ----
        from ultralytics import YOLO

        if self.gpu_benchmark and self._device_str.startswith("cuda"):
            _gpu_sanity_benchmark(self._device_str)

        self.model = YOLO(self.model_path)
        try:
            self.model.to(self._device_str)
        except Exception as e:
            print(f"[GPU] [WARN] could not move duck model to "
                  f"{self._device_str}: {e}")
        # FP16 is applied ONCE here by half-ing the model weights, instead of
        # passing the (now-deprecated) half=/quantize= argument on every
        # predict/track call -- which floods the console with deprecation
        # warnings on newer ultralytics. Half weights on a CUDA device give the
        # same FP16 speedup with no per-call argument.
        if self.use_half and self._device_str.startswith("cuda"):
            try:
                self.model.model.half()
            except Exception as e:
                print(f"[GPU] [WARN] could not set duck model to half: {e}")
        self._log_model_device("YOLO-duck")
        # dummy warmup so the first REAL frame isn't slow (CUDA init happens here)
        self._warmup_model()

        # ---- deep re-ID embedder (optional; None if use_embedder: false) ----
        # Reads use_embedder / embedder_backend from the same config. When on,
        # each duck crop gets an L2-normalized feature vector that is blended
        # into the rebind appearance score (embed_weight). Best-effort: any
        # failure returns None and the matcher falls back to the histogram
        # signature, so the rest of the pipeline is unaffected.
        try:
            try:
                from .embedder import DeepEmbedder      # installed as a package / relative
            except (ImportError, ValueError):
                try:
                    from app.ml.debug.embedder import DeepEmbedder  # absolute package import
                except ImportError:
                    from embedder import DeepEmbedder        # run as a flat script / on sys.path
            self.embedder = DeepEmbedder.from_config(cfg)
        except Exception as e:
            print(f"[EMB] [WARN] embedder import/init failed (non-fatal): {e}")
            self.embedder = None
        self.embed_weight = float(cfg.get("embed_weight", 0.5))
        # SWAP-GUARD: even on a direct tracker-id hit, verify the crop's deep
        # embedding still agrees with the slot it claims. When the tracker holds
        # an id but silently lands it on the WRONG duck (in-place swap during a
        # crossing), the direct-hit path would otherwise trust it blindly and
        # the embedder never gets a vote. If the appearance agreement falls
        # below this threshold, the detection is demoted to `pending` so the
        # global Hungarian rebind can reassign it by appearance. Needs the
        # embedder ON; set 0 (or embedder off) to disable the guard entirely.
        # ~0.5 is a reasonable start: lower = more tolerant (fewer demotions),
        # higher = stricter (catches subtler swaps but risks demoting a duck
        # that merely changed pose/lighting).
        # Default 0.0 = OFF (backward-compatible). Set ~0.5 in config to enable.
        # With the true-cosine embedder.cosine (identical~1.0, different~0.0),
        # 0.5 is a reasonable middle; tune per your footage.
        self.swap_verify_min_app = float(cfg.get("swap_verify_min_app", 0.0))

        # init the chosen hand backend
        if self.hand_backend == "mediapipe":
            try:
                import mediapipe as mp
                self._mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.hand_conf,
                    min_tracking_confidence=self.hand_track_conf,
                )
            except ImportError:
                raise ImportError(
                    "hand_backend is 'mediapipe' but the mediapipe package is "
                    "not installed. Run: pip install mediapipe  (or set "
                    "hand_backend: yolo in config.yaml to use a YOLO model).")
        elif self.hand_backend == "yolo":
            if self.hand_model_path:
                self.hand_model = YOLO(self.hand_model_path)
                try:
                    self.hand_model.to(self._device_str)
                    if self.use_half and self._device_str.startswith("cuda"):
                        self.hand_model.model.half()
                except Exception as e:
                    print(f"[GPU] [WARN] could not move hand model to "
                          f"{self._device_str}: {e}")

        # state
        self.frame_idx = 0
        self.diag = None
        self.anchor_locked = False
        self.warmup_best = None
        self.warmup_count = 0

        self.tid_to_display = {}
        self.display_info = {}
        self.next_display_id = 1
        self.num_anchor = 0

        self.otid_to_display = {}
        self.other_info = {}
        self.next_other_display = 1
        self.num_other_anchor = 0

        self.prov_new = {}
        self.reclaim_candidates = {}

        self.confirmed_sent = set()
        self.missing_active = set()
        self.other_sent = set()

        self.count_history = deque(maxlen=self.smooth_n)
        self._last_time = None

    # ------------------------------------------------------------------ #
    #  GPU helpers                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_cuda_operational():
        try:
            import torch
            if not torch.cuda.is_available():
                print("[GPU] CUDA is_available() is False. Ensure NVIDIA drivers are updated and PyTorch DLLs are packaged.")
                return False
            if torch.cuda.device_count() == 0:
                print("[GPU] CUDA device_count() is 0. No NVIDIA GPU detected by PyTorch.")
                return False
            t = torch.zeros((1, 1), device="cuda:0")
            _ = t + 1.0
            del t
            torch.cuda.synchronize()
            return True
        except Exception as e:
            print(f"[GPU] CUDA operational test failed with exception: {e}")
            return False

    @classmethod
    def _resolve_device(cls, device):
        """Turn a config device value (0, "0", "cuda:0", "cpu", None) into a
        torch/ultralytics device string, falling back to CPU if CUDA is
        unavailable or cannot run kernels."""
        if not cls._is_cuda_operational():
            return "cpu"
        if device is None:
            return "cuda:0"
        s = str(device).strip().lower()
        if s in ("cpu",):
            return "cpu"
        if s.startswith("cuda"):
            return s
        # bare integer like 0 / "0" -> cuda:0
        if s.isdigit():
            return f"cuda:{s}"
        return "cuda:0"

    def _log_model_device(self, label):
        try:
            actual = next(self.model.model.parameters()).device
        except Exception:
            actual = "unknown"
        print(f"[GPU-CHECK] {label:<12} -> device={actual}  half={self.use_half}")

    def _warmup_model(self):
        """One dummy inference so the first real frame doesn't pay CUDA init."""
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self._device_str,
                               imgsz=self.imgsz, verbose=False)
            print("[GPU] duck model warmup inference OK")
        except Exception as e:
            if self._device_str != "cpu":
                print(f"[GPU] [WARN] duck model warmup on {self._device_str} failed: {e}. Falling back to CPU.")
                self._device_str = "cpu"
                self.use_half = False
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            else:
                print(f"[GPU] [WARN] duck model warmup failed (non-fatal): {e}")

    def set_expected_duck_count(self, n):
        self.expected = int(n)

    def _embed(self, crop):
        """Deep embedding for a crop, or None when the embedder is off/failed.
        Thin wrapper so callers don't each repeat the None-check."""
        if self.embedder is None:
            return None
        return self.embedder.embed(crop)

    # ------------------------------------------------------------------ #
    #  Tray gating helpers                                                #
    # ------------------------------------------------------------------ #
    def _pick_tray_box(self, dets):
        """Return the padded (x1,y1,x2,y2) of the single best tray detection
        this frame, or None if no tray class / no tray detected. Exactly one
        tray is assumed, so we take the highest-confidence tray box. Coasts on
        the last known box for up to tray_coast_frames when the tray is briefly
        missing (e.g. a hand passes over it)."""
        if self.tray_cls is None or self.tray_cls < 0:
            return None  # gating disabled
        best = None
        best_cf = -1.0
        for xyxy, c, cf, tid in dets:
            if c == self.tray_cls and cf > best_cf:
                best_cf = cf
                best = xyxy
        if best is not None:
            x1, y1, x2, y2 = [float(v) for v in best]
            # pad outward so ducks on the rim aren't excluded
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * self.tray_pad_frac, bh * self.tray_pad_frac
            box = (x1 - px, y1 - py, x2 + px, y2 + py)
            self._last_tray_box = box
            self._tray_gone = 0
            return box
        # no tray this frame -> coast on last known box for a few frames
        if self._last_tray_box is not None and self._tray_gone < self.tray_coast_frames:
            self._tray_gone += 1
            return self._last_tray_box
        return None

    @staticmethod
    def _center_in_box(xyxy, box):
        """True if the CENTER of detection xyxy lies within tray box."""
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        bx1, by1, bx2, by2 = box
        return bx1 <= cx <= bx2 and by1 <= cy <= by2

    def _blend_app(self, hist_sim, dsig_emb, slot_emb):
        """Blend the histogram-signature similarity with the deep-embedding
        cosine, if the embedder is on and both vectors exist. Falls back to the
        histogram score unchanged when the embedder is off/missing a vector."""
        if self.embedder is None or dsig_emb is None or slot_emb is None:
            return hist_sim
        de = self.embedder.cosine(dsig_emb, slot_emb)
        w = self.embed_weight
        return (1.0 - w) * hist_sim + w * de

    # ------------------------------------------------------------------ #
    #  Hand ROI helpers                                                   #
    # ------------------------------------------------------------------ #
    def _load_roi(self):
        """Load the fixed hand ROI from roi_path if it exists."""
        try:
            import json
            if self.roi_path and os.path.exists(self.roi_path):
                with open(self.roi_path, "r") as f:
                    d = json.load(f)
                r = d.get("roi", d)  # accept {"roi":[...]} or a bare [...]
                if r and len(r) == 4:
                    self.roi = tuple(int(v) for v in r)
                    print(f"[ROI] loaded hand ROI {self.roi} from {self.roi_path}")
        except Exception as e:
            print(f"[ROI] [WARN] could not load ROI from {self.roi_path}: {e}")
            self.roi = None

    def set_roi(self, x, y, w, h, save=True):
        """Set the ROI programmatically and (optionally) persist it."""
        self.roi = (int(x), int(y), int(w), int(h))
        if save and self.roi_path:
            try:
                import json
                with open(self.roi_path, "w") as f:
                    json.dump({"roi": list(self.roi)}, f)
                print(f"[ROI] saved hand ROI {self.roi} to {self.roi_path}")
            except Exception as e:
                print(f"[ROI] [WARN] could not save ROI: {e}")

    def calibrate_roi(self, frame, window="Draw hand ROI - drag, ENTER to accept"):
        """Open an interactive window on `frame`, let the user drag ONE
        rectangle, save it to roi_path, and set it. Blocking; call once at
        setup. Returns the (x,y,w,h) chosen, or None if cancelled."""
        try:
            box = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window)
        except Exception as e:
            print(f"[ROI] [WARN] interactive selectROI failed: {e}")
            return None
        x, y, w, h = [int(v) for v in box]
        if w <= 0 or h <= 0:
            print("[ROI] selection cancelled / empty -> ROI unchanged.")
            return None
        self.set_roi(x, y, w, h, save=True)
        return self.roi

    @staticmethod
    def _rects_overlap(a, b):
        """True if rectangles a,b (each x,y,w,h) share any area."""
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or
                    ay + ah <= by or by + bh <= ay)

    # ------------------------------------------------------------------ #
    #  Hand detection                                                    #
    # ------------------------------------------------------------------ #
    def _hand_boxes(self, frame):
        """Return a list of hand bounding boxes [(x,y,w,h), ...] for this
        frame (empty if none). Works for both mediapipe and yolo backends."""
        boxes = []
        h, w = frame.shape[:2]

        if self._mp_hands is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    for hand_lms in res.multi_hand_landmarks:
                        xs = [lm.x for lm in hand_lms.landmark]
                        ys = [lm.y for lm in hand_lms.landmark]
                        x1 = max(0, int(min(xs) * w)); y1 = max(0, int(min(ys) * h))
                        x2 = min(w, int(max(xs) * w)); y2 = min(h, int(max(ys) * h))
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2 - x1, y2 - y1))
                return boxes
            except Exception as mp_err:
                print(f"[HAND] [WARN] mediapipe process failed (non-fatal): {mp_err}")
                return []

        if self.hand_model is not None:
            r = self.hand_model.predict(source=frame, conf=self.hand_conf,
                                        device=self._device_str,
                                        verbose=False)[0]
            if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                for xyxy in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
            else:
                kpts = getattr(r, "keypoints", None)
                if kpts is not None and getattr(kpts, "data", None) is not None \
                        and len(kpts.data) > 0:
                    for kp in kpts.data.cpu().numpy():
                        pts = kp[kp[:, 2] > 0.1] if kp.shape[-1] >= 3 else kp
                        if len(pts) == 0:
                            continue
                        x1 = max(0, int(pts[:, 0].min())); y1 = max(0, int(pts[:, 1].min()))
                        x2 = min(w, int(pts[:, 0].max())); y2 = min(h, int(pts[:, 1].max()))
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes

        return boxes

    def _hand_present(self, frame):
        """True if a hand should short-circuit this frame. If a ROI is set, only
        a hand OVERLAPPING the ROI counts; otherwise any detected hand counts."""
        boxes = self._hand_boxes(frame)
        if not boxes:
            return False
        if self.roi is None:
            return True
        return any(self._rects_overlap(b, self.roi) for b in boxes)

    def _emit_thumbnail(self, event, sid, species, crop):
        b64 = crop_to_base64(crop, self.jpeg_quality)
        if self.save_local and self.thumbnail_dir and crop is not None and crop.size > 0:
            fname = f"{species}_{sid}_{event}_frame{self.frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(self.thumbnail_dir, fname), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        return {"id": str(sid), "species": species, "event": event, "thumbnail": b64}

    def _update_slot(self, did, crop, xyxy, emb=None):
        self.display_info[did]["last_crop"] = crop
        self.display_info[did]["last_hist"] = color_hist(crop)
        self.display_info[did]["last_xyxy"] = xyxy_to_xywh(xyxy)
        self.display_info[did]["gone_frames"] = 0
        self.display_info[did]["present_now"] = True
        # store the latest deep embedding (used by the matcher next frame and
        # read by a FrameStore logger to persist per-frame vectors)
        if emb is not None:
            self.display_info[did]["emb"] = emb
        # Maintain a rolling appearance template (EMA of the signature) so
        # re-identification after occlusion matches a STABLE signature rather
        # than a single possibly-blurred last frame. Improves wrong-id cases.
        sig = appearance_signature(crop)
        self.display_info[did]["last_sig"] = sig
        prev = self.display_info[did].get("sig_ema")
        if sig is not None:
            if prev is None:
                self.display_info[did]["sig_ema"] = sig
            else:
                a = 0.7  # weight on the existing template
                ema = dict(prev)
                if prev.get("hist") is not None and sig.get("hist") is not None:
                    ema["hist"] = a * prev["hist"] + (1 - a) * sig["hist"]
                if prev.get("lab") is not None and sig.get("lab") is not None:
                    ema["lab"] = a * prev["lab"] + (1 - a) * sig["lab"]
                ema["ar"] = a * prev.get("ar", sig["ar"]) + (1 - a) * sig["ar"]
                self.display_info[did]["sig_ema"] = ema

    def _collect_tracked(self, r):
        out = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxys = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else [-1] * len(clss))
            for xyxy, c, cf, tid in zip(xyxys, clss, confs, ids):
                out.append((xyxy, int(c), float(cf), int(tid)))
        return out

    def _order_by_position(self, items, frame_h):
        tol = frame_h * self.row_tol_frac
        arr = [(center(xyxy), tid, xyxy) for tid, xyxy in items]
        arr.sort(key=lambda it: it[0][1])
        rows = []
        for c, tid, xyxy in arr:
            placed = False
            for row in rows:
                if abs(row[0] - c[1]) <= tol:
                    row[1].append((c, tid, xyxy)); placed = True; break
            if not placed:
                rows.append([c[1], [(c, tid, xyxy)]])
        ordered = []
        for _, ritems in rows:
            ritems.sort(key=lambda it: it[0][0])
            ordered.extend([(tid, xyxy) for _, tid, xyxy in ritems])
        return ordered

    def _lock_anchor(self):
        _, _, frame, ducks, others = self.warmup_best
        h = frame.shape[0]
        duck_items = [(tid, xyxy) for xyxy, tid in ducks]
        for did, (tid, xyxy) in enumerate(self._order_by_position(duck_items, h), start=1):
            self.tid_to_display[tid] = did
            crop = get_crop(frame, xyxy)
            sig = appearance_signature(crop)
            emb0 = self._embed(crop)   # deep template at lock (None if off)
            self.display_info[did] = {"last_crop": crop,
                                      "last_hist": color_hist(crop),
                                      "last_sig": sig,
                                      "sig_ema": sig,
                                      "emb": emb0,
                                      "last_xyxy": xyxy_to_xywh(xyxy),
                                      "gone_frames": 0,
                                      "present": True}
        self.num_anchor = len(self.tid_to_display)
        self.next_display_id = self.num_anchor + 1

        other_items = [(tid, xyxy) for xyxy, tid in others]
        for oid, (tid, xyxy) in enumerate(self._order_by_position(other_items, h), start=1):
            self.otid_to_display[tid] = oid
            crop = get_crop(frame, xyxy)
            self.other_info[oid] = {"last_crop": crop,
                                    "last_hist": color_hist(crop),
                                    "last_xyxy": xyxy_to_xywh(xyxy),
                                    "gone_frames": 0}
        self.num_other_anchor = len(self.otid_to_display)
        self.next_other_display = self.num_other_anchor + 1
        self.anchor_locked = True

    def process_frame(self, frame):
        self.frame_idx += 1
        if self.diag is None:
            self.diag = float(np.hypot(frame.shape[1], frame.shape[0]))

        now = time.time()
        fps = 0.0
        if self._last_time is not None:
            dt = now - self._last_time
            fps = round(1.0 / dt, 1) if dt > 0 else 0.0
        self._last_time = now

        # ============================================================== #
        #  DETECT FIRST, THEN GATE THE HAND SHORT-CIRCUIT ON THE TRAY     #
        # ============================================================== #
        # The requirement: skip a frame only when a hand is INSIDE the tray,
        # where the tray region comes from the model's own tray detection (no
        # hand-drawn ROI). That means we must run detection BEFORE the hand
        # decision, because the tray box is a detection output. So the order is:
        #   1. track()            -> all detections incl. the tray
        #   2. pick the tray box  -> _pick_tray_box (coasts if briefly missing)
        #   3. hand-in-tray test  -> a hand only counts if it OVERLAPS the tray;
        #                            if NO tray is found this frame, fall back to
        #                            "any hand anywhere skips" (safe for QC).
        try:
            r = self.model.track(source=frame, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, persist=True,
                                 tracker=self.tracker, device=self._device_str,
                                 verbose=False)[0]
        except Exception as track_err:
            if self._device_str != "cpu":
                print(f"[GPU] Tracker execution on {self._device_str} failed: {track_err}. Falling back to CPU.")
                self._device_str = "cpu"
                self.use_half = False
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                r = self.model.track(source=frame, conf=self.conf, iou=self.iou,
                                     imgsz=self.imgsz, persist=True,
                                     tracker=self.tracker, device="cpu",
                                     verbose=False)[0]
            else:
                raise
        dets = self._collect_tracked(r)

        # tray box for BOTH the hand gate and the object gate below
        tray_box = self._pick_tray_box(dets)

        # ---- hand short-circuit, gated by the tray ----
        hand_boxes = self._hand_boxes(frame)
        if hand_boxes:
            if tray_box is not None:
                # tray known -> a hand counts only if its box overlaps the tray
                # (tray_box is x1,y1,x2,y2; _rects_overlap wants x,y,w,h)
                tx1, ty1, tx2, ty2 = tray_box
                tray_xywh = (int(tx1), int(ty1),
                             int(tx2 - tx1), int(ty2 - ty1))
                hand_now = any(self._rects_overlap(b, tray_xywh)
                               for b in hand_boxes)
            else:
                # no tray detected this frame -> fall back to any-hand-skips
                hand_now = True
        else:
            hand_now = False

        if hand_now:
            self._hand_hold = self.hand_hold_frames
        elif self._hand_hold > 0:
            self._hand_hold -= 1

        if hand_now or self._hand_hold > 0:
            annotated = frame.copy()
            # draw the tray outline (if any) and the hand banner
            if tray_box is not None:
                tx1, ty1, tx2, ty2 = [int(v) for v in tray_box]
                cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), ORANGE, 2)
            draw_banner(annotated, "HAND DETECTED", ORANGE)
            return self._finish(
                annotated, status="HAND", detected=0, fps=fps,
                hand_detected=True,
                missing_ids=[], added_ids=[], other_ids=[],
                detections=[], thumbnails=[], reasons=[])

        # -------- TRAY GATE (object filtering) --------
        # Keep only detections whose CENTER is inside the detected tray, so all
        # downstream logic (anchor, matching, counting, missing) sees tray-only
        # objects automatically. Disabled when tray_class_id < 0. Reuses the
        # SAME tray_box computed above (no second _pick_tray_box call, which
        # would double-advance the coast counter).
        if tray_box is not None:
            dets = [(xyxy, c, cf, tid) for (xyxy, c, cf, tid) in dets
                    if c == self.tray_cls or self._center_in_box(xyxy, tray_box)]

        ducks = [(xyxy, tid) for xyxy, c, cf, tid in dets if c == self.duck_cls and tid >= 0]
        others = [(xyxy, tid) for xyxy, c, cf, tid in dets if c == self.other_cls and tid >= 0]
        duck_conf = {tid: cf for xyxy, c, cf, tid in dets if c == self.duck_cls}
        other_conf = {tid: cf for xyxy, c, cf, tid in dets if c == self.other_cls}
        detected = len(ducks)

        annotated = frame.copy()
        thumbnails = []

        # draw the tray outline (thin cyan) so you can see the gate region
        if tray_box is not None:
            tx1, ty1, tx2, ty2 = [int(v) for v in tray_box]
            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (255, 255, 0), 1,
                          cv2.LINE_AA)


        # ---- warmup ----
        if not self.anchor_locked:
            # FIX: seed the smoothing buffer DURING warmup too, so by the time
            # we lock the anomaly decision already has a full window to vote on.
            # Without this the buffer only starts filling after lock, forcing a
            # spurious NORMAL for `anomaly_smoothing_frames` frames before the
            # real too_many/too_few verdict can appear.
            self.count_history.append(detected)
            score = -abs(detected - self.expected)
            if self.warmup_best is None or score > self.warmup_best[0] \
               or (score == self.warmup_best[0] and detected > self.warmup_best[1]):
                self.warmup_best = (score, detected, frame.copy(), list(ducks), list(others))
            self.warmup_count += 1
            lock_now = False
            if self.warmup_require_exact:
                if detected == self.expected and self.warmup_count >= self.warmup_frames:
                    lock_now = True
                elif self.warmup_count >= self.warmup_max_frames:
                    lock_now = True
            else:
                if self.warmup_count >= self.warmup_frames:
                    lock_now = True
            if lock_now:
                self._lock_anchor()
                for did in range(1, self.num_anchor + 1):
                    if did not in self.confirmed_sent:
                        self.confirmed_sent.add(did)
                        thumbnails.append(self._emit_thumbnail(
                            "confirmed", did, "duck",
                            self.display_info[did]["last_crop"]))

            # FIX (Issue 2): populate detections during WARMING so the frontend
            # can render provisional boxes on its own canvas instead of only
            # having them burned into annotated_frame. id=-1 + provisional flag.
            warmup_detections = [
                {"bbox": [int(v) for v in xyxy],
                 "id": -1,
                 "species": "duck",
                 "confidence": round(float(duck_conf.get(tid, 0.0)), 4),
                 "provisional": True}
                for xyxy, tid in ducks
            ]

            for xyxy, tid in ducks:
                draw_box(annotated, xyxy, "duck ?", ORANGE)
            warm_status = "WARMING"
            if others:
                warm_status = "ANOMALY"
                for xyxy, tid in others:
                    draw_box(annotated, xyxy, "other", RED)
            draw_banner(annotated, warm_status,
                        RED if warm_status == "ANOMALY" else ORANGE)
            # If we just locked on this same frame, fall through so the locked
            # pipeline (below) runs on the next frame; here we still return the
            # warmup payload but with anchor_locked now reflecting reality.
            return self._finish(
                annotated, status=warm_status, detected=detected, fps=fps,
                hand_detected=False,
                missing_ids=[], added_ids=[], other_ids=[],
                detections=warmup_detections, thumbnails=thumbnails,
                reasons=(["other_species_present"] if others else []))

        # ---- map tracker ids -> display ids ----
        present_display = {}
        for d in self.display_info:
            self.display_info[d]["present_now"] = False

        used_slots = set()
        det_list = [(xyxy, tid, get_crop(frame, xyxy)) for xyxy, tid in ducks]

        pending = []
        for xyxy, tid, crop in det_list:
            did = self.tid_to_display.get(tid)
            if did is not None and did in self.display_info and did not in used_slots:
                # SWAP-GUARD: this is a direct tracker-id hit, but before we
                # trust it, check the crop still LOOKS like the duck this slot
                # holds. If the embedder is on, a guard threshold is set, and
                # the deep-embedding agreement is clearly too low, the tracker
                # has likely swapped this id onto the wrong duck -> demote to
                # pending so the appearance-aware rebind can reassign it. We
                # precompute the crop embedding here so it is reused (not
                # recomputed) whether the detection stays or is demoted.
                demb_hit = self._embed(crop)
                if (self.embedder is not None and self.swap_verify_min_app > 0.0
                        and demb_hit is not None
                        and self.display_info[did].get("emb") is not None):
                    agree = self.embedder.cosine(
                        demb_hit, self.display_info[did]["emb"])
                    if agree < self.swap_verify_min_app:
                        if self.debug:
                            print(f"[DBG f{self.frame_idx}] SWAP-GUARD demote "
                                  f"tid={tid} claimed slot #{did} "
                                  f"agree={agree:.3f} < {self.swap_verify_min_app} "
                                  f"-> pending")
                        # The tracker id is lying about identity; drop the stale
                        # tid->did binding so it is not re-trusted next frame,
                        # and push this detection through the rebind matcher.
                        self.tid_to_display.pop(tid, None)
                        pending.append((xyxy, tid, crop))
                        continue
                used_slots.add(did)
                present_display[did] = (xyxy, duck_conf.get(tid, 0.0))
                # direct tracker-id hit accepted -> refresh slot + its embedding
                self._update_slot(did, crop, xyxy, emb=demb_hit)
            else:
                pending.append((xyxy, tid, crop))

        for did in used_slots:
            self.reclaim_candidates.pop(did, None)

        added_ids_this_frame = []
        unbound = []
        matched_this_frame = set()

        # ---- GLOBAL rebind assignment (fixes wrong-id-after-occlusion) ----
        # Instead of letting each pending duck greedily grab its own best free
        # slot (order-dependent -> one duck steals a slot another duck matched
        # far better), we score EVERY pending duck against EVERY free slot and
        # solve one optimal assignment that maximizes total similarity. Score
        # uses the richer signature (hist+Lab+aspect) EMA template + position,
        # and -- when the embedder is on -- a deep-embedding cosine blended in.
        free_slots = [d for d in self.display_info if d not in used_slots]
        # precompute per-pending cues (now including a deep embedding, or None)
        pend_meta = []
        for xyxy, tid, crop in pending:
            pend_meta.append((xyxy, tid, crop, center(xyxy),
                              appearance_signature(crop), self._embed(crop)))

        sim_matrix = np.zeros((len(pend_meta), len(free_slots)), dtype=float)
        app_matrix = np.zeros((len(pend_meta), len(free_slots)), dtype=float)

        # Precompute each free slot's last-known center so we can measure how
        # crowded the neighbourhood of a candidate match is.
        slot_centers = {}
        for d in free_slots:
            lx = self.display_info[d].get("last_xyxy")
            if lx:
                slot_centers[d] = np.array([lx[0] + lx[2] / 2.0,
                                            lx[1] + lx[3] / 2.0])
        crowd_dist = self.crowd_dist_frac * self.diag

        for pi, (xyxy, tid, crop, dc, dsig, demb) in enumerate(pend_meta):
            for sj, d in enumerate(free_slots):
                info = self.display_info[d]
                # appearance against the rolling template (fallback to last),
                # then blend the deep-embedding cosine in when available.
                ap = signature_sim(dsig, info.get("sig_ema") or info.get("last_sig"))
                ap = self._blend_app(ap, demb, info.get("emb"))
                app_matrix[pi, sj] = ap
                lx = info.get("last_xyxy")
                if lx:
                    lc = np.array([lx[0] + lx[2] / 2.0, lx[1] + lx[3] / 2.0])
                    ps = position_sim(dc, lc, self.diag)
                else:
                    lc = None
                    ps = 0.0

                # crowded? -> another free slot sits within crowd_dist of THIS
                # slot, so position alone can't disambiguate. Lean on appearance.
                crowded = False
                if lc is not None:
                    for d2, c2 in slot_centers.items():
                        if d2 != d and np.linalg.norm(lc - c2) < crowd_dist:
                            crowded = True
                            break
                if crowded:
                    aw = self.rebind_app_weight_crowded
                    pw = 1.0 - aw
                else:
                    aw = self.rebind_app_weight
                    pw = self.rebind_pos_weight
                sim_matrix[pi, sj] = aw * ap + pw * ps

        # solve: Hungarian minimizes cost, so cost = -similarity
        assigned_slot = {}   # pending index -> slot did (best global match)
        assigned_score = {}  # pending index -> that match's similarity
        assigned_app = {}    # pending index -> that match's raw appearance sim
        if len(pend_meta) and len(free_slots):
            rows, cols = _hungarian(-sim_matrix)
            for pi, sj in zip(rows, cols):
                assigned_slot[int(pi)] = free_slots[int(sj)]
                assigned_score[int(pi)] = float(sim_matrix[int(pi), int(sj)])
                assigned_app[int(pi)] = float(app_matrix[int(pi), int(sj)])

        for pi, (xyxy, tid, crop, dc, dsig, demb) in enumerate(pend_meta):
            best_did = assigned_slot.get(pi)
            best_score = assigned_score.get(pi, -1.0)
            best_app = assigned_app.get(pi, 0.0)
            if self.debug:
                _crowded = False
                if best_did is not None and best_did in slot_centers:
                    _lc = slot_centers[best_did]
                    for _d2, _c2 in slot_centers.items():
                        if _d2 != best_did and np.linalg.norm(_lc - _c2) < crowd_dist:
                            _crowded = True
                            break
                print(f"[DBG f{self.frame_idx}] pending tid={tid} -> slot #{best_did} "
                      f"score={best_score:.3f} app={best_app:.3f} crowded={_crowded} "
                      f"missing_active={best_did in self.missing_active}")

            # Two different situations need different strictness:
            #  * Reclaiming a slot whose duck is in a MISSING episode -> be
            #    strict: also require appearance agreement, so a different duck
            #    drifting over the gap can't silently steal the slot and mask
            #    the loss (that caused the false NORMAL).
            #  * An ordinary free slot (its duck merely MOVED, not missing) ->
            #    the blended score threshold alone is enough. Do NOT also demand
            #    the appearance gate here, or a duck that changed pose/lighting
            #    gets wrongly orphaned to id:-1 while its slot goes MISSING --
            #    which shows up as a "duck" box with no id AND a false ANOMALY
            #    even though every duck is present.
            reject = (best_did is None or best_score < self.rebind_threshold)
            if not reject and best_did in self.missing_active:
                if best_app < self.rebind_min_app:
                    reject = True

            if reject:
                if self.strict_count:
                    unbound.append((xyxy, duck_conf.get(tid, 0.0)))
                else:
                    self.prov_new[tid] = self.prov_new.get(tid, 0) + 1
                    if self.prov_new[tid] >= self.new_id_patience:
                        did = self.next_display_id; self.next_display_id += 1
                        self.tid_to_display[tid] = did
                        _sig = appearance_signature(crop)
                        self.display_info[did] = {"last_crop": crop,
                                                  "last_hist": color_hist(crop),
                                                  "last_sig": _sig,
                                                  "sig_ema": _sig,
                                                  "emb": demb,
                                                  "last_xyxy": xyxy_to_xywh(xyxy),
                                                  "gone_frames": 0, "present": True}
                        del self.prov_new[tid]
                        present_display[did] = (xyxy, duck_conf.get(tid, 0.0))
                        added_ids_this_frame.append(did)
                continue

            if best_did in self.missing_active:
                cand = self.reclaim_candidates.get(best_did)
                if cand is not None and cand["tid"] == tid:
                    cand["count"] += 1
                else:
                    cand = {"tid": tid, "count": 1}
                    self.reclaim_candidates[best_did] = cand
                matched_this_frame.add(best_did)
                if cand["count"] >= self.reappear_patience:
                    used_slots.add(best_did)
                    self.tid_to_display[tid] = best_did
                    present_display[best_did] = (xyxy, duck_conf.get(tid, 0.0))
                    self._update_slot(best_did, crop, xyxy, emb=demb)
                    del self.reclaim_candidates[best_did]
            else:
                used_slots.add(best_did)
                self.tid_to_display[tid] = best_did
                present_display[best_did] = (xyxy, duck_conf.get(tid, 0.0))
                self._update_slot(best_did, crop, xyxy, emb=demb)

        for did in list(self.reclaim_candidates.keys()):
            if did not in matched_this_frame:
                del self.reclaim_candidates[did]

        # ---- shake-smoothed anomaly ----
        # NOTE: the buffer is already seeded during warmup, so at lock it holds
        # a full window. We no longer wait for buffer_ready before deciding --
        # we vote on whatever is in the window (min a couple of frames) so a
        # genuine count mismatch is flagged immediately after lock instead of
        # showing a spurious NORMAL for `anomaly_smoothing_frames` frames.
        self.count_history.append(detected)
        smoothed = Counter(self.count_history).most_common(1)[0][0]
        reasons = []
        if len(self.count_history) >= min(3, self.count_history.maxlen):
            if smoothed < self.expected:
                reasons.append("too_few_ducks")
            if smoothed > self.expected:
                reasons.append("too_many_ducks")

        # ---- added thumbnails (once each) ----
        for did in added_ids_this_frame:
            if did not in self.confirmed_sent:
                self.confirmed_sent.add(did)
                thumbnails.append(self._emit_thumbnail(
                    "added", did, "duck", self.display_info[did]["last_crop"]))

        # ---- missing: anchor id not present this frame ----
        # FIX (Issue 3): report the id in missing_ids AS SOON AS it is absent
        # (gone_frames >= 1), not only after missing_patience. missing_patience
        # now only gates the missing_active episode / frozen box, so the id no
        # longer vanishes from the payload during the patience window.
        missing_ids = []
        for did in range(1, self.num_anchor + 1):
            if did in present_display:
                self.display_info[did]["gone_frames"] = 0
                self.missing_active.discard(did)
                self.display_info[did].pop("frozen_xyxy", None)
            else:
                self.display_info[did]["gone_frames"] = \
                    self.display_info[did].get("gone_frames", 0) + 1
                # report immediately so the UI card never disappears
                missing_ids.append(did)
                if self.display_info[did]["gone_frames"] >= self.missing_patience:
                    if did not in self.missing_active:
                        self.missing_active.add(did)
                        self.display_info[did]["frozen_xyxy"] = \
                            self.display_info[did].get("last_xyxy")

        # A KNOWN anchor duck being absent is itself an anomaly, regardless of
        # what the shake-smoothed total count says. The count check above can
        # miss this: if one duck vanishes but a spurious box (reflection, tray
        # edge, mis-detection) keeps the TOTAL at `expected`, the count matches
        # and status would wrongly read NORMAL. For a QC pipeline a missing
        # duck must never be NORMAL, so drive the verdict off missing state too.
        #
        # We gate on missing_active (duck gone for >= missing_patience frames)
        # rather than the raw missing_ids, so a duck that is only BRIEFLY hidden
        # (behind a hand / another duck) and comes right back does not flash a
        # false ANOMALY. The id is still reported in missing_ids immediately;
        # only the anomaly VERDICT waits for the patience window.
        if self.missing_active and "missing_duck" not in reasons:
            reasons.append("missing_duck")

        # ---- other species (separate id space, thumbnail once) ----
        present_other = {}
        other_ids = []
        used_oslots = set()
        matched_prov_keys = set()
        for xyxy, tid in others:
            crop = get_crop(frame, xyxy)
            dc = center(xyxy); dh = color_hist(crop)
            best_oid, best_score = None, -1.0
            for o in range(1, self.num_other_anchor + 1):
                if o in used_oslots or o not in self.other_info:
                    continue
                info = self.other_info[o]
                ap = appearance_sim(dh, info.get("last_hist"))
                lx = info.get("last_xyxy")
                if lx:
                    lc = np.array([lx[0] + lx[2] / 2.0, lx[1] + lx[3] / 2.0])
                    ps = position_sim(dc, lc, self.diag)
                else:
                    ps = 0.0
                sc = self.rebind_app_weight * max(0.0, ap) + self.rebind_pos_weight * ps
                if sc > best_score:
                    best_score, best_oid = sc, o

            if best_oid is not None and best_score >= self.rebind_threshold:
                oid = best_oid
                self.other_info[oid]["last_crop"] = crop
                self.other_info[oid]["last_hist"] = color_hist(crop)
                self.other_info[oid]["last_xyxy"] = xyxy_to_xywh(xyxy)
                used_oslots.add(oid)
                present_other[oid] = (xyxy, other_conf.get(tid, 0.0))
                other_ids.append(oid)
                if oid not in self.other_sent:
                    self.other_sent.add(oid)
                    thumbnails.append(self._emit_thumbnail(
                        "other_present", oid, "other_toys", crop))
            else:
                key = tid if tid >= 0 else (int(dc[0] // 40), int(dc[1] // 40))
                prov = self._prov_other.get(key)
                if prov is not None and prov.get("tid") == tid:
                    prov["count"] += 1
                else:
                    prov = {"tid": tid, "count": 1}
                    self._prov_other[key] = prov
                prov["last"] = (xyxy, crop, other_conf.get(tid, 0.0))
                matched_prov_keys.add(key)

                if prov["count"] >= self.other_id_patience:
                    oid = self.next_other_display
                    self.next_other_display += 1
                    self.other_info[oid] = {"last_crop": crop,
                                            "last_hist": color_hist(crop),
                                            "last_xyxy": xyxy_to_xywh(xyxy),
                                            "gone_frames": 0}
                    if oid > self.num_other_anchor:
                        self.num_other_anchor = oid
                    del self._prov_other[key]
                    used_oslots.add(oid)
                    present_other[oid] = (xyxy, other_conf.get(tid, 0.0))
                    other_ids.append(oid)
                    if oid not in self.other_sent:
                        self.other_sent.add(oid)
                        thumbnails.append(self._emit_thumbnail(
                            "other_present", oid, "other_toys", crop))

        for key in list(self._prov_other.keys()):
            if key not in matched_prov_keys:
                del self._prov_other[key]

        if len(present_other) > 0:
            reasons.append("other_species_present")

        is_anomaly = len(reasons) > 0
        status = "ANOMALY" if is_anomaly else "NORMAL"

        box_color = RED if is_anomaly else GREEN

        excess_ids = []
        detections = []
        for did, (xyxy, cf) in present_display.items():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            # An excess duck is any duck beyond expected count or beyond anchor count
            is_excess = (self.expected is not None and did > self.expected) or (did > self.num_anchor)
            if is_excess:
                excess_ids.append(int(did))
            this_color = RED if is_anomaly else GREEN
            detections.append({"bbox": [x1, y1, x2, y2], "id": int(did),
                               "species": "duck", "confidence": round(float(cf), 4),
                               "status": "excess" if is_excess else "present",
                               "excess": bool(is_excess),
                               "is_anomaly": bool(is_excess or is_anomaly)})
            lbl = f"#{did} EXCESS {int(cf * 100)}%" if is_excess else f"#{did} {int(cf * 100)}%"
            draw_box(annotated, xyxy, lbl, this_color)

        for xyxy, cf in unbound:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            # An unbound duck (strict_count mode: a detection that matched no
            # slot) is an extra/unexpected duck -> always red, even if the
            # frame is otherwise NORMAL, so the offending duck stands out.
            detections.append({"bbox": [x1, y1, x2, y2], "id": -1,
                               "species": "duck", "confidence": round(float(cf), 4),
                               "status": "unbound", "excess": True,
                               "is_anomaly": True})
            draw_box(annotated, xyxy, "duck", RED)

        for oid, (xyxy, cf) in present_other.items():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            detections.append({"bbox": [x1, y1, x2, y2], "id": int(oid),
                               "species": "other_toys", "confidence": round(float(cf), 4),
                               "status": "present"})
            draw_box(annotated, xyxy, f"other #{oid} {int(cf * 100)}%", box_color)

        # FIX (Issue 1): include missing ducks in detections with their last
        # known box + status="missing", so the frontend keeps the card in its
        # correct ID position and can show the frozen crop, instead of the id
        # only appearing as a bare number in missing_ids.
        for did in missing_ids:
            lx = self.display_info[did].get("frozen_xyxy") \
                or self.display_info[did].get("last_xyxy")
            if lx:
                x, y, w, hh = lx
                detections.append({"bbox": [int(x), int(y), int(x + w), int(y + hh)],
                                   "id": int(did), "species": "duck",
                                   "confidence": 0.0, "status": "missing"})
                # only draw the frozen dashed box once the episode is confirmed
                if did in self.missing_active:
                    draw_box(annotated, [x, y, x + w, y + hh],
                             f"#{did} MISSING", box_color, dashed=True)

        draw_banner(annotated, status, box_color)

        if self.debug:
            print(f"[DBG f{self.frame_idx}] status={status} detected={detected} "
                  f"present={sorted(present_display.keys())} excess={excess_ids} pending={len(pend_meta)} "
                  f"unbound={len(unbound)} missing={missing_ids} reasons={reasons}")
        return self._finish(
            annotated, status=status, detected=detected, fps=fps,
            hand_detected=False,
            missing_ids=missing_ids, added_ids=added_ids_this_frame,
            other_ids=other_ids, detections=detections, thumbnails=thumbnails,
            reasons=reasons, excess_ids=excess_ids)

    def _finish(self, annotated, status, detected, fps, hand_detected,
                missing_ids, added_ids, other_ids, detections, thumbnails,
                reasons=None, excess_ids=None):
        if reasons is None:
            reasons = []
        if excess_ids is None:
            excess_ids = []
        if self.save_local and self.annotated_dir:
            cv2.imwrite(os.path.join(self.annotated_dir,
                        f"frame_{self.frame_idx:05d}.jpg"), annotated)
        return {
            "frame": self.frame_idx,
            "fps": fps,
            "status": status,
            "anchor_locked": self.anchor_locked,   # FIX (Issue 4): expose state
            "reasons": reasons,                    # FIX (Issue 4): expose reasons
            "hand_detected": hand_detected,
            "detected_duck_count": detected,
            "expected_duck_count": self.expected,
            "missing_count": len(missing_ids),
            "missing_ids": missing_ids,
            "added_count": len(added_ids),
            "added_ids": added_ids,
            "excess_count": len(excess_ids),
            "excess_ids": excess_ids,
            "other_count": len(other_ids),
            "detected_other_toy_count": len(other_ids),  # compatibility alias
            "other_ids": other_ids,
            "detections": detections,
            "thumbnails": thumbnails,
            "annotated_frame": annotated,
        }

    def embeddings_for_frame(self, result):
        """Helper for a FrameStore logger: build a {(species, id): vector} map
        for the ducks present in `result` by reading each slot's latest deep
        embedding. Returns {} when the embedder is off. Call right after
        process_frame(), before the next frame overwrites the slot embeddings."""
        embs = {}
        if self.embedder is None:
            return embs
        for det in result.get("detections", []):
            if det.get("species") == "duck" and det.get("id", -1) > 0:
                info = self.display_info.get(det["id"], {})
                if info.get("emb") is not None:
                    embs[("duck", det["id"])] = info["emb"]
        return embs

    def close(self):
        if self._mp_hands is not None:
            try:
                self._mp_hands.close()
            except Exception:
                pass


# ---------------------------------------------------------------------- #
#  Optional singleton: load the model ONCE for the whole process.        #
#  If your backend calls this instead of constructing DuckAnalyzer()     #
#  per request, the weights load a single time and every request reuses  #
#  the same in-memory model.                                             #
# ---------------------------------------------------------------------- #
_SHARED_ANALYZER = None


def get_shared_analyzer(config_path="config.yaml", expected_duck_count=None):
    global _SHARED_ANALYZER
    if _SHARED_ANALYZER is None:
        _SHARED_ANALYZER = DuckAnalyzer(config_path, expected_duck_count)
    elif expected_duck_count is not None:
        _SHARED_ANALYZER.set_expected_duck_count(expected_duck_count)
    return _SHARED_ANALYZER


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    video = sys.argv[2] if len(sys.argv) > 2 else None
    expected = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out_path = sys.argv[4] if len(sys.argv) > 4 else "annotated_output.mp4"
    # optional 5th arg -- folder to save frames where the detected duck count
    # did NOT match `expected` (raw for retraining, annotated for a quick look).
    mismatch_dir = sys.argv[5] if len(sys.argv) > 5 else None
    # optional 6th arg -- SQLite path for per-frame detections + embeddings.
    db_path = sys.argv[6] if len(sys.argv) > 6 else None
    if mismatch_dir:
        os.makedirs(os.path.join(mismatch_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(mismatch_dir, "annotated"), exist_ok=True)

    analyzer = DuckAnalyzer(cfg_path, expected_duck_count=expected)

    # optional per-frame SQLite logger (detections + deep embeddings as BLOBs)
    store = None
    if db_path:
        try:
            try:
                from .frame_store import FrameStore      # installed as a package
            except ImportError:
                from frame_store import FrameStore        # run as a flat script
            store = FrameStore(db_path,
                               run_id=os.path.basename(video or "run"),
                               expected=expected)
        except Exception as e:
            print(f"[DB] [WARN] could not open FrameStore ({db_path}): {e}")
            store = None

    if video:
        cap = cv2.VideoCapture(video)
        # set up the output video writer from the source's size + fps
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), in_fps, (w, h))
        if not writer.isOpened():
            print(f"[WARN] VideoWriter failed to open for: {out_path} "
                  f"-- check the destination folder exists.")
        mismatch_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = analyzer.process_frame(frame)
            # write the annotated frame to the output video
            writer.write(res["annotated_frame"])
            # log this frame's detections + embeddings BEFORE the next frame
            # overwrites the slot embeddings
            if store is not None:
                store.log(res, embeddings=analyzer.embeddings_for_frame(res))
            if res["thumbnails"]:
                print(f"frame {res['frame']} events:",
                      [(t["event"], t["id"]) for t in res["thumbnails"]])
            print(f"frame {res['frame']}: {res['status']}, "
                  f"ducks={res['detected_duck_count']}, "
                  f"missing={res['missing_ids']}, added={res['added_ids']}, "
                  f"other={res['other_ids']}, hand={res['hand_detected']}, "
                  f"fps={res['fps']}")
            # extract frames where the count didn't match expected
            if (mismatch_dir and res["expected_duck_count"] is not None
                    and res["detected_duck_count"] != res["expected_duck_count"]):
                mismatch_count += 1
                fname = (f"frame_{res['frame']:06d}"
                         f"_det{res['detected_duck_count']}"
                         f"_exp{res['expected_duck_count']}.jpg")
                cv2.imwrite(os.path.join(mismatch_dir, "raw", fname), frame)
                cv2.imwrite(os.path.join(mismatch_dir, "annotated", fname),
                            res["annotated_frame"])
        cap.release()
        writer.release()
        if store is not None:
            store.close()
        print(f"done -> saved annotated video to {out_path}")
        if mismatch_dir:
            print(f"done -> saved {mismatch_count} mismatched-count frame(s) to "
                  f"{mismatch_dir} (raw/ and annotated/ subfolders)")
        if db_path:
            print(f"done -> logged per-frame detections + embeddings to {db_path}")
    else:
        print("Loaded. Feed frames via analyzer.process_frame(frame).")
