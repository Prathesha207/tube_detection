"""
main.py - the wrapper your backend calls. No pipeline logic lives here: model
loading / unloading, the per-frame pipeline and video in/out are in
run_video_frames.py. This file only ties them together in one class.

  HeadTailAnalyzer  one object per video stream / session. process_frame(frame)
                    runs the full pipeline for one frame and returns the result dict.
  ModelManager, MODELS, LoadedModel, AnalyzerError, Track,
  load_model / unload_model / is_model_loaded
                    defined in run_video_frames.py, importable from here too.

From a backend:

    from head_tail_analyzer.main import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer("config.yaml")      # nothing is loaded yet
    analyzer.load_model()                           # optional: load now instead of on frame 1
    result = analyzer.process_frame(frame)          # model already loaded -> just inference
    result, picture = analyzer.process_frame_annotated(frame)   # + the drawn frame
    analyzer.show_cards = False                     # boxes only (results stay in the top-left panel)
    analyzer.toggle_cards()                         # flip on/off (e.g. from a frontend button)
    analyzer.toggle_panel()                         # full results in the top-left panel on/off
    result, picture = analyzer.process_frame_annotated(frame, show_cards=True)   # just this frame
    analyzer.unload_model()                         # the ONLY thing that frees it (GPU / RAM)

Trace out of the wrapper (wrapper_trace.py) - the FRONTEND sends the pixel distance:

    # whole video, value from the request (number or text like "150"):
    for result in analyzer.process_video(video, trace_px=request_trace_px):
        for tr in result["traces"]:                 # one per tube end leaving the wrapper
            tr["tube_id"], tr["side"], tr["end_px"], tr["end_xyz_m"]   # X, Y, Z metres
    # user changes the value while the video is running (another request):
    analyzer.set_trace_px(new_value)                # used from the next frame on
    # frame by frame:
    result = analyzer.process_frame(frame, depth=depth_frame, trace_px=request_trace_px)
    0 / None / "" = tracing off.  Bad value (e.g. "abc", -5) -> AnalyzerError.

Rules
  * The model is loaded ONCE per process. Every process_frame() after that only runs inference.
  * NOTHING unloads the model by itself. Only unload_model(); it waits for a prediction
    that is running right now (from any session), then frees the model and the GPU memory.
  * After unload_model(): auto_load=True (default) -> the next process_frame() loads it
    again; auto_load=False -> AnalyzerError until you call load_model() yourself.
  * One analyzer = one video stream / session_id (it remembers the tracks). Two sessions
    -> two analyzers; they share ONE loaded model. reset() starts a new stream.
  * Concurrency: sessions sharing a model take the model's inference lock around
    model.predict, so two simultaneous requests queue on the GPU instead of running
    predict on the same model object at once. Tracking/drawing still run in parallel.
    No extra global lock is needed in the API.
  * config: pass config.yaml (needed - there is no bundled default), or at least
    model_path=... as a keyword. Extra keywords override the file:
        HeadTailAnalyzer("config.yaml", device="cpu", conf=0.4, min_frames=3)
  * Web servers: run 1 worker and no --reload in production. Every worker process
    loads its own copy of the model (about 218 MB each).
  * frame = a BGR image as read by cv2 (gray and BGRA are converted for you).

From the terminal:

    python main.py --config config.yaml --video part_0.avi --output out.mp4
    python main.py --config config.yaml --video part_0.avi --output out.mp4 --preview
        preview keys while it runs: c = result cards on/off | p = panel details on/off | q = quit
    python main.py --config config.yaml --video a.mp4 b.mp4 videos_folder --output results/out.mp4
        -> model loaded ONCE, results/out_a.mp4, results/out_b.mp4, ... one per video
"""
import argparse
import functools
import threading
from pathlib import Path
from typing import Iterator

try:                                   # part of the package: python -m head_tail_analyzer.main
    from . import run_video_frames as rvf
except ImportError:                    # plain script next to run_video_frames.py
    import run_video_frames as rvf

# re-exported so existing imports from main keep working
from_rvf = ("AnalyzerError", "LoadedModel", "ModelManager", "MODELS", "Track",
            "load_model", "unload_model", "is_model_loaded")
AnalyzerError = rvf.AnalyzerError
LoadedModel = rvf.LoadedModel
ModelManager = rvf.ModelManager
MODELS = rvf.MODELS
Track = rvf.Track
load_model = rvf.load_model
unload_model = rvf.unload_model
is_model_loaded = rvf.is_model_loaded

__all__ = ["HeadTailAnalyzer", "requires_model", "parse_trace_px", "main", *from_rvf]


def parse_trace_px(value):
    """Trace distance from the frontend -> float px, or None (= tracing off).
    Accepts 150, 150.0, "150", " 150 px "; 0 / None / "" -> None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower().removesuffix("px").strip()
        if value in ("", "none", "null", "off"):
            return None
    try:
        px = float(value)
    except (TypeError, ValueError):
        raise AnalyzerError(f"trace_px must be a number of pixels, got {value!r}") from None
    if px != px or px < 0 or px == float("inf"):
        raise AnalyzerError(f"trace_px must be 0 or a positive number of pixels, got {value!r}")
    return px or None


def requires_model(fn):
    """Decorator: make sure the model is loaded (only if it is not in memory yet),
    then run the method with it. One analyzer never processes two frames at once."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            entry = self._ensure_model(self.auto_load)
            return fn(self, entry, *args, **kwargs)
    return wrapper


class HeadTailAnalyzer:
    """Wrapper around the pipeline: detect -> track -> measure width -> bigger tube.

    config     path to config.yaml (required unless you pass model_path=...)
    manager    ModelManager to use (default: the shared MODELS)
    preload    True = load the model right now instead of on the first frame
    auto_load  True (default) = process_frame loads the model if it is not loaded;
               False = process_frame raises AnalyzerError until you call load_model()
    use_roi    False = ignore roi_json and use the whole frame
    **overrides  any setting from DEFAULTS in run_video_frames.py (device, conf, tile, ...)
    """

    def __init__(self, config=None, *, manager=None, preload=False, auto_load=True,
                 use_roi=True, **overrides):
        self.cfg, self.roi = rvf.load_stream_config(config, use_roi=use_roi, **overrides)
        self.auto_load = bool(auto_load)
        self.manager = manager or MODELS
        self.state = rvf.StreamState()
        self._lock = threading.RLock()
        self._masks_ok = bool(self.cfg.draw_masks)     # user's choice; off for non-seg models
        self._configured_serial = None
        if preload:
            self.load_model()

    # ---------------------------------------------------------- model control
    def load_model(self):
        """Load the model now. Does nothing if it is already loaded."""
        with self._lock:
            self._ensure_model(allow_load=True)
        return self

    def unload_model(self):
        """Free the model and its GPU memory. Affects every analyzer sharing this model
        file. Tracks are kept (reset() forgets them)."""
        with self._lock:
            return self.manager.unload(self.cfg.model_path)

    load = load_model                  # old names kept so existing code keeps working
    unload = unload_model

    @property
    def is_loaded(self):
        return self.manager.is_loaded(self.cfg.model_path)

    @property
    def model_load_count(self):
        """How many times a model was really loaded by this manager (1 = loaded once)."""
        return self.manager.load_count

    def _ensure_model(self, allow_load=True):
        if not allow_load and not self.is_loaded:
            raise AnalyzerError("the model is not loaded - call analyzer.load_model() first "
                                "(this analyzer was created with auto_load=False)")
        entry = self.manager.get(self.cfg.model_path, device=self.cfg.device,
                                 imgsz=int(self.cfg.imgsz or self.cfg.tile or 640))
        if entry.serial != self._configured_serial:     # new load -> adapt the settings once
            if not rvf.configure_for_model(self.cfg, entry):
                self._masks_ok = False
            self._configured_serial = entry.serial
        return entry

    # ---------------------------------------------------------------- state
    def reset(self):
        """Forget all tracks and counters (new video / new stream). Does NOT unload the model."""
        with self._lock:
            self.state.reset()

    @property
    def tracks(self):
        return self.state.tracks

    @property
    def frame_index(self):
        return self.state.frame_index

    def totals(self):
        """Distinct confirmed heads / tails that appeared in the ROI so far."""
        return self.state.totals()

    def diagnosis(self):
        """Print every track (confirmed / flicker) - handy after a video."""
        rvf.print_diagnosis(self.state.tracks, self.cfg.min_frames, self.cfg.mm_per_px)

    # ------------------------------------------------------------ result cards
    @property
    def show_cards(self):
        """True = result card next to every box; False = boxes only (details in the panel)."""
        return bool(self.cfg.show_cards)

    @show_cards.setter
    def show_cards(self, on):
        with self._lock:
            self.cfg.show_cards = bool(on)

    def toggle_cards(self):
        """Flip the result cards on/off. Returns the new state."""
        with self._lock:
            self.cfg.show_cards = not self.cfg.show_cards
            return self.cfg.show_cards

    @property
    def panel_details(self):
        """True = every end + bigger tube listed in the top-left panel."""
        return bool(self.cfg.panel_details)

    @panel_details.setter
    def panel_details(self, on):
        with self._lock:
            self.cfg.panel_details = bool(on)

    # ------------------------------------------------------- trace distance
    @property
    def trace_px(self):
        """Pixels to follow each tube out of the wrapper (None = tracing off)."""
        return self.cfg.trace_px

    @trace_px.setter
    def trace_px(self, px):
        px = parse_trace_px(px)
        with self._lock:
            if px != self.cfg.trace_px:
                self.state.trace_hist.clear()    # new distance -> old end points no longer apply
            self.cfg.trace_px = px

    def set_trace_px(self, px):
        """For the API: the value the frontend sent (number or text). Takes effect from
        the next frame, also while process_video() is running. Returns the value used."""
        self.trace_px = px
        return self.cfg.trace_px

    def toggle_panel(self):
        """Flip the panel details on/off. Returns the new state."""
        with self._lock:
            self.cfg.panel_details = not self.cfg.panel_details
            return self.cfg.panel_details

    def redraw_last(self, show_cards=None):
        """Draw the last annotated frame again with the current settings (no inference)."""
        ld = self.state.last_draw
        if ld is None:
            return None
        frame, dets, result, roi = ld
        return rvf.draw(frame, dets, result, roi, self.cfg, show_cards=show_cards)

    # ------------------------------------------------------------- per frame
    def _with_trace(self, trace_px, fn):
        """Run fn() with trace_px set for this call only (None = keep the current value)."""
        if trace_px is None:
            return fn()
        px = parse_trace_px(trace_px)
        saved = self.cfg.trace_px
        if px != saved:
            self.state.trace_hist.clear()
        self.cfg.trace_px = px
        try:
            return fn()
        finally:
            self.cfg.trace_px = saved

    @requires_model
    def process_frame(self, entry, frame, frame_index=None, depth=None, trace_px=None):
        """Run the full pipeline on one frame and return the result dict
        (heads_count, tails_count, detections, bigger_tube, frame_total, fps, ...).
        depth: optional uint16 depth frame (mm) aligned to `frame` -> each detection
        gets xyz_m / z_m / width_mm_depth (needs camera_matrix in the config).
        trace_px: follow each tube this many px out of the wrapper, for this frame only
        (None = analyzer.trace_px) -> result["traces"], result["wrapper"]."""
        return self._with_trace(trace_px, lambda: rvf.analyze_frame(
            entry, frame, self.state, self.roi, self.cfg, frame_index, annotate=False,
            depth=depth)[0])

    @requires_model
    def process_frame_annotated(self, entry, frame, frame_index=None, depth=None, show_cards=None,
                                trace_px=None):
        """Same as process_frame, returns (result dict, annotated BGR frame).
        show_cards: None = current setting (analyzer.show_cards); True/False = this frame only."""
        return self._with_trace(trace_px, lambda: rvf.analyze_frame(
            entry, frame, self.state, self.roi, self.cfg, frame_index, annotate=True,
            masks=self._masks_ok, depth=depth, show_cards=show_cards))

    # ----------------------------------------------------------------- video
    def process_video(self, video=None, output=None, stride=None, preview=False,
                      depth_video=None, trace_px=None) -> Iterator[dict]:
        """Generator: one result dict per processed frame of a video (or webcam "0").
        `output` = annotated .mp4 to write. The model is NOT unloaded when the video ends.
        `depth_video` = aligned 16-bit depth video (default: depth_video from the config;
        "auto" = the <name>_depth.mkv next to <name>_video.mp4).
        `trace_px` = the distance the frontend sent (None = keep analyzer.trace_px). It stays
        set after the video; change it while running with analyzer.set_trace_px()."""
        if trace_px is not None:
            self.trace_px = trace_px
        video = video or self.cfg.video
        if not video:
            raise AnalyzerError("no video: pass video=... or set it in the config")
        output = output or self.cfg.output_video
        stride = max(int(stride or self.cfg.stride), 1)
        cap, fps, total = rvf.open_video(video)
        self.state.frame_total, self.state.video_fps = total, round(fps, 2)
        saved_depth = self.cfg.depth_video
        if depth_video is not None:
            self.cfg.depth_video = depth_video
        try:
            depth_reader = rvf.depth3d.open_depth(self.cfg, fps, video=video)
        except RuntimeError as e:
            cap.release()
            raise AnalyzerError(str(e)) from None
        finally:
            self.cfg.depth_video = saved_depth
        out = rvf.VideoOutput(output, fps / stride, preview) if (output or preview) else None
        if out:
            out.on_key["c"] = self.toggle_cards          # preview window: c = cards on/off
            out.on_key["p"] = self.toggle_panel          #                 p = panel details on/off
        src = rvf.FrameSource(cap, depth_reader, stride, getattr(self.cfg, "prefetch", True))
        n = 0
        try:
            for idx, frame, depth in src:          # next frames are decoded in the background
                n += 1
                if out:
                    result, vis = self.process_frame_annotated(frame, frame_index=idx, depth=depth)
                    if not out.write(vis, redraw=self.redraw_last if preview else None):
                        break
                else:
                    result = self.process_frame(frame, frame_index=idx, depth=depth)
                yield result
                if idx % 10 == 0:
                    print(f"[progress] frame {idx}/{total or '?'}  "
                          f"{result.get('fps') or 0:.1f} frames/s", flush=True)
                if getattr(self.cfg, "profile", False) and n % 30 == 0:
                    print(rvf.PROF.report(), flush=True)
        finally:
            src.close()
            cap.release()
            if depth_reader:
                rvf.report_depth_sync(depth_reader, src.frames_read)
                depth_reader.close()
            if out:
                out.close()


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Count heads and tails in a video, measure widths, find the bigger tube.",
        epilog="Any flag overrides the same setting in the config file.")
    ap.add_argument("--config", help="config.yaml (paths inside it are relative to its folder)")
    ap.add_argument("--video", nargs="+",
                    help="one or more video files and/or folders of videos (or 0 for webcam). "
                         "The model is loaded ONCE and reused for all of them")
    ap.add_argument("--model", help="trained YOLO11-seg weights (.pt)")
    ap.add_argument("--output", help="annotated video, e.g. out.mp4")
    ap.add_argument("--json", help="latest-frame JSON, overwritten every frame")
    ap.add_argument("--history", help="JSONL with one line per frame")
    ap.add_argument("--roi", help="roi.json polygon")
    ap.add_argument("--no-roi", action="store_true", help="ignore any ROI, use the whole frame")
    ap.add_argument("--device", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--stride", type=int, help="process every Nth frame")
    ap.add_argument("--mm-per-px", type=float, help="millimetres per pixel, for widths in mm")
    ap.add_argument("--depth", help="aligned 16-bit depth video for --video, or 'auto' "
                                    "(<name>_depth.mkv next to each <name>_video.mp4)")
    ap.add_argument("--no-depth", action="store_true", help="ignore depth_video from the config")
    ap.add_argument("--trace-px", type=float,
                    help="follow each tube this many px out of the wrapper -> X/Y/Z of the end")
    ap.add_argument("--half", action="store_true", help="FP16 inference (faster on GPU)")
    ap.add_argument("--profile", action="store_true", help="print ms per stage every 30 frames")
    ap.add_argument("--preview", action="store_true",
                    help="show a live window. Keys: c = result cards on/off, "
                         "p = panel details on/off, q = quit")
    ap.add_argument("--no-cards", action="store_true",
                    help="boxes only, no result cards (results stay in the top-left panel)")
    ap.add_argument("--quiet", action="store_true", help="don't print every frame")
    a = ap.parse_args(argv)

    try:
        analyzer = HeadTailAnalyzer(
            a.config, use_roi=not a.no_roi, model_path=a.model,
            output_video=a.output, latest_json=a.json, history_jsonl=a.history, roi_json=a.roi,
            device=a.device, conf=a.conf, stride=a.stride, mm_per_px=a.mm_per_px,
            depth_video=a.depth, print_frames=False if a.quiet else None,
            show_cards=False if a.no_cards else None, trace_px=a.trace_px,
            half=True if a.half else None, profile=True if a.profile else None)
        if a.no_depth:
            analyzer.cfg.depth_video = None
        videos = expand_videos(a.video) if a.video else [analyzer.cfg.video]
        if not videos or videos == [None]:
            raise AnalyzerError("no video: pass --video (files or folders) or set video in the config")
        out_base, hist_base = analyzer.cfg.output_video, analyzer.cfg.history_jsonl
        many = len(videos) > 1
        if many and analyzer.cfg.depth_video and analyzer.cfg.depth_video.lower() != "auto":
            print("[depth] several videos but ONE depth_video - using depth_video=auto instead "
                  "(each <name>_video.mp4 is paired with its own <name>_depth.mkv)")
            analyzer.cfg.depth_video = "auto"

        analyzer.load_model()                            # loaded ONCE for every video below
        summary = []
        for i, video in enumerate(videos, 1):
            print(f"\n===== video {i}/{len(videos)}: {video} =====")
            analyzer.reset()                             # new video -> fresh tracks; model stays loaded
            out = per_video(out_base, video) if many else out_base
            analyzer.cfg.history_jsonl = per_video(hist_base, video) if many else hist_base
            n = sum(1 for _ in analyzer.process_video(video=video, output=out, preview=a.preview))
            analyzer.diagnosis()
            summary.append((video, n, analyzer.totals(), out, analyzer.cfg.history_jsonl))
        # no unload_model() here: the process ends next, which frees the model anyway.
        # In a backend, keep the analyzer alive and never call unload_model() between videos.
    except AnalyzerError as e:
        raise SystemExit(f"error: {e}")

    print(f"\n[done] {len(summary)} video(s), model loaded {analyzer.model_load_count}x")
    for video, n, totals, out, hist in summary:
        print(f"  {video}: {n} frames  {totals}")
        for label, p in (("video", out), ("history", hist)):
            if p:
                print(f"      {label}: {p}")
    if analyzer.cfg.latest_json:
        print(f"  json (last frame of the last video): {analyzer.cfg.latest_json}")


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def expand_videos(items):
    """Files stay as given, folders become their video files (sorted), "0" = webcam."""
    out = []
    for it in items:
        p = Path(it).expanduser()
        if str(it).isdigit():
            out.append(str(it))
        elif p.is_dir():                           # *_depth*.mkv files are depth, not RGB
            out += sorted(str(f) for f in p.iterdir()
                          if f.suffix.lower() in VIDEO_EXT and "_depth" not in f.stem.lower())
        elif p.exists():
            out.append(str(p))
        else:
            raise AnalyzerError(f"video not found: {it}")
    seen = set()                                   # same video listed twice -> run it once
    return [v for v in out if not (str(Path(v).resolve()) in seen or seen.add(str(Path(v).resolve())))]


def per_video(base, video):
    """results/output.mp4 + part_3.avi -> results/output_part_3.mp4 (None stays None)."""
    if not base:
        return None
    b = Path(base)
    return str(b.with_name(f"{b.stem}_{Path(str(video)).stem}{b.suffix}"))


if __name__ == "__main__":
    main()