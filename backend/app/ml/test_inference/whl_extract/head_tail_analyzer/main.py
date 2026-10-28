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
    analyzer.unload_model()                         # the ONLY thing that frees it (GPU / RAM)

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

__all__ = ["HeadTailAnalyzer", "requires_model", "main", *from_rvf]


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

    # ------------------------------------------------------------- per frame
    @requires_model
    def process_frame(self, entry, frame, frame_index=None):
        """Run the full pipeline on one frame and return the result dict
        (heads_count, tails_count, detections, bigger_tube, frame_total, fps, ...)."""
        result, _ = rvf.analyze_frame(entry, frame, self.state, self.roi, self.cfg,
                                      frame_index, annotate=False)
        return result

    @requires_model
    def process_frame_annotated(self, entry, frame, frame_index=None):
        """Same as process_frame, returns (result dict, annotated BGR frame)."""
        return rvf.analyze_frame(entry, frame, self.state, self.roi, self.cfg,
                                 frame_index, annotate=True, masks=self._masks_ok)

    # ----------------------------------------------------------------- video
    def process_video(self, video=None, output=None, stride=None, preview=False) -> Iterator[dict]:
        """Generator: one result dict per processed frame of a video (or webcam "0").
        `output` = annotated .mp4 to write. The model is NOT unloaded when the video ends."""
        video = video or self.cfg.video
        if not video:
            raise AnalyzerError("no video: pass video=... or set it in the config")
        output = output or self.cfg.output_video
        stride = max(int(stride or self.cfg.stride), 1)
        cap, fps, total = rvf.open_video(video)
        self.state.frame_total, self.state.video_fps = total, round(fps, 2)
        out = rvf.VideoOutput(output, fps / stride, preview) if (output or preview) else None
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                if out:
                    result, vis = self.process_frame_annotated(frame, frame_index=idx)
                    if not out.write(vis):
                        break
                else:
                    result = self.process_frame(frame, frame_index=idx)
                yield result
                if idx % 10 == 0:
                    print(f"[progress] frame {idx}/{total or '?'}  "
                          f"{result.get('fps') or 0:.1f} frames/s", flush=True)
        finally:
            cap.release()
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
    ap.add_argument("--preview", action="store_true", help="show a live window (q to stop)")
    ap.add_argument("--quiet", action="store_true", help="don't print every frame")
    a = ap.parse_args(argv)

    try:
        analyzer = HeadTailAnalyzer(
            a.config, use_roi=not a.no_roi, model_path=a.model,
            output_video=a.output, latest_json=a.json, history_jsonl=a.history, roi_json=a.roi,
            device=a.device, conf=a.conf, stride=a.stride, mm_per_px=a.mm_per_px,
            print_frames=False if a.quiet else None)
        videos = expand_videos(a.video) if a.video else [analyzer.cfg.video]
        if not videos or videos == [None]:
            raise AnalyzerError("no video: pass --video (files or folders) or set video in the config")
        out_base, hist_base = analyzer.cfg.output_video, analyzer.cfg.history_jsonl
        many = len(videos) > 1

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
        elif p.is_dir():
            out += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
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