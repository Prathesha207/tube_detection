"""
app_state.py - shared in-process state that coordinates GPU access between
your live inference service and the training subprocess.

Import this SAME module from both your inference code and training_service.py
(they must live in the same FastAPI process for this in-memory lock to work --
which they will, since only train.py/prepare_dataset.py itself run as a
separate OS process, not the FastAPI app).

Why this exists: a consumer GPU can't do 30fps live inference and YOLO
training at the same time without one starving the other (or crashing with
CUDA OOM). Before starting a training job, the router calls
try_enter_training(); if it returns False, training is refused (409) rather
than silently fighting inference for VRAM.

NEW: the same problem exists BETWEEN the two inference paths -- a video
upload session and a live camera session can each load their own
TubeAnalyzer (their own YOLO model instance) into VRAM at the same time,
with nothing stopping it. active_inference_kind tracks which *kind* of
inference currently owns the GPU ("video" or "camera"). Multiple sessions
of the SAME kind are still allowed to overlap (that's existing, intentional
behavior on both sides) -- only a kind MISMATCH is rejected. A ref-count
tracks how many sessions of that kind are currently active so the owner
resets to None only once the last one exits.
"""

import threading

_lock = threading.Lock()

_state = {
    "mode": "INFERENCE",        # "INFERENCE" | "TRAINING"
    "active_session_id": None,  # session_id currently training, if any
    "active_pid": None,         # PID of the running train.py/prepare_dataset.py subprocess

    # NEW: cross-kind inference exclusion (video vs camera)
    "active_inference_kind": None,   # None | "video" | "camera"
    "active_inference_count": 0,     # how many sessions of that kind are live
}


def get_mode() -> str:
    with _lock:
        return _state["mode"]


def try_enter_training(session_id: str, pid: int = None) -> bool:
    """Claims the GPU for training. Returns False if a training/prepare job
    is already running, OR if any inference (video or camera) is currently
    active -- training should never start underneath live inference any
    more than inference should start underneath training. Caller should
    reject the new request instead of letting two GPU jobs collide. pid can
    be set later via set_active_pid() once the subprocess actually exists."""
    with _lock:
        if _state["mode"] == "TRAINING":
            return False
        if _state["active_inference_kind"] is not None:
            return False
        _state["mode"] = "TRAINING"
        _state["active_session_id"] = session_id
        _state["active_pid"] = pid
        return True


def set_active_pid(pid: int) -> None:
    with _lock:
        _state["active_pid"] = pid


def exit_training() -> None:
    with _lock:
        _state["mode"] = "INFERENCE"
        _state["active_session_id"] = None
        _state["active_pid"] = None


def get_active_pid():
    with _lock:
        return _state["active_pid"]


def get_active_session_id():
    with _lock:
        return _state["active_session_id"]


# ---------------- NEW: video vs camera inference exclusion ----------------

def try_enter_inference(kind: str) -> bool:
    """Claims the GPU for an inference session of the given kind
    ("video" or "camera"). Returns False if training owns the GPU, or if a
    DIFFERENT inference kind is already active. Safe to call once per
    session at session-start time; call exit_inference(kind) exactly once
    per successful try_enter_inference(kind) call, when that session ends."""
    if kind not in ("video", "camera"):
        raise ValueError(f"unknown inference kind: {kind!r}")
    with _lock:
        if _state["mode"] == "TRAINING":
            return False
        if _state["active_inference_kind"] not in (None, kind):
            return False
        _state["active_inference_kind"] = kind
        _state["active_inference_count"] += 1
        return True


def exit_inference(kind: str) -> None:
    """Releases one claim taken by try_enter_inference(kind). Once the count
    for that kind drops to zero, the GPU is marked free for the other kind
    (or for training) again."""
    with _lock:
        if _state["active_inference_kind"] != kind:
            # Mismatched exit (e.g. double-release, or called for a kind
            # that never successfully entered) -- ignore rather than
            # corrupt state for the kind that's actually active.
            return
        _state["active_inference_count"] = max(0, _state["active_inference_count"] - 1)
        if _state["active_inference_count"] == 0:
            _state["active_inference_kind"] = None


def get_active_inference_kind():
    with _lock:
        return _state["active_inference_kind"]
