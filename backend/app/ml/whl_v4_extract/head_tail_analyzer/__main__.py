"""Run the analyzer with plain positional arguments (like the duck analyzer):

    python -m head_tail_analyzer config.yaml video.mp4 [trace_px] [out.mp4] [extra flags]

    python -m head_tail_analyzer config.yaml clip001_temp.mp4 150 out.mp4
    python -m head_tail_analyzer config.yaml clip001_temp.mp4 150 out.mp4 --preview
    python -m head_tail_analyzer config.yaml clip001_temp.mp4            (trace_px from config)

trace_px = px along each tube after the wrapper (0 = tracing off).
Anything starting with -- is passed on as a normal flag (see: hta --help).
"""
import sys

from .main import main as _main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pos = [a for a in argv if not a.startswith("-")]
    flags = argv[len(pos):] if argv[:len(pos)] == pos else [a for a in argv if a not in pos]
    if len(pos) < 2 or pos[0] in ("help",):
        print(__doc__)
        raise SystemExit(1)
    args = ["--config", pos[0], "--video", pos[1]]
    if len(pos) >= 3:
        args += ["--trace-px", pos[2]]
    if len(pos) >= 4:
        args += ["--output", pos[3]]
    _main(args + flags)


if __name__ == "__main__":
    main()
