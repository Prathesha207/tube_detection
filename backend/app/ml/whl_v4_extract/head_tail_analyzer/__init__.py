"""head_tail_analyzer - tube head/tail detection, width, bigger tube, depth XYZ and the
trace out of the wrapper (YOLO11-seg).

    from head_tail_analyzer import HeadTailAnalyzer
    analyzer = HeadTailAnalyzer("config.yaml", preload=True)
    for result in analyzer.process_video("clip.mp4", trace_px=150):
        ...
"""
__version__ = "4.0.1"

from .main import HeadTailAnalyzer, AnalyzerError, parse_trace_px  # noqa: E402

__all__ = ["HeadTailAnalyzer", "AnalyzerError", "parse_trace_px", "__version__"]
