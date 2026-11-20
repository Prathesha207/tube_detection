import React, { useMemo } from 'react';
import { AnomalyStatus, TubeEntity, DetectionMetrics, LogEntry } from '../types';
import {
  ChevronLeft,
  ChevronRight,
  Terminal,
} from 'lucide-react';
import { Badge, IconButton } from './ui';
import { useInferenceStore } from '../store/inferenceStore';
import { CameraImageAdjustmentsCard } from './CameraImageAdjustmentsCard';
import { CameraConfig } from '../types';

interface DetectionDrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  anomalyStatus: AnomalyStatus;
  tubes: TubeEntity[];
  metrics: DetectionMetrics;
  selectedTubeId: string | null;
  onSelectTube: (id: string | null) => void;
  isStandby?: boolean;
  logs?: LogEntry[];
  isCameraSource?: boolean;
  isRunning?: boolean;
  isCameraConnected?: boolean;
  cameraConfig?: CameraConfig;
  onUpdateCameraConfig?: (newConfig: Partial<CameraConfig>) => void;
  isRecording?: boolean;
}

export const DetectionDrawer: React.FC<DetectionDrawerProps> = ({
  isOpen,
  onToggle,
  anomalyStatus,
  tubes,
  metrics,
  selectedTubeId,
  onSelectTube,
  isStandby = false,
  logs: _logs = [],
  isCameraSource = false,
  isRunning = false,
  isCameraConnected = false,
  cameraConfig,
  onUpdateCameraConfig,
  isRecording = false,
}) => {
  const mlStats = useInferenceStore((state) => state.stats);

  // Format uptime in hh:mm:ss
  const formatTime = (seconds: number) => {
    if (!isFinite(seconds) || isNaN(seconds)) return '00:00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const hrs = Math.floor(mins / 60);
    return `${hrs.toString().padStart(2, '0')}:${(mins % 60).toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const displayFps = isRunning ? (mlStats.fps > 0 ? mlStats.fps : metrics.fps) : 0;
  const rawBackendFrames = mlStats.frames_processed ?? mlStats.frame ?? mlStats.frame_seq;
  const displayFrames = mlStats.status !== 'idle' && mlStats.status !== 'stopped' && rawBackendFrames !== undefined
    ? rawBackendFrames
    : (isRunning ? (metrics.framesProcessed || 0) : 0);
  const calculatedProgress = (mlStats.total_frames > 0 && displayFrames > 0)
    ? Math.min(100, Math.round((displayFrames / mlStats.total_frames) * 100))
    : (mlStats.progress ?? 0);
  const displayProgress = mlStats.status !== 'idle' && mlStats.status !== 'stopped'
    ? (mlStats.progress > 0 ? mlStats.progress : calculatedProgress)
    : 0;
  const latencyMs = mlStats.latency_ms ?? mlStats.frame_time_ms ?? metrics.latencyMs ?? metrics.inferenceTimeMs;

  const displayUptime = isRunning ? (metrics.uptimeSeconds || 0) : 0;

  const isCompleted =
    !isRunning &&
    displayFrames > 0 &&
    (mlStats.status === 'completed' ||
      anomalyStatus.message === 'COMPLETED' ||
      (mlStats.total_frames > 0 && mlStats.frames_processed >= mlStats.total_frames));

  const isSessionActive = isRunning || (isCompleted && displayFrames > 0);
  const isEmptyState = !isSessionActive;
  const isWarming = anomalyStatus.message === 'WARMING';

  // Construct frame numbers and verdict
  const frameNumber = typeof mlStats.frames_processed === 'number'
    ? mlStats.frames_processed
    : (typeof mlStats.frame === 'number' ? mlStats.frame : (metrics.framesProcessed ?? 0));
  const bigger = mlStats.bigger_tube;
  const smaller = mlStats.smaller_tube;

  const telemetryStatusText = useMemo(() => {
    if (isRunning) {
      return isCameraSource ? 'Camera Inference' : 'Video Inference';
    }
    if (isCompleted) {
      return 'Completed';
    }
    if (mlStats.status === 'stopped' || anomalyStatus.message === 'STOPPED') {
      return 'Stopped';
    }
    if (mlStats.status === 'paused' || anomalyStatus.message === 'PAUSED') {
      return 'Paused';
    }
    if (isStandby) {
      return 'Standby';
    }
    return anomalyStatus.message || 'Standby';
  }, [isRunning, isCameraSource, isCompleted, mlStats.status, isStandby, anomalyStatus.message]);

  const verdictText = useMemo(() => {
    if (bigger && typeof bigger === 'object') {
      const getBiggerWidth = (t: any) => {
        const wDepth = t.width_mm_depth ?? t.width_mm;
        const wPx = t.width_px;
        const hasDepth = typeof wDepth === 'number' && wDepth > 0;
        const hasPx = typeof wPx === 'number' && wPx > 0;
        if (hasDepth && hasPx) return `${Number(wDepth).toFixed(1)}mm (${Number(wPx).toFixed(1)}px)`;
        if (hasDepth) return `${Number(wDepth).toFixed(1)}mm`;
        if (hasPx) return `${Number(wPx).toFixed(1)}px`;
        return '';
      };
      if (bigger.status === 'OK') {
        const bw = getBiggerWidth(bigger);
        const role = bigger.role ? `${String(bigger.role).toLowerCase()} ` : '';
        const id = bigger.id != null ? `#${bigger.id} ` : '';
        const ratio = bigger.ratio ? ` (x${Number(bigger.ratio).toFixed(2)})` : '';
        return `BIGGER TUBE: ${role}${id}${bw}${ratio}`.trim();
      }
      if (bigger.status === 'SAME') {
        const ratio = bigger.ratio ? ` (x${Number(bigger.ratio).toFixed(2)})` : '';
        return `TUBES SAME SIZE${ratio}`;
      }
      if (bigger.status === 'NOT_ENOUGH') {
        return 'BIGGER TUBE: need 2 measured tubes';
      }
      if (smaller) {
        const bw = getBiggerWidth(bigger);
        const sw = getBiggerWidth(smaller);
        return `BIGGER tube: ${bw} | SMALLER tube: ${sw}`;
      }
      const bw = getBiggerWidth(bigger);
      if (bw) return `BIGGER tube: ${bw}`;
    }
    if ((anomalyStatus.headsCount ?? 0) === 0 && (anomalyStatus.tailsCount ?? 0) === 0 && tubes.length === 0) {
      return 'No tube ends currently detected';
    }
    if (anomalyStatus.headsCount !== anomalyStatus.tailsCount) {
      return `${Math.abs(anomalyStatus.headsCount - anomalyStatus.tailsCount)} Unmatched Tube End(s)`;
    }
    return anomalyStatus.subMessage || 'All tube ends paired & balanced';
  }, [bigger, smaller, anomalyStatus, tubes.length]);

  // Group tube ends by physical tube identity (Tube 1 = Blue, Tube 2 = White)
  const tubeTelemetry = useMemo(() => {
    const active = tubes.filter(
      (t) => !(t as any).held && t.status !== 'COASTED' && !t.is_coasting
    );

    const findEnd = (color: 'blue' | 'white', role: 'HEAD' | 'TAIL') => {
      const byColor = active.find(
        (t) => String(t.tube_color || '').toLowerCase() === color && t.role === role
      );
      if (byColor) return byColor;
      const targetPairId = color === 'blue' ? 1 : 2;
      return active.find((t) => t.pair_id === targetPairId && t.role === role);
    };

    const blueHead = findEnd('blue', 'HEAD');
    const blueTail = findEnd('blue', 'TAIL');
    const whiteHead = findEnd('white', 'HEAD');
    const whiteTail = findEnd('white', 'TAIL');

    const formatZ = (end?: TubeEntity) => {
      if (!end || typeof end.z_mm !== 'number' || end.z_mm <= 0) return '-';
      return `${end.z_mm.toFixed(0)}mm`;
    };

    const getStatus = (head?: TubeEntity, tail?: TubeEntity) => {
      if (head?.size && tail?.size && head.size === tail.size) {
        return head.size === 'SAME' ? 'Same' : head.size;
      }
      if (typeof head?.z_mm === 'number' && typeof tail?.z_mm === 'number' && head.z_mm > 0 && tail.z_mm > 0) {
        const diff = Math.abs(head.z_mm - tail.z_mm);
        if (diff <= 15) return 'Same';
        return `Δ${diff.toFixed(0)}mm`;
      }
      return null;
    };

    return [
      {
        id: 1,
        name: 'TUBE 1',
        colorName: 'BLUE',
        badgeColor: 'text-sky-400 bg-sky-500/15 border-sky-500/30',
        dotColor: 'bg-sky-400 shadow-[0_0_6px_rgba(56,189,248,0.8)]',
        headZ: formatZ(blueHead),
        tailZ: formatZ(blueTail),
        hasData: Boolean(blueHead || blueTail),
        status: getStatus(blueHead, blueTail),
      },
      {
        id: 2,
        name: 'TUBE 2',
        colorName: 'WHITE',
        badgeColor: 'text-slate-100 bg-white/15 border-white/30',
        dotColor: 'bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)]',
        headZ: formatZ(whiteHead),
        tailZ: formatZ(whiteTail),
        hasData: Boolean(whiteHead || whiteTail),
        status: getStatus(whiteHead, whiteTail),
      },
    ];
  }, [tubes]);

  if (!isOpen) {
    return (
      <button
        onClick={() => {
          onToggle();
        }}
        title="Open Detection Details"
        className="fixed right-3 top-1/2 transform -translate-y-1/2 z-40 flex items-center gap-1.5 px-2.5 py-4 rounded-l-2xl bg-[var(--bg-card)] border border-r-0 border-[var(--border-color)] shadow-md text-xs font-semibold text-[var(--text-primary)] group hover:bg-[var(--btn-secondary-hover)] cursor-pointer"
      >
        <ChevronLeft className="w-4 h-4 text-[var(--accent-pond)] group-hover:-translate-x-0.5 transition-transform" />
        <span className="[writing-mode:vertical-lr] tracking-wider uppercase text-[10px] text-[var(--text-secondary)]">
          {isEmptyState ? 'Standby' : 'Inference Details'}
        </span>
      </button>
    );
  }

  return (
    <aside
      className="w-full lg:w-[25%] lg:min-w-[280px] h-auto lg:h-full flex-shrink-0 flex flex-col gap-3 min-h-0 overflow-y-auto invisible-scrollbar items-stretch"
    >
      {/* SINGLE UNIFIED INFERENCE CARD */}
      <div
        className="p-3.5 rounded-2xl border border-[var(--border-color)] bg-[var(--bg-card)] shadow-md flex flex-col w-full shrink-0 justify-between overflow-hidden transition-all duration-200"
      >
        {/* 1. CARD HEADER: Terminal icon, Title, Status badge & Close button (subtitle removed) */}
        <div className="flex items-center justify-between pb-2 border-b border-[var(--border-color)] shrink-0">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-[var(--accent-pond)]" />
            <span className="font-semibold text-xs tracking-wider uppercase text-[var(--text-primary)]">
              Inference Details
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            {mlStats.wrapper && (
              <span
                className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-purple-500/15 text-purple-600 dark:text-purple-300 border border-purple-500/30 flex items-center gap-1 shrink-0"
                title={mlStats.trace_px ? `Wrapper Detected • Tracing ${mlStats.trace_px}px` : 'Wrapper Detected'}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-purple-500 dark:bg-purple-400 animate-pulse" />
                WRAPPER{mlStats.trace_px ? ` ${mlStats.trace_px}px` : ''}
              </span>
            )}
            <Badge
              variant={
                isWarming
                  ? 'warning'
                  : isRunning || isCompleted
                    ? 'active'
                    : 'neutral'
              }
              size="sm"
              dot
            >
              <span>{isRunning ? 'PROCESSING' : (isCompleted ? 'COMPLETED' : (isStandby ? 'STANDBY' : (mlStats.status === 'stopped' || anomalyStatus.message === 'STOPPED' ? 'STOPPED' : anomalyStatus.message || 'STANDBY')))}</span>
              {displayFps > 0 && isRunning && (
                <span className="ml-1 opacity-90 font-mono text-[10px]">
                  &bull; {displayFps.toFixed(1)} fps
                </span>
              )}
            </Badge>

            <IconButton
              size="sm"
              variant="ghost"
              aria-label="Close Drawer"
              title="Close Drawer"
              icon={<ChevronRight className="w-4 h-4 text-[var(--accent-pond)]" />}
              onClick={() => {
                onToggle();
              }}
            />
          </div>
        </div>

        {/* 2. CARD BODY */}
        <div className="pt-2 flex-1 flex flex-col justify-between overflow-hidden gap-2">
          {isEmptyState ? (
            <div className="flex-1 flex flex-col justify-between py-2">
              <div className="flex-1 flex flex-col items-center justify-center py-6 px-2 text-center text-[var(--text-secondary)]">
                <Terminal className="w-8 h-8 text-[var(--accent-pond)] mb-2 animate-pulse" />
                <p className="text-sm font-bold text-[var(--text-primary)]">
                  {isStandby
                    ? (isCameraSource ? 'Camera Ready' : 'Video Ready')
                    : 'Awaiting Tube Video Stream'}
                </p>
                <p className="text-xs mt-1 max-w-[240px] leading-relaxed text-[var(--text-secondary)]">
                  {anomalyStatus.subMessage || 'Upload a tube video to begin real-time ML inference.'}
                </p>
              </div>

              {/* BOTTOM TELEMETRY (EMPTY STATE) */}
              <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-secondary)] px-1 pt-1 border-t border-[var(--border-color)]">
                <span>Script: head_tail_classification</span>
                <span className={isRunning ? "text-emerald-500 font-bold" : isCompleted ? "text-emerald-600 dark:text-emerald-400 font-semibold" : "text-[var(--text-secondary)] font-medium"}>
                  &bull; {telemetryStatusText}
                </span>
                <span>0 Detections</span>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {/* VIDEO PROGRESS BLOCK (Integrated in the same card) */}
              <div className="p-2.5 rounded-xl border border-[var(--border-color)] bg-[var(--bg-card-subtle)] flex flex-col gap-1.5">
                {isCameraSource ? (
                  <div className="flex justify-between items-center text-xs text-[var(--text-secondary)]">
                    <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      Live Feed
                    </span>
                    <span className="font-mono">{displayFrames.toLocaleString()} frames</span>
                    <span className="font-mono">{formatTime(displayUptime)}</span>
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <div className="flex justify-between items-center text-xs text-[var(--text-secondary)]">
                      <span className="font-medium text-[var(--text-primary)]">Processing Progress</span>
                      <span className="font-mono font-bold text-[var(--accent-pond)]">{Math.round(displayProgress)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-[var(--btn-secondary-border)] rounded-full overflow-hidden relative">
                      <div
                        className="h-full rounded-full bg-[var(--accent-pond)] transition-all duration-300 ease-out"
                        style={{ width: `${Math.max(0, Math.min(100, displayProgress))}%` }}
                      />
                    </div>
                    <div className="flex justify-between items-center font-mono text-[10px] text-[var(--text-secondary)] pt-0.5">
                      <span>{displayFrames.toLocaleString()} frames</span>
                      <span>{formatTime(displayUptime)}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* TOP FRAME BANNER (commented out per request) */}
              {/* <div className="flex items-center justify-between px-2.5 py-1.5 rounded-xl bg-[var(--bg-card-subtle)] text-[var(--text-primary)] font-mono text-xs border border-[var(--border-color)] shadow-xs">
                <div className="flex items-center gap-2">
                  <span className="text-[var(--accent-pond)] font-bold">
                    Frame {frameNumber}:
                  </span>
                  <span className="text-sky-600 dark:text-sky-400 font-bold">
                    HEADS={anomalyStatus.headsCount}
                  </span>
                  <span className="text-indigo-600 dark:text-indigo-400 font-bold">
                    TAILS={anomalyStatus.tailsCount}
                  </span>
                </div>
              </div> */}

              {/* DETECTIONS LIST */}
              <div className="space-y-1.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)] p-2">
                <div className="text-[10px] font-mono text-[var(--text-secondary)] uppercase tracking-wider px-1 flex justify-between">
                  <span>Detections (Index &bull; Class)</span>
                  <span>Confidence</span>
                </div>
                {/* {Object.entries({ ...(mlStats.class_counts || {}), ...(mlStats.tube_counts || {}) }).length > 0 && (
                  <div className="flex flex-wrap gap-1 px-1" aria-label="Current model class counts">
                    {Object.entries({ ...(mlStats.class_counts || {}), ...(mlStats.tube_counts || {}) }).map(([name, count]) => (
                      <span key={name} className="rounded bg-[var(--bg-card)] px-1.5 py-0.5 text-[9px] font-mono text-[var(--text-secondary)] border border-[var(--border-color)]">
                        {name}: {count}
                      </span>
                    ))}
                  </div>
                )} */}

                <div
                  className={`flex flex-col gap-1 justify-start pr-0.5 transition-all duration-200 ${
                    tubes.length > 4
                      ? 'h-auto max-h-[380px] overflow-y-auto invisible-scrollbar'
                      : 'h-[122px] overflow-hidden'
                  }`}
                >
                  {tubes.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-center text-xs font-mono text-[var(--text-secondary)] opacity-80">
                      No tube ends detected in frame
                    </div>
                  ) : (
                    tubes.map((d, idx) => {
                      const isHead = d.role === 'HEAD';
                      const isSelected = selectedTubeId != null && String(d.id) === String(selectedTubeId);
                      const widthVal = Number(d.width_px ?? d.width_mm ?? 0);
                      const isCheck = d.status && d.status !== 'OK';

                      return (
                        <div
                          key={`det-${d.id ?? idx}`}
                          onClick={() => onSelectTube?.(isSelected ? null : String(d.id))}
                          role="button"
                          tabIndex={0}
                          className={`flex items-center justify-between px-2 py-1 rounded-lg font-mono text-xs border border-[var(--border-color)] bg-[var(--bg-card)] text-[var(--text-primary)] shadow-xs transition-all cursor-pointer shrink-0 ${isSelected ? 'ring-2 ring-[var(--accent-pond)] shadow-sm' : 'hover:border-[var(--accent-pond)]'
                            }`}
                        >
                          {/* Index -> Class -> Tracker ID */}
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="font-bold text-[var(--text-secondary)] min-w-[20px]">
                              #{idx + 1}
                            </span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-black ${
                              String(d.tube_color || '').toLowerCase() === 'blue'
                                ? (isHead ? 'bg-sky-500 text-white' : 'bg-sky-600 text-white')
                                : (isHead ? 'bg-white text-slate-900 border border-slate-200' : 'bg-slate-100 text-slate-900 border border-slate-300')
                            }`}>
                              {d.role || 'END'}
                            </span>
                            {d.tube_color && (
                              <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                                String(d.tube_color || '').toLowerCase() === 'blue'
                                  ? 'bg-sky-700 text-white'
                                  : 'bg-white text-slate-900 border border-slate-300'
                              }`}>
                                {d.tube_color}
                              </span>
                            )}
                            <span className="text-[10px] font-mono text-[var(--text-secondary)] opacity-80 whitespace-nowrap ml-1">
                              ID:{d.displayId ?? d.id}
                            </span>
                          </div>

                          {/* Width, Size & Confidence */}
                          <div className="flex items-center gap-2 shrink-0 text-[10px] font-mono text-[var(--text-secondary)] ml-auto">
                            <span>
                              w:{' '}
                              {typeof (d.width_mm_depth ?? d.width_mm) === 'number' && (d.width_mm_depth ?? d.width_mm)! > 0 ? (
                                <>
                                  <strong className="text-[var(--text-primary)]">
                                    {(d.width_mm_depth ?? d.width_mm)!.toFixed(1)}mm
                                  </strong>
                                  {typeof d.width_px === 'number' && d.width_px > 0 && (
                                    <span className="text-[var(--text-secondary)] opacity-80 ml-1">
                                      ({d.width_px.toFixed(1)}px)
                                    </span>
                                  )}
                                </>
                              ) : typeof d.width_px === 'number' && d.width_px > 0 ? (
                                <strong className="text-[var(--text-primary)]">
                                  {d.width_px.toFixed(1)}px
                                </strong>
                              ) : (
                                <strong className="text-[var(--text-secondary)]">
                                  null
                                </strong>
                              )}
                            </span>
                            {typeof d.z_mm === 'number' && d.z_mm > 0 && (
                              <span className="font-mono text-[9.5px] font-bold px-1.5 py-0.5 rounded border leading-none whitespace-nowrap text-cyan-900 dark:text-cyan-300 bg-cyan-100/90 dark:bg-cyan-950/60 border-cyan-300 dark:border-cyan-500/40" title={`Distance Z: ${d.z_mm} mm`}>
                                z:{d.z_mm.toFixed(0)}mm
                              </span>
                            )}
                            {d.size === 'BIGGER' && (
                              <span className="px-1.5 py-0.2 rounded text-[8.5px] font-black uppercase tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/35">
                                BIGGER
                              </span>
                            )}
                            {d.size === 'SMALLER' && (
                              <span className="px-1.5 py-0.2 rounded text-[8.5px] font-black uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/35">
                                SMALLER
                              </span>
                            )}
                            {d.size === 'SAME' && (
                              <span className="px-1.5 py-0.2 rounded text-[8.5px] font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/35">
                                SAME
                              </span>
                            )}
                            {d.confidence != null ? (
                              <span className="text-[11px] text-[var(--text-secondary)] font-medium">
                                {Math.round(Number(d.confidence) <= 1 ? Number(d.confidence) * 100 : Number(d.confidence))}%
                              </span>
                            ) : null}
                          </div>

                          {/* COMMENTED OUT: width and unpaired / pair tags
                        <div className="flex items-center gap-2 shrink-0">
                          <span className="text-[11px] text-[var(--text-secondary)]">
                            width=
                          </span>
                          <span className="font-black text-sm text-[var(--text-primary)] min-w-[50px] text-right">
                            {widthVal > 0 ? `${widthVal.toFixed(1)}px` : '--'}
                          </span>
                          {d.size === 'SAME' && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-600 text-white shadow-2xs">
                              SAME
                            </span>
                          )}
                          {d.size === 'SMALLER' && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-black bg-yellow-400 text-slate-950 shadow-2xs">
                              SMALLER
                            </span>
                          )}
                          {d.size === 'BIGGER' && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white shadow-2xs">
                              BIGGER
                            </span>
                          )}
                          {isCheck ? (
                            <span className="px-1.5 py-0.5 rounded text-[9px] bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 font-medium max-w-[120px] truncate" title={d.status}>
                              {d.status}
                            </span>
                          ) : d.pair_id != null ? (
                            <span className="px-1.5 py-0.5 rounded text-[9.5px] bg-slate-200 dark:bg-slate-700 text-emerald-600 dark:text-emerald-400 font-semibold">
                              tube#{d.pair_id}
                            </span>
                          ) : (
                            <span className="px-1.5 py-0.5 rounded text-[9.5px] bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 font-medium">
                              unpaired
                            </span>
                          )}
                        </div>
                        */}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* TUBE 1 & TUBE 2 Z-DEPTH TELEMETRY CARD */}
              {tubeTelemetry.some((t) => t.hasData) && (
                <div className="rounded-xl border border-[var(--border-color)] bg-[var(--bg-card)]/80 p-2 font-mono shadow-xs">
                  <div className="flex flex-col gap-1.5">
                    {tubeTelemetry.map((tube) => (
                      <div
                        key={tube.name}
                        className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)]/70 text-xs"
                      >
                        {/* Tube Label */}
                        <div className="flex items-center gap-1.5 shrink-0 min-w-[105px]">
                          <span className={`w-2 h-2 rounded-full ${tube.dotColor}`} />
                          <span className="font-extrabold text-[var(--text-primary)] text-[11px] tracking-wide">
                            {tube.name}
                          </span>
                          <span className={`px-1 py-0.2 rounded text-[8px] font-black uppercase border ${tube.badgeColor}`}>
                            {tube.colorName}
                          </span>
                        </div>

                        {/* Head Z | Tail Z */}
                        <div className="flex items-center gap-2 text-[11px] shrink-0 font-mono">
                          <span className="text-[var(--text-secondary)]">
                            Head Z: <strong className="text-cyan-600 dark:text-cyan-300 font-bold">{tube.headZ}</strong>
                          </span>
                          <span className="text-[var(--border-color)] opacity-70">│</span>
                          <span className="text-[var(--text-secondary)]">
                            Tail Z: <strong className="text-cyan-600 dark:text-cyan-300 font-bold">{tube.tailZ}</strong>
                          </span>
                        </div>

                        {/* Match Status */}
                        <div className="shrink-0 text-right min-w-[45px]">
                          {tube.status ? (
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border border-emerald-500/20">
                              ({tube.status})
                            </span>
                          ) : (
                            <span className="text-[10px] text-[var(--text-secondary)] opacity-40">-</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* VERDICT BANNER (In Yellow/Amber) */}
              {verdictText && (
                <div className="px-2.5 py-2 rounded-xl text-xs font-mono font-bold flex items-center justify-between border-2 border-amber-400 dark:border-amber-500 bg-amber-400/15 dark:bg-amber-950/40 text-amber-950 dark:text-amber-100 shadow-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-amber-600 dark:text-amber-400 text-sm">&minus;&gt;</span>
                    <span className="text-amber-950 dark:text-amber-100 font-bold tracking-tight">
                      {verdictText}
                    </span>
                  </div>
                </div>
              )}

              {/* BOTTOM TELEMETRY */}
              <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-secondary)] px-1 pt-0.5">
                <span>Script: head_tail_classification</span>
                <span className={isRunning ? "text-emerald-500 font-bold" : isCompleted ? "text-emerald-600 dark:text-emerald-400 font-semibold" : "text-[var(--text-secondary)] font-medium"}>
                  &bull; {telemetryStatusText}
                </span>
                <span>{tubes.length} Detections</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 2. CAMERA IMAGE ADJUSTMENTS CARD (Shown in Sidebar when Camera is connected) */}
      {isCameraSource && isCameraConnected && (
        <CameraImageAdjustmentsCard
          cameraId={cameraConfig?.id}
          config={cameraConfig}
          onUpdateConfig={onUpdateCameraConfig}
          isLive={true}
          isRecording={isRecording}
        />
      )}
    </aside>
  );
};
