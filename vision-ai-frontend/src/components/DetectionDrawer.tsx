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

  const displayFps = mlStats.status !== 'idle' && mlStats.fps > 0 ? mlStats.fps : metrics.fps;
  const displayFrames = mlStats.status !== 'idle' && mlStats.frames_processed > 0 ? mlStats.frames_processed : metrics.framesProcessed;
  const displayProgress = mlStats.status !== 'idle' ? mlStats.progress : 100;
  const latencyMs = mlStats.latency_ms ?? metrics.inferenceTimeMs;

  const fpsForTime = displayFps > 0 ? displayFps : 30;
  const computedUptime = displayFrames / fpsForTime;

  const isEmptyState = tubes.length === 0 && !anomalyStatus.headsCount && !anomalyStatus.tailsCount;
  const isWarming = anomalyStatus.message === 'WARMING';

  // Construct frame numbers and verdict
  const frameNumber = mlStats.frame || mlStats.frames_processed || metrics.framesProcessed || 0;
  const bigger = mlStats.bigger_tube;
  const smaller = mlStats.smaller_tube;

  const isCompleted =
    mlStats.status === 'completed' ||
    anomalyStatus.message === 'COMPLETED' ||
    (mlStats.total_frames > 0 && mlStats.frames_processed >= mlStats.total_frames) ||
    displayProgress >= 100;

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
    return anomalyStatus.message || 'Completed';
  }, [isRunning, isCameraSource, isCompleted, mlStats.status, isStandby, anomalyStatus.message]);

  const verdictText = useMemo(() => {
    if (bigger && smaller) {
      const bw = bigger.width_px ? `${Number(bigger.width_px).toFixed(1)}px` : (bigger.width_mm ? `${Number(bigger.width_mm).toFixed(1)}mm` : '');
      const sw = smaller.width_px ? `${Number(smaller.width_px).toFixed(1)}px` : (smaller.width_mm ? `${Number(smaller.width_mm).toFixed(1)}mm` : '');
      return `BIGGER tube: ${bw} | SMALLER tube: ${sw}`;
    }
    if (bigger) {
      const bw = bigger.width_px ? `${Number(bigger.width_px).toFixed(1)}px` : (bigger.width_mm ? `${Number(bigger.width_mm).toFixed(1)}mm` : '');
      return `BIGGER tube: ${bw}`;
    }
    if (anomalyStatus.headsCount !== anomalyStatus.tailsCount) {
      return `${Math.abs(anomalyStatus.headsCount - anomalyStatus.tailsCount)} Unmatched Tube End(s)`;
    }
    return anomalyStatus.subMessage || 'All tube ends paired & balanced';
  }, [bigger, smaller, anomalyStatus]);

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
      className="w-full lg:w-[22rem] xl:w-[24rem] 2xl:w-[26rem] h-auto lg:h-full flex-shrink-0 flex flex-col gap-3 min-h-0 overflow-y-auto invisible-scrollbar items-stretch"
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
              <span>{isRunning ? 'PROCESSING' : (isCompleted ? 'COMPLETED' : (isStandby ? 'STANDBY' : anomalyStatus.message))}</span>
              {displayFps > 0 && isRunning && (
                <span className="ml-1 opacity-90 font-mono text-[10px]">
                  &bull; {displayFps.toFixed(1)} fps{latencyMs ? ` (${latencyMs}ms)` : ''}
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
                    <span className="font-mono">{formatTime(computedUptime)}</span>
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
                      <span>{formatTime(computedUptime)}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* TOP FRAME BANNER */}
              <div className="flex items-center justify-between px-2.5 py-1.5 rounded-xl bg-[var(--bg-card-subtle)] text-[var(--text-primary)] font-mono text-xs border border-[var(--border-color)] shadow-xs">
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
                <span className="text-[10px] text-[var(--text-secondary)] font-normal">
                  {formatTime(computedUptime)}
                </span>
              </div>

              {/* DETECTIONS LIST */}
              <div className="space-y-1.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)] p-2">
                <div className="text-[10px] font-mono text-[var(--text-secondary)] uppercase tracking-wider px-1 flex justify-between">
                  <span>Detections (Index &bull; Class &bull; Width)</span>
                  <span>Tag &bull; Tube</span>
                </div>

                <div className="flex flex-col gap-1.5 max-h-[260px] overflow-y-auto invisible-scrollbar">
                  {tubes.map((d, idx) => {
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
                        className={`flex items-center justify-between p-2 rounded-lg font-mono text-xs border border-[var(--border-color)] bg-[var(--bg-card)] text-[var(--text-primary)] shadow-xs transition-all cursor-pointer ${
                          isSelected ? 'ring-2 ring-[var(--accent-pond)] shadow-sm' : 'hover:border-[var(--accent-pond)]'
                        }`}
                      >
                        {/* Index -> Class */}
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="font-bold text-[var(--text-secondary)] min-w-[20px]">
                            #{d.id ?? idx}
                          </span>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-black ${
                            isHead ? 'bg-sky-500 text-white' : 'bg-indigo-500 text-white'
                          }`}>
                            {d.role || 'END'}
                          </span>
                        </div>

                        {/* Width & Value -> Size Tag -> Next Tube (tube#1) or CHECK */}
                        <div className="flex items-center gap-2 shrink-0">
                          <span className="text-[11px] text-[var(--text-secondary)]">
                            width=
                          </span>
                          <span className="font-black text-sm text-[var(--text-primary)] min-w-[50px] text-right">
                            {widthVal > 0 ? `${widthVal.toFixed(1)}px` : '--'}
                          </span>

                          {/* Size Tags: SAME Green, SMALLER Yellow, BIGGER Red */}
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

                          {/* Tube Tag or Check Flag */}
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
                      </div>
                    );
                  })}
                </div>
              </div>

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
        />
      )}
    </aside>
  );
};
