

import React from 'react';
import { AnomalyStatus, DuckEntity, DetectionMetrics, LogEntry } from '../types';
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Activity,
  Layers,
  ArrowUp,
  ArrowDown,
  Hand,
} from 'lucide-react';
import { Badge, IconButton } from './ui';
import { playWaterDropSound } from '../utils/audio';
import { DetectionGallery } from './AnomalyGallery';
import { useInferenceStore } from '../store/inferenceStore';

interface DetectionDrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  anomalyStatus: AnomalyStatus;
  ducks: DuckEntity[];
  metrics: DetectionMetrics;
  selectedDuckId: string | null;
  onSelectDuck: (id: string | null) => void;
  isStandby?: boolean;
  logs?: LogEntry[];
  isCameraSource?: boolean;
}

export const DetectionDrawer: React.FC<DetectionDrawerProps> = ({
  isOpen,
  onToggle,
  anomalyStatus,
  ducks,
  metrics,
  selectedDuckId,
  onSelectDuck,
  isStandby = false,
  logs = [],
  isCameraSource = false,
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

  // If we have actual ML stats, use those! Otherwise fallback to mock metrics.
  const displayFps = mlStats.status !== 'idle' ? mlStats.fps : metrics.fps;
  const displayFrames = mlStats.status !== 'idle' ? mlStats.frames_processed : metrics.framesProcessed;
  const displayProgress = mlStats.status !== 'idle' ? mlStats.progress : 100;

  // Calculate uptime strictly from frames / fps, defaulting to 30fps if unknown
  const fpsForTime = displayFps > 0 ? displayFps : 30;
  const computedUptime = displayFrames / fpsForTime;

  const speciesList = Array.from(new Set(ducks.map((d) => d.species))).join(', ') || '--';
  const isEmptyState = ducks.length === 0;
  const anomalyDucks = ducks.filter((d) => d.isAnomaly && !d.provisional);
  const hasDetections = ducks.length > 0;
  const isHand = anomalyStatus.message === 'HAND DETECTED';

  if (!isOpen) {
    return (
      <button
        onClick={() => {
          playWaterDropSound();
          onToggle();
        }}
        title="Open Detection Details"
        className="fixed right-3 top-1/2 transform -translate-y-1/2 z-40 flex items-center gap-1.5 px-2.5 py-4 rounded-l-2xl bg-[var(--bg-card)] border border-r-0 border-[var(--border-color)] shadow-md text-xs font-semibold text-[var(--text-primary)] group hover:bg-[var(--btn-secondary-hover)] cursor-pointer"
      >
        <ChevronLeft className="w-4 h-4 text-[var(--accent-pond)] group-hover:-translate-x-0.5 transition-transform" />
        <span className="[writing-mode:vertical-lr] tracking-wider uppercase text-[10px] text-[var(--text-secondary)]">
          {isEmptyState ? 'Standby' : isHand ? 'Hand Present' : anomalyStatus.isAnomaly ? 'Anomaly Alert' : 'Detection Details'}
        </span>
      </button>
    );
  }

  // Theme-responsive classes for Card 1 (Overview)
  const cardBgClass = 'bg-[var(--bg-card)] border-[var(--border-color)]';

  const headerColorClass = isEmptyState
    ? 'text-[var(--text-secondary)]'
    : isHand
      ? 'text-amber-500 dark:text-amber-400 font-bold'
      : anomalyStatus.isAnomaly
        ? 'text-[var(--status-anomaly-text)]'
        : 'text-emerald-600 dark:text-emerald-400 font-bold';

  const headerIconClass = isEmptyState
    ? 'text-[var(--accent-pond)]'
    : isHand
      ? 'text-amber-500 dark:text-amber-400'
      : anomalyStatus.isAnomaly
        ? 'text-[var(--status-anomaly-text)]'
        : 'text-emerald-600 dark:text-emerald-400';

  return (
    <aside className="w-full lg:w-[21rem] xl:w-[23rem] 2xl:w-[25rem] h-auto lg:h-full flex-shrink-0 flex flex-col md:flex-row lg:flex-col gap-3 min-h-0 overflow-y-auto invisible-scrollbar items-stretch">

      {/* 1. INFERENCE & METRICS OVERVIEW CARD */}
      <div
        className={`p-3.5 rounded-2xl border shadow-xs flex flex-col w-full md:w-1/2 lg:w-full min-h-[220px] lg:min-h-[230px] shrink-0 justify-between overflow-hidden ${cardBgClass}`}
      >

        {/* Card Header */}
        <div className="flex items-center justify-between pb-2 border-b border-[var(--border-color)] shrink-0">
          <div className="flex items-center gap-2">
            {isEmptyState ? (
              <Activity className={`w-4 h-4 ${headerIconClass}`} />
            ) : isHand ? (
              <Hand className={`w-4 h-4 ${headerIconClass}`} />
            ) : anomalyStatus.isAnomaly ? (
              <AlertTriangle className={`w-4 h-4 ${headerIconClass}`} />
            ) : (
              <ShieldCheck className={`w-4 h-4 ${headerIconClass}`} />
            )}
            <span className={`font-semibold text-xs tracking-wider uppercase ${headerColorClass}`}>
              {isEmptyState ? 'Inference Details' : isHand ? 'Hand Present' : anomalyStatus.isAnomaly ? 'Anomaly Detection' : 'Normal'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {!isEmptyState && (
              <Badge variant={isHand ? 'warning' : anomalyStatus.isAnomaly ? 'anomaly' : 'normal'}>
                {isHand ? 'Hand Present' : anomalyStatus.isAnomaly ? 'Anomaly' : 'Normal'}
              </Badge>
            )}
            <IconButton
              size="sm"
              variant="ghost"
              aria-label="Close Drawer"
              title="Close Drawer"
              icon={<ChevronRight className="w-4 h-4 text-[var(--accent-pond)]" />}
              onClick={() => {
                playWaterDropSound();
                onToggle();
              }}
            />
          </div>
        </div>

        {/* Card 1 Body */}
        <div className="pt-2 flex-1 flex flex-col justify-between overflow-hidden">
          {isEmptyState ? (
            <div className="flex-1 flex flex-col items-center justify-center py-4 px-2 text-center text-[var(--text-secondary)]">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-pond-subtle)] border border-[var(--border-color)] flex items-center justify-center mb-3 shadow-xs">
                <Activity className="w-6 h-6 text-[var(--accent-pond)] animate-pulse" />
              </div>
              <p className="text-sm font-bold text-[var(--text-primary)]">
                {anomalyStatus.message === 'NO CAMERA'
                  ? 'Camera Offline'
                  : isStandby
                    ? 'Stream Inactive'
                    : 'Stream Active'}
              </p>
              <p className="text-xs mt-1 max-w-[230px] leading-relaxed text-[var(--text-secondary)]">
                {anomalyStatus.subMessage || (isStandby
                  ? 'Connect an OAK camera or switch to video mode to begin live YOLOv8 anomaly evaluation.'
                  : 'Waiting for YOLOv8 to detect objects...')}
              </p>

              <div className="mt-3 px-3.5 py-1.5 rounded-xl bg-[var(--accent-pond-subtle)] border border-[var(--border-color)] text-xs font-mono text-[var(--accent-pond)] flex items-center gap-2 shadow-2xs">
                <span className="w-2 h-2 rounded-full bg-[var(--accent-pond)]" />
                <span>Expected: <strong className="text-[var(--text-primary)] font-bold">{anomalyStatus.expectedCount}</strong> &middot; Detected: <strong className="text-[var(--text-primary)] font-bold">0</strong></span>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col justify-between overflow-hidden">
              {/* Large Numbers Area */}
              <div className="py-1 shrink-0">
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl sm:text-4xl font-bold text-[var(--text-primary)] leading-none">
                    {anomalyStatus.detectedCount}
                  </span>
                  <span className="text-xs sm:text-sm font-medium text-[var(--text-secondary)]">
                    detected &middot; expected
                  </span>
                  <span className="text-lg sm:text-xl font-bold text-[var(--text-secondary)]">
                    {anomalyStatus.expectedCount}
                  </span>
                </div>

                {isHand ? (
                  <div className="mt-1 flex items-center gap-1.5 text-xs font-semibold text-amber-600 dark:text-amber-400">
                    <Hand className="w-3.5 h-3.5 shrink-0 text-amber-500" />
                    <span>Evaluation paused &middot; Hand in target area</span>
                  </div>
                ) : anomalyStatus.difference !== 0 ? (
                  <div className={`mt-1 flex items-center gap-1.5 text-xs font-semibold ${anomalyStatus.isAnomaly ? 'text-[var(--status-anomaly-text)]' : 'text-[var(--status-normal-text)]'
                    }`}>
                    {anomalyStatus.difference > 0 ? (
                      <ArrowUp className="w-3.5 h-3.5 shrink-0" />
                    ) : (
                      <ArrowDown className="w-3.5 h-3.5 shrink-0" />
                    )}
                    <span>
                      {Math.abs(anomalyStatus.difference)} {anomalyStatus.difference > 0 ? 'above' : 'below'} expected count
                    </span>
                  </div>
                ) : (
                  <div className="mt-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                    <span>Perfect match with expected count</span>
                  </div>
                )}
              </div>

              {/* Warning Banner */}
              {anomalyStatus.foreignSpecies.length > 0 && (
                <div className="my-1.5 p-1.5 rounded-xl flex items-start gap-2 bg-[var(--status-warn-bg)] border border-[var(--status-warn-border)] text-[var(--status-warn-text)] text-[11px] font-medium shrink-0">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <span className="leading-tight">
                    {anomalyStatus.foreignCount ? `${anomalyStatus.foreignCount} ` : ''}Non-duck ({anomalyStatus.foreignSpecies.join(', ')}) detected
                  </span>
                </div>
              )}

              {/* Stream Metrics Block */}
              <div className="mt-1.5 shrink-0">
                <div className="flex justify-between items-center mb-1 px-0.5">
                  <div className="text-[11px]">
                    <span className="text-[var(--text-secondary)]">Species </span>
                    <span className="font-bold text-[var(--text-primary)]">{speciesList}</span>
                  </div>
                  <div className="text-[11px]">
                    <span className="text-[var(--text-secondary)]">Confidence </span>
                    <span className="font-bold text-[var(--accent-pond)]">{`${(metrics.avgConfidence * 100).toFixed(1)}%`}</span>
                  </div>
                </div>

                <div className="p-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-card-subtle)] flex flex-col gap-1.5">
                  <div className="flex justify-between items-center">

                    <Badge variant="active" size="sm" dot className="mr-1 hidden xl:inline-flex">
                      {isCameraSource ? 'LIVE CAMERA' : 'VIDEO INFERENCE'}
                    </Badge>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={mlStats.anchor_locked ? 'active' : 'warning'}
                        size="sm"
                      >
                        {mlStats.anchor_locked ? 'Lock: YES' : 'Lock: WARMING'}
                      </Badge>

                      <div className="font-mono text-xs">
                        <span className="font-bold text-[var(--text-primary)]">{displayFps.toFixed(1)}</span>
                        <span className="text-[var(--text-secondary)] ml-1">fps</span>
                      </div>
                    </div>
                  </div>

                  {/* Progress: Real Progress Bar for Video, Live Status for Infinite Camera */}
                  {isCameraSource ? (
                    <div className="flex justify-between items-center text-[10px] text-[var(--text-secondary)] pt-0.5">
                      <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-bold">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        Live Continuous Feed
                      </span>
                      <span className="font-mono">{displayFrames.toLocaleString()} frames</span>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <div className="flex justify-between items-center text-[10px] text-[var(--text-secondary)]">
                        <span>Processing Progress</span>
                        <span className="font-mono font-bold text-[var(--accent-pond)]">{Math.round(displayProgress)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-[var(--btn-secondary-border)] rounded-full overflow-hidden relative">
                        <div
                          className="h-full rounded-full bg-[var(--accent-pond)] transition-all duration-300 ease-out"
                          style={{ width: `${Math.max(0, Math.min(100, displayProgress))}%` }}
                        />
                      </div>
                      <div className="flex justify-between items-center font-mono text-[10px] text-[var(--text-secondary)]">
                        <span>{displayFrames.toLocaleString()} frames</span>
                        <span>{formatTime(computedUptime)}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 2. DETECTION GALLERY CARD (Row aligned on tablet, fixed height, strictly no inner scroll) */}
      <div
        className="p-3.5 rounded-2xl border border-[var(--border-color)] bg-[var(--bg-card)] shadow-xs flex flex-col w-full md:w-1/2 lg:w-full md:h-[370px] lg:h-auto min-h-[320px] flex-1 overflow-hidden justify-between"
      >
        <div className="flex items-center justify-between mb-2 pb-2 border-b border-[var(--border-color)] shrink-0 px-0.5">
          <div className="flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-[var(--accent-pond)]" />
            <span className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wider">
              Detection Gallery
            </span>
          </div>
          {hasDetections && (
            <div className="flex items-center gap-1.5">
              {anomalyDucks.length > 0 && (
                <span className="px-1.5 py-0.5 rounded-full text-[9px] font-bold bg-[var(--status-anomaly-bg)] text-[var(--status-anomaly-text)] border border-[var(--status-anomaly-border)]">
                  {anomalyDucks.length} Alert{anomalyDucks.length !== 1 ? 's' : ''}
                </span>
              )}
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border ${anomalyStatus.isAnomaly && (anomalyStatus.difference !== 0 || anomalyStatus.type === 'OVER_COUNT' || anomalyStatus.type === 'UNDER_COUNT')
                ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/30'
                : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
                }`}>
                {ducks.length} Total
              </span>
            </div>
          )}
        </div>

        <div className="w-full flex-1 min-h-0 flex flex-col overflow-hidden">
          {hasDetections ? (
            <DetectionGallery
              ducks={ducks}
              selectedDuckId={selectedDuckId}
              onSelectDuck={onSelectDuck}
              anomalyStatus={anomalyStatus}
              expectedCount={anomalyStatus.expectedCount}
            />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center py-4 px-4 text-center text-[var(--text-secondary)] bg-[var(--bg-card-subtle)] rounded-xl border border-[var(--border-color)]">
              <div className="w-12 h-12 rounded-2xl bg-[var(--status-normal-bg)] border border-[var(--status-normal-border)] flex items-center justify-center mb-3 shadow-xs">
                <ShieldCheck className="w-6 h-6 text-[var(--status-normal-text)]" />
              </div>
              <p className="text-sm font-bold text-[var(--text-primary)]">
                {isEmptyState ? 'Awaiting Feed Stream' : 'No Objects Detected'}
              </p>
              <p className="text-xs mt-1 text-[var(--text-secondary)] max-w-[220px] leading-relaxed">
                {isEmptyState
                  ? 'Detected objects will appear here ordered by ID.'
                  : 'Start the live stream to evaluate detections.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
