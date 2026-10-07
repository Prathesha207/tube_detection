import React from 'react';
import { CheckCircle2, Minimize2, Expand, Eye, EyeOff, Video, Disc, Clock, Loader2, Trash2, Tag, Hash, Target } from 'lucide-react';
import type { AnomalyStatus, LabelMode } from '../../types';
import { useInferenceStore } from '../../store/inferenceStore';

const formatRecordingTime = (totalSeconds: number) => {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};


interface TopToolbarProps {
  isRunning: boolean;
  hasActiveVideo: boolean;
  feedMode: 'raw' | 'inference';
  onFeedModeChange: (mode: 'raw' | 'inference') => void;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
  showHUD: boolean;
  onToggleHUD: () => void;
  backendStatus?: string;
  isCameraSource: boolean;
  isVideoSource: boolean;
  hasCameraRecording?: boolean;
  anomalyStatus: AnomalyStatus;
  isStreaming: boolean;
  isFirstFrameLoaded: boolean;
  framesProcessed?: number;
  onClearCustomVideo?: () => void;
  labelMode?: LabelMode;
  onLabelModeChange?: (mode: LabelMode) => void;
  zoom?: number;
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onResetZoom?: () => void;
  onSetZoom?: (zoom: number) => void;
  isRoiActive?: boolean;
  onToggleRoi?: () => void;
}


export const TopToolbar: React.FC<TopToolbarProps> = ({
  isRunning,
  hasActiveVideo,
  feedMode,
  onFeedModeChange,
  isFullscreen,
  onToggleFullscreen,
  showHUD,
  onToggleHUD,
  isCameraSource,
  isVideoSource: _isVideoSource,
  hasCameraRecording,
  anomalyStatus,
  isStreaming,
  framesProcessed = 0,
  onClearCustomVideo,
  labelMode,
  onLabelModeChange,
  zoom,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  onSetZoom,
  isRoiActive = false,
  onToggleRoi,
}) => {
  const latencyMs = useInferenceStore((state) => state.stats.latency_ms ?? state.stats.frame_time_ms);
  const isMediaActive = isRunning || hasActiveVideo || Boolean(hasCameraRecording) || (isCameraSource && isStreaming);
  if (!isMediaActive) return null;

  const hasInferenceResult = framesProcessed > 0 || (anomalyStatus.headsCount ?? 0) > 0 || (anomalyStatus.tailsCount ?? 0) > 0;
  const showInferenceStatus = isRunning || hasInferenceResult;

  return (
    <div className="absolute top-0 left-0 right-0 p-2 sm:p-3 flex items-center justify-between gap-2 pointer-events-none z-30 max-w-full overflow-hidden">
      {/* Top-Left Corner: Real-Time Status Badge or Live Stream Indicator */}
      <div className="pointer-events-auto flex items-center gap-2 min-w-0 shrink-0">
        {!isRoiActive && (
          showInferenceStatus ? (
            <div className="flex items-center h-7 sm:h-8 px-2.5 sm:px-3 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-xs shrink-0 select-none">
              {(anomalyStatus.message === 'WARMING' || anomalyStatus.message === 'WARMING UP' || (isRunning && (!isCameraSource || isStreaming) && framesProcessed > 0 && framesProcessed < 5)) ? (
                <div className="flex items-center gap-1.5 text-amber-500 font-bold text-xs sm:text-sm animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0 shadow-[0_0_6px_rgba(245,158,11,0.6)]" />
                  <span>WARMING UP</span>
                </div>
              ) : (anomalyStatus.message === 'LOADING MODEL' || (isRunning && (!isCameraSource || isStreaming) && framesProcessed === 0)) ? (
                <div className="flex items-center gap-1.5 text-amber-500 font-bold text-xs sm:text-sm animate-pulse">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-500 shrink-0" />
                  <span>{anomalyStatus.message === 'LOADING MODEL' ? 'LOADING MODEL...' : 'WARMING UP...'}</span>
                </div>
              ) : isRunning ? (
                <div className="flex items-center gap-2 font-bold text-xs sm:text-sm">
                  <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0 shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
                    <span>LIVE</span>
                  </div>
                  {latencyMs !== undefined && latencyMs !== null && Number(latencyMs) > 0 ? (
                    <span className="inline-flex items-center px-1.5 sm:px-2 py-0.5 rounded-lg text-xs font-mono font-bold text-emerald-700 dark:text-emerald-300 bg-emerald-500/15 border border-emerald-500/35 shadow-xs">
                      {Number(latencyMs).toFixed(1)} ms
                    </span>
                  ) : null}
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-[var(--text-secondary)] font-bold text-xs sm:text-sm">
                  <span className="w-2 h-2 rounded-full bg-[var(--accent-pond)] shrink-0" />
                  <span>{anomalyStatus.message || 'STOPPED'}</span>
                </div>
              )}
            </div>
          ) : isCameraSource && isStreaming ? (
            <div className="flex items-center h-7 sm:h-8 px-2.5 sm:px-3 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-xs shrink-0 select-none">
              <div className="flex items-center gap-1.5 text-sky-400 font-bold text-xs sm:text-sm">
                <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse shrink-0 shadow-[0_0_6px_rgba(56,189,248,0.6)]" />
                <span>LIVE STREAM</span>
              </div>
            </div>
          ) : null
        )}
      </div>

      {/* Top-Center: Label Mode View Switcher (Quadrant | Pins | Boxes) using App Theme */}
      {!isRoiActive && (showInferenceStatus || hasActiveVideo) && (feedMode === 'inference' || !isCameraSource || hasCameraRecording) && onLabelModeChange && (
        <div className="pointer-events-auto absolute left-1/2 -translate-x-1/2 top-2 sm:top-3 flex items-center h-7 sm:h-8 p-0.5 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-xs select-none z-30">
          <button
            type="button"
            onClick={() => onLabelModeChange('compact')}
            title="Quadrant Labels: labels point into North, South, East, West with zero overlap"
            className={`h-6 sm:h-7 px-2 sm:px-2.5 rounded-lg text-[10px] sm:text-xs font-bold tracking-wide flex items-center gap-1 sm:gap-1.5 cursor-pointer transition-all ${(labelMode ?? 'compact') === 'compact'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs scale-100'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <Tag className="w-3 h-3" />
            <span>Quadrant</span>
          </button>

          <button
            type="button"
            onClick={() => onLabelModeChange('pins')}
            title="Pins Mode: tiny numbered circle pins (#1, #2) with zero obstruction"
            className={`h-6 sm:h-7 px-2 sm:px-2.5 rounded-lg text-[10px] sm:text-xs font-bold tracking-wide flex items-center gap-1 sm:gap-1.5 cursor-pointer transition-all ${labelMode === 'pins'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs scale-100'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <Hash className="w-3 h-3" />
            <span>Pins</span>
          </button>

          <button
            type="button"
            onClick={() => onLabelModeChange('hidden')}
            title="Boxes Mode: pure bounding boxes with crosshairs and 0 text"
            className={`h-6 sm:h-7 px-2 sm:px-2.5 rounded-lg text-[10px] sm:text-xs font-bold tracking-wide flex items-center gap-1 sm:gap-1.5 cursor-pointer transition-all ${labelMode === 'hidden'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs scale-100'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <EyeOff className="w-3 h-3" />
            <span>Boxes</span>
          </button>
        </div>
      )}

      {/* Top-Right Corner: Action Controls */}
      <div className="pointer-events-auto flex items-center gap-1 sm:gap-1.5 flex-nowrap justify-end shrink-0">
        {!isRoiActive && showInferenceStatus && (
          <>
            {/* Feed toggle pill: RAW vs INFERENCE - Only shown for OAK camera */}
            {isCameraSource && (
              <div className="flex items-center h-7 sm:h-8 p-0.5 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-xs shrink-0">
                <button
                  onClick={() => {
                    onFeedModeChange('raw');
                  }}
                  className={`h-6 sm:h-7 px-2 sm:px-2.5 rounded-lg text-[10px] sm:text-xs font-bold tracking-wide flex items-center justify-center cursor-pointer transition-all ${feedMode === 'raw'
                    ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs scale-100'
                    : 'text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
                    }`}
                >
                  RAW
                </button>
                <button
                  onClick={() => {
                    onFeedModeChange('inference');
                  }}
                  className={`h-6 sm:h-7 px-2 sm:px-2.5 rounded-lg text-[10px] sm:text-xs font-bold tracking-wide flex items-center justify-center cursor-pointer transition-all ${feedMode === 'inference'
                    ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs scale-100'
                    : 'text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
                    }`}
                >
                  INFERENCE
                </button>
              </div>
            )}


          </>
        )}

        {/* ROI Region of Interest Mask Toggle: Clean Icon Button [ 🎯 ] */}
        {onToggleRoi && (
          <button
            onClick={onToggleRoi}
            aria-label={isRoiActive ? "Close ROI Mask Editor" : "Open ROI Mask Editor"}
            title={isRoiActive ? "Close ROI Mask Editor (🎯)" : "Region of Interest Mask Editor (🎯)"}
            className={`w-7 sm:w-8 h-7 sm:h-8 flex items-center justify-center rounded-xl backdrop-blur-md border active:scale-95 shrink-0 shadow-xs cursor-pointer transition-all ${
              isRoiActive
                ? 'bg-cyan-500/25 border-cyan-400 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.5)] ring-1 ring-cyan-400/50'
                : 'bg-[var(--btn-secondary-bg)] border-[var(--btn-secondary-border)] text-[var(--text-primary)] hover:text-cyan-400 hover:bg-[var(--btn-secondary-hover)] hover:border-cyan-500/30'
            }`}
          >
            <Target className={`w-4 h-4 transition-transform ${isRoiActive ? 'scale-110 text-cyan-300' : 'text-[var(--text-primary)] dark:text-white'}`} />
          </button>
        )}

        {/* Fullscreen Button */}
        <button
          onClick={onToggleFullscreen}
          aria-label={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
          title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
          className="w-7 sm:w-8 h-7 sm:h-8 flex items-center justify-center rounded-xl bg-[var(--btn-secondary-bg)] backdrop-blur-md border border-[var(--btn-secondary-border)] text-[var(--text-primary)] hover:text-white hover:bg-[var(--btn-secondary-hover)] active:scale-95 shrink-0 shadow-xs cursor-pointer"
        >
          {isFullscreen ? (
            <Minimize2 className="w-4 h-4 text-[var(--text-primary)] dark:text-white" />
          ) : (
            <Expand className="w-4 h-4 text-[var(--text-primary)] dark:text-white" />
          )}
        </button>

        {/* Quick HUD Visibility Toggle */}
        {/* <button
          onClick={onToggleHUD}
          aria-label={showHUD ? 'Hide HUD overlay' : 'Show HUD overlay'}
          title={showHUD ? 'Hide HUD overlay' : 'Show HUD overlay'}
          className="w-7 sm:w-8 h-7 sm:h-8 flex items-center justify-center rounded-xl bg-[var(--btn-secondary-bg)] backdrop-blur-md border border-[var(--btn-secondary-border)] text-[var(--text-primary)] hover:text-white hover:bg-[var(--btn-secondary-hover)] active:scale-95 shrink-0 shadow-xs cursor-pointer"
        >
          {showHUD ? (
            <Eye className="w-4 h-4 text-[var(--text-primary)] dark:text-white" />
          ) : (
            <EyeOff className="w-4 h-4 text-[var(--text-muted)] dark:text-white/70" />
          )}
        </button> */}

        {/* Clear Video button when video is loaded and stopped */}
        {!isRoiActive && !isCameraSource && hasActiveVideo && !isRunning && onClearCustomVideo && (
          <button
            onClick={onClearCustomVideo}
            title="Clear loaded video and upload a new one"
            className="h-7 sm:h-8 px-2 sm:px-2.5 flex items-center gap-1.5 rounded-xl bg-[var(--btn-secondary-bg)] hover:bg-rose-500/15 border border-[var(--border-color)] hover:border-rose-500/40 text-[var(--text-primary)] hover:text-rose-600 dark:hover:text-rose-400 font-bold text-[10px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all shrink-0"
          >
            <Trash2 className="w-3.5 h-3.5 text-rose-500" />
            <span className="hidden sm:inline">Clear Video</span>
          </button>
        )}
      </div>
    </div>
  );
};
