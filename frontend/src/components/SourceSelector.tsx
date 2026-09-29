import React from 'react';
import { StreamSourceType } from '../types';
import { Video, Camera, Play, Square, Loader2, Trash2, Disc, Clock } from 'lucide-react';
import { useInferenceStore } from '../store/inferenceStore';

const formatTime = (totalSeconds: number = 0) => {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

interface SourceSelectorProps {
  sourceType: StreamSourceType;
  onSourceChange: (type: StreamSourceType) => void;
  onOpenSettings?: () => void;
  onCustomVideoUploaded?: (videoUrl: string, fileName: string, sessionId?: string) => void;
  customVideoName?: string;
  customVideoUrl?: string;
  videoSessionId?: string | null;
  hasActiveVideo?: boolean;
  onClearCustomVideo?: () => void;

  isRunning?: boolean;
  isStarting?: boolean;
  isRecording?: boolean;
  isSavingRecording?: boolean;
  recordingDuration?: number;
  onToggleRecording?: () => void;
  onToggleRunning?: () => void;
  onStopInference?: () => void;
  onResumeInference?: () => void;
  onRequestSwitchMode?: (type: StreamSourceType) => void;
  isStreaming?: boolean;
  onStartStream?: () => void;
  onStopStream?: () => void;
  isCameraConnected?: boolean;
  cameraStartingState?: 'idle' | 'waking_camera' | 'waiting_frame' | 'ready';
  cameraRecordSessionId?: string | null;
  onClearCameraRecord?: () => void;
}

export const SourceSelector: React.FC<SourceSelectorProps> = ({
  sourceType,
  onSourceChange,
  onOpenSettings: _onOpenSettings,
  customVideoName,
  customVideoUrl,
  hasActiveVideo = false,
  onClearCustomVideo,
  isRunning = false,
  isStarting = false,
  isRecording = false,
  isSavingRecording = false,
  recordingDuration = 0,
  onToggleRecording,
  onToggleRunning,
  onStopInference,
  onResumeInference,
  onRequestSwitchMode,
  isStreaming = false,
  onStartStream,
  onStopStream,
  isCameraConnected = true,
  cameraStartingState = 'ready',
  cameraRecordSessionId,
  onClearCameraRecord,
}) => {
  const isVideoLoading = useInferenceStore((state) => state.isVideoLoading);
  const storeHeads = useInferenceStore((state) => state.stats.heads_count ?? 0);
  const storeTails = useInferenceStore((state) => state.stats.tails_count ?? 0);

  const handleSourceClick = (targetType: StreamSourceType) => {
    if (isRunning) return;
    if (onRequestSwitchMode) {
      onRequestSwitchMode(targetType);
    } else {
      onSourceChange(targetType);
    }
  };

  const isVideoMode = sourceType === 'sample-pond' || sourceType === 'uploaded-video';
  const isCameraMode = sourceType === 'oak-camera' || sourceType === 'webcam';
  // Always show controls in video mode; in camera mode show if hardware is online
  const hasMediaToPlay = (isCameraMode && isCameraConnected) || isVideoMode;
  const isBackendStream = Boolean(customVideoUrl && customVideoUrl.includes('/video/stream/'));

  return (
    <div className="w-full flex-shrink-0">
      <div className="flex flex-row items-center justify-between gap-1.5 sm:gap-4 p-1.5 sm:p-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-sm">

        {/* Source Toggle Pills */}
        <div className="flex bg-[var(--bg-card-subtle)] p-0.5 sm:p-1 rounded-xl border border-[var(--border-color)] flex-shrink-0">
          <button
            disabled={isRunning}
            onClick={() => handleSourceClick('uploaded-video')}
            title={isRunning ? 'Stop inference before switching source' : 'Switch to Video File mode'}
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all ${
              isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
            } ${sourceType === 'uploaded-video'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <Video className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>Video<span className="hidden sm:inline"> File</span></span>
          </button>

          <button
            disabled={isRunning}
            onClick={() => handleSourceClick('oak-camera')}
            title={isRunning ? 'Stop inference before switching source' : 'Switch to OAK Camera mode'}
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all ${
              isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
            } ${sourceType === 'oak-camera'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>OAK<span className="hidden sm:inline"> Camera</span></span>
          </button>
        </div>

        {/* Global Tube Ends Count & Stream Controls */}
        <div className="flex items-center justify-end gap-1.5 sm:gap-4 flex-shrink-0 ml-auto">
          {/* Tube Ends Counter */}
          <div className="flex items-center gap-2 sm:gap-3 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)]">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[10px] sm:text-xs font-semibold text-[var(--text-secondary)]">Heads:</span>
              <span className="text-xs sm:text-sm font-bold font-mono text-[var(--text-primary)]">{storeHeads}</span>
            </div>
            <span className="text-xs text-[var(--border-color)]">•</span>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[10px] sm:text-xs font-semibold text-[var(--text-secondary)]">Tails:</span>
              <span className="text-xs sm:text-sm font-bold font-mono text-[var(--text-primary)]">{storeTails}</span>
            </div>
          </div>

          {/* Stream Play/Stop Controls */}
          {hasMediaToPlay && (
            <div className="flex items-center gap-1 sm:gap-1.5 pl-1.5 sm:pl-4 border-l border-[var(--border-color)]">
              {isCameraMode && cameraStartingState !== 'ready' ? (
                /* When Camera is Actively Waking up or Warming: provide busy state */
                <button
                  disabled
                  className="h-8 sm:h-9 flex items-center gap-1.5 px-3.5 sm:px-4 rounded-xl bg-amber-600/80 text-white font-bold text-[11px] sm:text-xs shadow-xs cursor-wait opacity-90 transition-all shrink-0"
                >
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>{cameraStartingState === 'waking_camera' ? 'WAKING CAMERA...' : 'WARMING UP...'}</span>
                </button>
              ) : isStarting ? (
                /* When Starting inference: provide busy state */
                <button
                  disabled
                  className="h-8 sm:h-9 flex items-center gap-1.5 px-3.5 sm:px-4 rounded-xl bg-emerald-600/80 text-white font-bold text-[11px] sm:text-xs shadow-xs cursor-wait opacity-90 transition-all shrink-0"
                >
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>STARTING...</span>
                </button>
              ) : isRunning ? (
                /* When Running: provide ONLY STOP INFERENCE button */
                <div className="flex items-center gap-1 sm:gap-2">
                  <button
                    onClick={() => {
                      if (onStopInference) onStopInference();
                      else if (onToggleRunning) onToggleRunning();
                    }}
                    title="Stop AI inference"
                    className="h-8 sm:h-9 flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-4 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all"
                  >
                    <Square className="w-3 h-3 sm:w-3.5 sm:h-3.5 fill-current" />
                    <span>STOP<span className="hidden sm:inline"> INFERENCE</span></span>
                  </button>
                </div>
              ) : (
                /* When Stopped / Paused: provide START INFERENCE (and START/STOP STREAM for Camera) */
                <div className="flex items-center gap-1 sm:gap-1.5">
                  <button
                    disabled={isStarting || isVideoLoading}
                    onClick={() => {
                      if (isStarting || isVideoLoading) return;
                      if (onResumeInference) onResumeInference();
                      else if (onToggleRunning) onToggleRunning();
                    }}
                    title={
                      isVideoLoading
                        ? "Video is uploading... please wait"
                        : isStarting
                          ? "Starting inference... please wait"
                          : isCameraMode && !isStreaming
                            ? "Start camera stream and real-time AI inference"
                            : "Start real-time YOLOv8 AI inference"
                    }
                    className={`h-8 sm:h-9 flex items-center gap-1 sm:gap-2 px-2.5 sm:px-5 rounded-xl font-bold text-[11px] sm:text-xs shadow-sm transition-all shrink-0 ${isStarting || isVideoLoading
                      ? "bg-emerald-950/80 text-emerald-400/80 border border-emerald-700/50 cursor-not-allowed opacity-80"
                      : "bg-emerald-600 hover:bg-emerald-500 text-white hover:shadow-emerald-600/30 active:scale-95 cursor-pointer"
                      }`}
                  >
                    {isStarting || isVideoLoading ? (
                      <Loader2 className="w-3 h-3 sm:w-3.5 sm:h-3.5 animate-spin text-emerald-400" />
                    ) : (
                      <Play className="w-3 h-3 sm:w-3.5 sm:h-3.5 fill-current" />
                    )}
                    <span className="whitespace-nowrap">
                      {isVideoLoading ? "UPLOADING..." : isStarting ? "STARTING..." : (
                        <>START<span className="hidden sm:inline"> INFERENCE</span></>
                      )}
                    </span>
                  </button>

                  {/* For Camera: Record Button next to START INFERENCE when stream is active and inference not running */}
                  {isCameraMode && isStreaming && !isRunning && !cameraRecordSessionId && (
                    <button
                      disabled={isSavingRecording}
                      onClick={() => onToggleRecording?.()}
                      title={
                        isSavingRecording
                          ? 'Finalizing recording... please wait'
                          : isRecording
                            ? 'Stop camera recording'
                            : 'Start recording camera stream'
                      }
                      className={`h-8 sm:h-9 flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3.5 rounded-xl font-bold text-[11px] sm:text-xs shadow-xs transition-all shrink-0 cursor-pointer active:scale-95 ${
                        isSavingRecording
                          ? 'bg-rose-700/80 text-white cursor-wait opacity-80'
                          : isRecording
                            ? 'bg-rose-600 hover:bg-rose-500 text-white animate-pulse ring-2 ring-rose-400/50 shadow-md shadow-rose-600/30'
                            : 'bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/40 text-rose-500 dark:text-rose-400 hover:text-rose-600 dark:hover:text-white'
                      }`}
                    >
                      {isSavingRecording ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 animate-spin text-white shrink-0" />
                          <span className="whitespace-nowrap uppercase tracking-wider text-[10px] sm:text-xs">SAVING...</span>
                        </>
                      ) : isRecording ? (
                        <>
                          <span className="relative flex h-2 w-2 shrink-0">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-80" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                          </span>
                          <Disc className="w-3.5 h-3.5 text-white shrink-0 animate-spin [animation-duration:3s]" />
                          <span className="whitespace-nowrap font-black tracking-wider text-white">REC</span>
                          <span className="font-mono text-[10px] sm:text-[11px] font-bold bg-black/40 border border-white/20 px-1.5 py-0.2 rounded-md text-white">
                            {formatTime(recordingDuration)}
                          </span>
                        </>
                      ) : (
                        <>
                          <span className="w-2 h-2 rounded-full bg-rose-500 shrink-0 shadow-[0_0_6px_rgba(244,63,94,0.8)]" />
                          <Video className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400 shrink-0" />
                          <span className="whitespace-nowrap">REC</span>
                        </>
                      )}
                    </button>
                  )}

                  {/* For Uploaded Video: Allow clearing the video so the upload card reappears */}
                  {!isCameraMode && hasActiveVideo && onClearCustomVideo && (
                    <button
                      onClick={() => onClearCustomVideo()}
                      title="Clear loaded video and upload a new one"
                      className="h-8 sm:h-9 flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-3 rounded-xl bg-[var(--btn-secondary-bg)] border border-[var(--btn-secondary-border)] text-[var(--btn-secondary-text)] hover:bg-rose-500/15 hover:border-rose-500/30 hover:text-rose-600 dark:hover:text-rose-400 font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all shrink-0"
                    >
                      <Trash2 className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
                      <span className="whitespace-nowrap">CLEAR<span className="hidden sm:inline"> VIDEO</span></span>
                    </button>
                  )}

                  {/* For Camera: allow streaming-only if user wants to align/view camera without AI */}
                  {isCameraMode && !cameraRecordSessionId && (
                    !isStreaming ? (
                      <button
                        onClick={() => {
                          onStartStream?.();
                        }}
                        title="Start camera stream only (no AI inference)"
                        className="h-8 sm:h-9 flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-4 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all shrink-0"
                      >
                        <Play className="w-3 h-3 sm:w-3.5 sm:h-3.5 fill-current" />
                        <span className="whitespace-nowrap">START<span className="hidden sm:inline"> STREAM</span></span>
                      </button>
                    ) : (
                      <button
                        onClick={() => {
                          onStopStream?.();
                        }}
                        title="Stop camera stream"
                        className="h-8 sm:h-9 flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-4 rounded-xl bg-slate-600 hover:bg-slate-500 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all"
                      >
                        <Square className="w-3 h-3 sm:w-3.5 sm:h-3.5 fill-current" />
                        <span>STOP<span className="hidden sm:inline"> STREAM</span></span>
                      </button>
                    )
                  )}
                </div>
              )}


            </div>
          )}
        </div>
      </div>
    </div>
  );
};
