import React from 'react';
import { StreamSourceType } from '../types';
import { Video, Camera, Play, Square, Loader2 } from 'lucide-react';
import { playWaterDropSound } from '../utils/audio';
import { NumberStepper } from './ui/NumberStepper';
import { useInferenceStore } from '../store/inferenceStore';


interface SourceSelectorProps {
  sourceType: StreamSourceType;
  onSourceChange: (type: StreamSourceType) => void;
  expectedDucks: number;
  onExpectedDucksChange: (count: number) => void;
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
  expectedDucks,
  onExpectedDucksChange,
  onOpenSettings: _onOpenSettings,
  customVideoName,
  customVideoUrl,
  hasActiveVideo = false,
  onClearCustomVideo: _onClearCustomVideo,
  isRunning = false,
  isStarting = false,
  isRecording = false,
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

  const handleSourceClick = (targetType: StreamSourceType) => {
    playWaterDropSound();
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
            onClick={() => handleSourceClick('uploaded-video')}
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all cursor-pointer ${sourceType === 'uploaded-video'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <Video className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>Video<span className="hidden sm:inline"> File</span></span>
          </button>

          <button
            onClick={() => handleSourceClick('oak-camera')}
            className={`flex items-center justify-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all cursor-pointer ${sourceType === 'oak-camera'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
              }`}
          >
            <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>OAK<span className="hidden sm:inline"> Camera</span></span>
          </button>
        </div>

        {/* Global Expected Count & Stream Controls */}
        <div className="flex items-center justify-end gap-1.5 sm:gap-4 flex-shrink-0 ml-auto">
          {/* Expected Ducks Control */}
          <div className="flex items-center gap-1 sm:gap-3">
            <div className="flex items-center gap-1 sm:gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-pond)] animate-pulse" />
              <span className="text-[10px] sm:text-xs font-bold text-[var(--text-primary)]">
                <span className="inline sm:hidden">Exp:</span>
                <span className="hidden sm:inline">Expected Ducks:</span>
              </span>
            </div>
            <NumberStepper
              value={expectedDucks}
              onChange={(val) => {
                playWaterDropSound();
                onExpectedDucksChange(val);
              }}
              min={1}
              max={50}
            />
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
                /* When Running: provide STOP INFERENCE button */
                <div className="flex items-center gap-1 sm:gap-2">
                  <button
                    onClick={() => {
                      playWaterDropSound();
                      if (onStopInference) onStopInference();
                      else if (onToggleRunning) onToggleRunning();
                    }}
                    title="Stop AI inference"
                    className="h-8 sm:h-9 flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-4 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all"
                  >
                    <Square className="w-3 h-3 sm:w-3.5 sm:h-3.5 fill-current" />
                    <span>STOP<span className="hidden sm:inline"> INFERENCE</span></span>
                  </button>
                  {isCameraMode && isStreaming && !cameraRecordSessionId && (
                    <button
                      onClick={() => {
                        playWaterDropSound();
                        onStopStream?.();
                      }}
                      title="Stop camera stream and clear feed"
                      className="h-8 sm:h-9 flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-3.5 rounded-xl bg-slate-600 hover:bg-slate-500 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all"
                    >
                      <Square className="w-3 h-3 sm:w-3.5 sm:h-3.5 fill-current" />
                      <span>STOP<span className="hidden sm:inline"> STREAM</span></span>
                    </button>
                  )}
                </div>
              ) : (
                /* When Stopped / Paused: provide START INFERENCE (and START/STOP STREAM for Camera) */
                <div className="flex items-center gap-1 sm:gap-1.5">
                  <button
                    disabled={isStarting || isVideoLoading}
                    onClick={() => {
                      if (isStarting || isVideoLoading) return;
                      playWaterDropSound();
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

                  {/* For Camera: allow streaming-only if user wants to align/view camera without AI */}
                  {isCameraMode && !cameraRecordSessionId && (
                    !isStreaming ? (
                      <button
                        onClick={() => {
                          playWaterDropSound();
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
                          playWaterDropSound();
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
