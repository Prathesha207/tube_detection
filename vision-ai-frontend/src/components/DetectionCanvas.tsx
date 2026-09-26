import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import type { TubeEntity, StreamSourceType, AnomalyStatus, LabelMode, CameraConfig } from '../types';
import { getApiBaseUrl } from '../lib/api';
import { useInferenceStore } from '../store/inferenceStore';
import { useRecording } from './hooks/useRecording';
import { cameraService } from './service/cameraService';

import { Loader2 } from 'lucide-react';

// Extracted Canvas Components
import { BoundingBoxOverlay } from './canvas/BoundingBoxOverlay';
import { VideoUploadCard } from './canvas/VideoUploadCard';
import { CameraOfflineCard } from './canvas/CameraOfflineCard';
import { CameraStandbyCard } from './canvas/CameraStandbyCard';
import { TopToolbar } from './canvas/TopToolbar';
import { StatusBar } from './canvas/StatusBar';
import { LoadingOverlay } from './canvas/LoadingOverlay';
import { ZoomControls } from './canvas/ZoomControls';

// Extracted Canvas Hooks
import { useFullscreen } from '../hooks/useFullscreen';
import { useContainerFit } from '../hooks/useContainerFit';
import { useVideoUpload } from '../hooks/useVideoUpload';
import { useCanvasZoom } from '../hooks/useCanvasZoom';

interface DetectionCanvasProps {
  tubes: TubeEntity[];
  setTubes?: React.Dispatch<React.SetStateAction<TubeEntity[]>>;
  anomalyStatus: AnomalyStatus;
  feedMode: 'raw' | 'inference';
  onFeedModeChange: (mode: 'raw' | 'inference') => void;
  isRunning: boolean;
  isStarting?: boolean;
  onToggleRunning?: () => void;
  onStopInference?: () => void;
  onResumeInference?: () => void;
  isStreaming?: boolean;
  onStartStream?: () => void;
  onRequestSwitchMode?: (type: StreamSourceType) => void;
  fps: number;
  sourceType: StreamSourceType;
  customVideoUrl?: string;
  videoSessionId?: string | null;
  customVideoName?: string;
  selectedTubeId: string | null;
  onSelectTube: (id: string | null) => void;
  onCustomVideoUploaded?: (videoUrl: string, fileName: string, sessionId?: string, isCameraRecording?: boolean) => void | Promise<void>;
  onClearCustomVideo?: () => void;
  cameraStartingState?: 'idle' | 'waking_camera' | 'waiting_frame' | 'ready';
  onCameraDeviceChange?: (active: boolean) => void;
  videoDimensions?: { width: number; height: number } | null;
  isCameraConnected?: boolean;
  initialUploadFile?: File;
  isBackendConnected?: boolean;
  onRegisterTriggerUpload?: (trigger: () => void) => void;
  lastCameraFrame?: string;
  lastVideoFrame?: string;
  onCaptureVideoFrame?: (frame: string) => void;
  onCaptureCameraFrame?: (frame: string) => void;
  onRetryConnection?: () => void;
  framesProcessed?: number;
  cameraRecordSessionId?: string | null;
  cameraRecordUrl?: string;
  cameraRecordName?: string;
  onClearCameraRecord?: () => void;
  cameraError?: string | null;
  cameraTargetFps?: number;
  recordingFormat?: 'AVI' | 'MP4' | 'FFV1';
  cameraConfig?: CameraConfig;
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  tubes,
  setTubes,
  anomalyStatus,
  feedMode,
  onFeedModeChange,
  isRunning,
  isStarting,
  cameraConfig,
  onToggleRunning,
  onStopInference,
  onResumeInference,
  isStreaming = false,
  onStartStream,
  onRequestSwitchMode,
  fps,
  sourceType,
  customVideoUrl,
  videoSessionId,
  customVideoName,
  selectedTubeId,
  onSelectTube,
  onCustomVideoUploaded,
  onClearCustomVideo,
  cameraStartingState = 'ready',
  onCameraDeviceChange,
  videoDimensions,
  isCameraConnected = false,
  initialUploadFile,
  isBackendConnected = true,
  onRegisterTriggerUpload,
  lastCameraFrame,
  lastVideoFrame,
  onCaptureVideoFrame,
  onCaptureCameraFrame,
  onRetryConnection,
  framesProcessed = 0,
  cameraRecordSessionId,
  cameraRecordUrl,
  cameraRecordName,
  onClearCameraRecord,
  cameraTargetFps,
  recordingFormat = 'MP4',
  cameraError,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraImgRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [showHUD, setShowHUD] = useState(true);
  const [labelMode, setLabelMode] = useState<LabelMode>('compact');
  const [videoAspect, setVideoAspect] = useState<number | null>(null);
  const [isFirstFrameLoaded, setIsFirstFrameLoaded] = useState<boolean>(false);
  const [streamCacheBuster, setStreamCacheBuster] = useState<number>(Date.now());
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    };
  }, []);

  const { isRecording, isSaving, recordedFile, recordingDuration, startRecording, stopRecording, clearRecording } = useRecording();
  const backendStats = useInferenceStore((state) => state.stats);

  const isVideoSource = sourceType === 'uploaded-video' || sourceType === 'sample-pond';
  const isCameraSource = sourceType === 'oak-camera' || sourceType === 'webcam';
  const hasCameraRecording = isCameraSource && Boolean(cameraRecordSessionId);
  const hasActiveVideo = isVideoSource && Boolean(customVideoUrl || videoSessionId);
  const isWaitingForVideo = isVideoSource && !hasActiveVideo && !isRunning;
  const isCameraOffline = isCameraSource && !isCameraConnected && !hasCameraRecording;
  const isMediaActive = Boolean(hasActiveVideo || videoSessionId || hasCameraRecording || (isCameraSource && isCameraConnected && isStreaming) || (isCameraSource && isRunning) || (isVideoSource && isRunning));

  // Canvas Zoom & Pan
  const {
    zoom,
    pan: _pan,
    isPanning: _isPanning,
    zoomIn,
    zoomOut,
    setZoomLevel,
    resetZoom,
    viewportRef,
    transformStyle,
    handleMouseDown,
    handleDoubleClick,
    cursorStyle,
    hasMoved,
  } = useCanvasZoom(isMediaActive);

  // Reset zoom whenever stream or video source changes
  useEffect(() => {
    resetZoom();
  }, [sourceType, customVideoUrl, cameraRecordSessionId, resetZoom]);

  // Ensure camera stream reconnects cleanly whenever running state changes while streaming
  useEffect(() => {
    if (isCameraSource && isStreaming) {
      setStreamCacheBuster(Date.now());
    }
  }, [isRunning, isCameraSource, isStreaming]);

  const effectiveFramesProcessed = framesProcessed || backendStats?.frames_processed || 0;
  const hasInferenceResult = (effectiveFramesProcessed > 0 || tubes.length > 0) && tubes.length > 0;

  const isOverlayShowing =
    Boolean(isStarting) ||
    Boolean(isCameraSource && !hasCameraRecording && isCameraConnected && cameraStartingState !== 'ready') ||
    Boolean((isVideoSource || hasCameraRecording) && (hasActiveVideo || hasCameraRecording) && isRunning && !isFirstFrameLoaded);

  // Hooks
  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef);
  const { fittedRect } = useContainerFit(containerRef, canvasRef, videoAspect, videoDimensions, isCameraSource);
  const { uploadProgress, isSelectingVideo, handleFileInputChange, handleSelectVideoAndStart, handleDragOver, handleDragLeave, handleDrop, isDragOver } = useVideoUpload(fileInputRef, onCustomVideoUploaded, recordedFile, clearRecording, initialUploadFile);
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (hasMoved) return; // Do not trigger click/deselection if user was dragging to pan
    if (e.target !== e.currentTarget && (e.target as HTMLElement).closest('[role="button"]')) {
      return;
    }
    if (selectedTubeId) {
      onSelectTube(null);
    }
  };

  useEffect(() => {
    if (onRegisterTriggerUpload) {
      onRegisterTriggerUpload(handleSelectVideoAndStart);
    }
  }, [handleSelectVideoAndStart, onRegisterTriggerUpload]);

  // Cache buster for stream URL
  useEffect(() => {
    setStreamCacheBuster(Date.now());
  }, [isRunning, videoSessionId, cameraRecordSessionId]);

  useEffect(() => {
    if (backendStats?.status === 'stopped' && !isRunning) {
      setStreamCacheBuster(Date.now());
    }
  }, [backendStats?.status, isRunning]);

  // When window is un-minimized or focused, immediately refresh stream URL to display latest frame
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        setStreamCacheBuster(Date.now());
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('focus', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', handleVisibilityChange);
    };
  }, []);

  const effectiveVideoUrl = useMemo(() => {
    if (!hasActiveVideo) return undefined;
    if (videoSessionId) {
      if (isRunning) {
        return `${getApiBaseUrl()}/video/stream/${videoSessionId}?t=${streamCacheBuster}`;
      }
      return `${getApiBaseUrl()}/video/last_frame/${videoSessionId}?t=${streamCacheBuster}`;
    }
    return customVideoUrl;
  }, [customVideoUrl, hasActiveVideo, isRunning, videoSessionId, streamCacheBuster]);

  const [streamError, setStreamError] = useState<boolean>(false);

  useEffect(() => {
    setStreamError(false);
  }, [effectiveVideoUrl, streamCacheBuster, isRunning, sourceType]);

  const activeLastFrame = isCameraSource ? lastCameraFrame : lastVideoFrame;
  const fallbackLastFrameUrl = !isRunning && isVideoSource && videoSessionId
    ? `${getApiBaseUrl()}/video/last_frame/${videoSessionId}?t=${streamCacheBuster}`
    : undefined;
  const isVideoCompleted = backendStats?.status === 'completed';
  const effectiveBackdrop = isVideoCompleted && fallbackLastFrameUrl
    ? fallbackLastFrameUrl
    : (activeLastFrame || fallbackLastFrameUrl);

  const captureFrame = useCallback(() => {
    const img = cameraImgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) return;
    if (!isCameraSource && effectiveFramesProcessed === 0) return;
    try {
      const offscreen = document.createElement('canvas');
      offscreen.width = img.naturalWidth;
      offscreen.height = img.naturalHeight;
      const ctx = offscreen.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        const dataUrl = offscreen.toDataURL('image/jpeg', 0.85);
        if (dataUrl && dataUrl.length > 200) {
          if (isCameraSource) {
            onCaptureCameraFrame?.(dataUrl);
          } else {
            onCaptureVideoFrame?.(dataUrl);
          }
        }
      }
    } catch {
      // Ignore if tainted or cross-origin canvas security restriction
    }
  }, [isCameraSource, effectiveFramesProcessed, onCaptureCameraFrame, onCaptureVideoFrame]);

  // When inference starts, clear any stale captured video frame
  useEffect(() => {
    if (isRunning && !isCameraSource) {
      onCaptureVideoFrame?.(undefined as any);
    }
  }, [isRunning, isCameraSource, onCaptureVideoFrame]);

  // When inference stops, capture the current frame
  useEffect(() => {
    if (!isRunning && (videoSessionId || hasActiveVideo) && effectiveFramesProcessed > 0) {
      const timer = setTimeout(() => {
        captureFrame();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isRunning, videoSessionId, hasActiveVideo, effectiveFramesProcessed, captureFrame]);

  // Reset states on source change
  useEffect(() => {
    setVideoAspect(null);
  }, [effectiveVideoUrl, hasActiveVideo, sourceType]);

  useEffect(() => {
    if (isRunning) {
      setIsFirstFrameLoaded(false);
      // Fallback timeout: ensure overlay stays visible while stream connects and frames buffer
      const timer = setTimeout(() => {
        setIsFirstFrameLoaded(true);
      }, 15000);
      return () => clearTimeout(timer);
    }
  }, [isRunning]);

  // Once backend starts processing frames or tubes arrive, mark first frame loaded immediately
  useEffect(() => {
    if ((backendStats?.frames_processed && backendStats.frames_processed > 0) || tubes.length > 0) {
      setIsFirstFrameLoaded(true);
    }
    if (backendStats?.video_width && backendStats?.video_height) {
      setVideoAspect(backendStats.video_width / backendStats.video_height);
    }
  }, [backendStats?.frames_processed, backendStats?.video_width, backendStats?.video_height, tubes.length]);

  // Video autoplay behavior for local preview
  useEffect(() => {
    if (!videoRef.current) return;
    if (hasActiveVideo && !videoSessionId) {
      videoRef.current.muted = true;
      videoRef.current.play().catch(() => { });
    } else {
      videoRef.current.pause();
    }
  }, [hasActiveVideo, customVideoUrl, videoSessionId]);

  return (
    <div
      ref={containerRef}
      id="detection-hero-viewport"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={(e) => handleDrop(e, isVideoSource)}
      className={`relative w-full flex-1 h-full min-h-[350px] lg:min-h-0 overflow-hidden border select-none group transition-colors duration-200 ${
        isFullscreen ? 'rounded-none border-none' : 'rounded-3xl'
      } ${isDragOver ? 'ring-4 ring-cyan-500/80 border-cyan-400' : 'border-[var(--border-color)]'} shadow-sm`}
      style={{
        backgroundColor: (isWaitingForVideo || (!isCameraConnected && isCameraSource)) ? 'var(--bg-card)' : '#000000',
        ...(isFullscreen ? { width: '100%', height: '100%', minHeight: '100vh', maxHeight: '100vh' } : {})
      }}
    >
      <input type="file" ref={fileInputRef} onChange={handleFileInputChange} accept="video/*" className="hidden" />

      {/* Video Upload Card (simple non-interactive display card) */}
      {isWaitingForVideo && (
        <VideoUploadCard
          uploadProgress={uploadProgress}
          isSelectingVideo={isSelectingVideo}
          isBackendConnected={isBackendConnected}
        />
      )}

      {isCameraSource && !hasCameraRecording && !isCameraConnected && (
        <CameraOfflineCard
          onSwitchToVideo={() => onRequestSwitchMode?.('uploaded-video')}
          onRetryConnection={onRetryConnection || (async () => {
            try {
              await cameraService.start();
              await cameraService.startStream();
            } catch (e) {
              console.error('Retry connection failed:', e);
              window.location.reload();
            }
          })}
          onCanvasClick={handleCanvasClick}
          errorMessage={cameraError}
        />
      )}

      {isCameraSource && !hasCameraRecording && isCameraConnected && !isStreaming && (
        <CameraStandbyCard
          onStartStream={onStartStream}
          onSwitchToVideo={() => onRequestSwitchMode?.('uploaded-video')}
          onCanvasClick={handleCanvasClick}
        />
      )}

      {/* 2 & 3. VIDEO & CAMERA VIEWPORT WITH TRUE ASPECT RATIO & ZOOM/PAN */}
      {isMediaActive && (
        <div
          ref={viewportRef}
          onMouseDown={handleMouseDown}
          onDoubleClick={handleDoubleClick}
          className="absolute inset-0 flex items-center justify-center pointer-events-auto bg-black overflow-hidden select-none"
          style={{ cursor: cursorStyle }}
        >
          <div
            className="relative shrink-0"
            style={{
              ...fittedRect,
              ...transformStyle,
            }}
            onClick={handleCanvasClick}
          >
            {/* BACKDROP: Cached or stopped frame rendered as a reliable persistent background */}
            {effectiveBackdrop && (
              <img
                src={effectiveBackdrop}
                className="absolute inset-0 z-0 h-full w-full pointer-events-none rounded bg-black object-contain"
                alt=""
                onLoad={(e) => {
                  const tgt = e.target as HTMLImageElement;
                  if (tgt.naturalWidth && tgt.naturalHeight) {
                    const aspect = tgt.naturalWidth / tgt.naturalHeight;
                    setVideoAspect((prev) => (prev !== aspect ? aspect : prev));
                  }
                  setIsFirstFrameLoaded(true);
                }}
              />
            )}

            {/* STREAM VIEWPORT: If backend session is active (video or camera), render via <img> to support MJPEG streaming */}
            {(videoSessionId || cameraRecordSessionId || isCameraSource) ? (
              <img
                ref={cameraImgRef}
                crossOrigin="anonymous"
                src={
                  hasCameraRecording
                    ? (isRunning
                      ? `${getApiBaseUrl()}/video/stream/${cameraRecordSessionId}?t=${streamCacheBuster}`
                      : `${getApiBaseUrl()}/video/last_frame/${cameraRecordSessionId}?t=${streamCacheBuster}`)
                    : isCameraSource
                      ? `${getApiBaseUrl()}/oak/inference/stream/live?t=${streamCacheBuster}`
                      : effectiveVideoUrl
                }
                className={`absolute inset-0 z-0 h-full w-full pointer-events-none rounded bg-black object-contain ${streamError && effectiveBackdrop ? 'opacity-0' : 'opacity-100'
                  }`}
                style={{
                  filter: isCameraSource && !hasCameraRecording && cameraConfig
                    ? `brightness(${Math.max(0.2, 1 + ((cameraConfig.brightness ?? 0) / 100))}) contrast(${Math.max(0.2, (cameraConfig.contrast ?? 50) / 50)})`
                    : undefined
                }}
                alt=""
                onLoad={(e) => {
                  const tgt = e.target as HTMLImageElement;
                  if (tgt.naturalWidth && tgt.naturalHeight) {
                    const aspect = tgt.naturalWidth / tgt.naturalHeight;
                    setVideoAspect((prev) => (prev !== aspect ? aspect : prev));
                  }
                  setStreamError(false);
                  setIsFirstFrameLoaded((prev) => (!prev ? true : prev));
                  if (!isRunning && effectiveFramesProcessed > 0) {
                    captureFrame();
                  }
                }}
                onError={() => {
                  console.warn('[DetectionCanvas] Stream frame interrupted or reconnecting...');
                  setStreamError(true);
                  if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
                  retryTimeoutRef.current = setTimeout(() => {
                    setStreamCacheBuster(Date.now());
                  }, 1200);
                }}
              />
            ) : hasActiveVideo ? (
              /* Local MP4 video preview before backend session starts */
              <video
                ref={videoRef}
                src={customVideoUrl}
                className="absolute inset-0 z-0 h-full w-full pointer-events-none rounded bg-black object-contain"
                loop
                muted
                playsInline
                onLoadedMetadata={(e) => {
                  const tgt = e.target as HTMLVideoElement;
                  if (tgt.videoWidth && tgt.videoHeight) {
                    setVideoAspect(tgt.videoWidth / tgt.videoHeight);
                  }
                  setIsFirstFrameLoaded(true);
                }}
              />
            ) : null}

            <canvas ref={canvasRef} className="absolute inset-0 z-10 h-full w-full pointer-events-none rounded" />

            {/* AI Bounding Boxes: Shown in INFERENCE mode or always for video upload / camera recording */}
            {!isOverlayShowing && (isRunning || hasInferenceResult) && (feedMode === 'inference' || !isCameraSource || hasCameraRecording) && tubes.length > 0 && (
              <BoundingBoxOverlay
                tubes={tubes}
                selectedTubeId={selectedTubeId}
                onSelectTube={onSelectTube}
                labelMode={labelMode}
              />
            )}
          </div>
        </div>
      )}

      <LoadingOverlay
        isStarting={isStarting}
        isCameraSource={isCameraSource}
        isVideoSource={isVideoSource}
        hasCameraRecording={hasCameraRecording}
        cameraStartingState={cameraStartingState}
        hasActiveVideo={hasActiveVideo}
        isRunning={isRunning}
        isFirstFrameLoaded={isFirstFrameLoaded}
        isCameraConnected={isCameraConnected}
      />

      {!isOverlayShowing && (
        <TopToolbar
          isRunning={isRunning}
          hasActiveVideo={hasActiveVideo}
          feedMode={feedMode}
          onFeedModeChange={onFeedModeChange}
          isRecording={isRecording}
          isSaving={isSaving}
          recordingDuration={recordingDuration}
          onToggleRecording={async () => {
            if (hasCameraRecording) return; // block recording while reviewing a clip
            if (isRecording) {
              if (isRunning && isCameraSource) {
                // Ensure live inference claim is stopped before the recording is finalized
                await onStopInference?.();
              }
              const res = await stopRecording();
              if (res && res.session_id && res.stream_url && res.filename) {
                const fullStreamUrl = `${getApiBaseUrl()}${res.stream_url}`;
                onCustomVideoUploaded?.(fullStreamUrl, res.filename, res.session_id, true);
              }
            } else {
              // If stream is not running yet, start the stream first automatically
              if (!isStreaming && onStartStream) {
                await onStartStream();
                await new Promise((resolve) => setTimeout(resolve, 1000));
              }
              await startRecording(recordingFormat || 'MP4');
            }
          }}
          isFullscreen={isFullscreen}
          onToggleFullscreen={toggleFullscreen}
          showHUD={showHUD}
          onToggleHUD={() => setShowHUD(!showHUD)}
          backendStatus={backendStats?.status}
          isCameraSource={isCameraSource}
          isVideoSource={isVideoSource}
          hasCameraRecording={hasCameraRecording}
          anomalyStatus={anomalyStatus}
          isStreaming={isStreaming}
          isFirstFrameLoaded={isFirstFrameLoaded}
          framesProcessed={effectiveFramesProcessed}
          onClearCustomVideo={onClearCustomVideo}
          labelMode={labelMode}
          onLabelModeChange={setLabelMode}
          zoom={zoom}
          onZoomIn={zoomIn}
          onZoomOut={zoomOut}
          onResetZoom={resetZoom}
          onSetZoom={setZoomLevel}
        />
      )}

      {!isOverlayShowing && showHUD && !isCameraOffline && (isRunning || isStarting || hasInferenceResult) && (
        <StatusBar
          anomalyStatus={anomalyStatus}
          fps={fps}
          backendStatus={backendStats?.status}
          tubes={tubes}
        />
      )}

      {/* Floating Bottom-Right Corner Zoom Controls: Always prominent and accessible on canvas */}
      {!isOverlayShowing && showHUD && isMediaActive && (
        <div className="absolute bottom-3.5 right-3.5 sm:bottom-4 sm:right-4 z-30 pointer-events-auto">
          <ZoomControls
            zoom={zoom}
            onZoomIn={zoomIn}
            onZoomOut={zoomOut}
            onResetZoom={resetZoom}
            onSetZoom={setZoomLevel}
          />
        </div>
      )}
    </div>
  );
};
