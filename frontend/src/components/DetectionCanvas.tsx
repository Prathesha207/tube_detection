import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import type { TubeEntity, StreamSourceType, AnomalyStatus, LabelMode, CameraConfig } from '../types';
import { getApiBaseUrl } from '../lib/api';
import { showToast } from '../lib/toast';
import { useInferenceStore } from '../store/inferenceStore';
import { cameraService } from './service/cameraService';

import { Loader2 } from 'lucide-react';

// Extracted Canvas Components
import { BoundingBoxOverlay } from './canvas/BoundingBoxOverlay';
import { VideoUploadCard } from './canvas/VideoUploadCard';
import { CameraOfflineCard } from './canvas/CameraOfflineCard';
import { CameraStandbyCard } from './canvas/CameraStandbyCard';
import { TopToolbar } from './canvas/TopToolbar';
// import { StatusBar } from './canvas/StatusBar';
import { LoadingOverlay } from './canvas/LoadingOverlay';
import { ZoomControls } from './canvas/ZoomControls';

// Extracted Canvas Hooks
import { useFullscreen } from '../hooks/useFullscreen';
import { useContainerFit } from '../hooks/useContainerFit';
import { useVideoUpload } from '../hooks/useVideoUpload';
import { useCanvasZoom } from '../hooks/useCanvasZoom';

// ROI Components & Utilities
import { RoiDrawingOverlay } from './canvas/RoiDrawingOverlay';
import { RoiLeftToolbar, RoiToolMode } from './canvas/RoiLeftToolbar';
import { roiService } from './service/roiService';
import { Point, validateRoi } from '../utils/roiUtils';

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
  recordedFile?: File | null;
  clearRecording?: () => void;
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
  recordedFile,
  clearRecording,
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
  const [streamError, setStreamError] = useState<boolean>(false);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    };
  }, []);

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

  // ROI Mask Management State
  const [isRoiActive, setIsRoiActive] = useState<boolean>(false);
  const [activeRoiTool, setActiveRoiTool] = useState<RoiToolMode>('pointer');
  const [roiPoints, setRoiPoints] = useState<Point[]>([]);
  const [savedRoiPoints, setSavedRoiPoints] = useState<Point[]>([]);
  const [isSavingRoi, setIsSavingRoi] = useState<boolean>(false);
  const [roiFrameSize, setRoiFrameSize] = useState<{ width: number; height: number }>({
    width: 1920,
    height: 1080,
  });

  // Effective native frame width and height (from authoritative ML stats or video dimensions)
  const effectiveFw = Number(backendStats?.video_width) || Number(videoDimensions?.width) || 1920;
  const effectiveFh = Number(backendStats?.video_height) || Number(videoDimensions?.height) || 1080;

  // Active session id (video or camera)
  const activeSessionId = videoSessionId || cameraRecordSessionId || (isCameraSource ? 'camera' : undefined);

  // Pure per-session ROI: Every start of inference or new session starts with a full frame (no ROI).
  // Never restore old ROIs from disk.
  useEffect(() => {
    setRoiPoints([]);
    setSavedRoiPoints([]);
    setIsRoiActive(false);
  }, [activeSessionId, isRunning]);

  const hasRoiChanges = useMemo(() => {
    if (roiPoints.length !== savedRoiPoints.length) return true;
    return JSON.stringify(roiPoints) !== JSON.stringify(savedRoiPoints);
  }, [roiPoints, savedRoiPoints]);

  const handleSaveRoi = async () => {
    const val = validateRoi(roiPoints);
    if (!val.valid) {
      showToast('error', val.error || 'Invalid ROI geometry');
      return;
    }
    setIsSavingRoi(true);
    try {
      const res = await roiService.saveRoi(
        roiPoints,
        roiFrameSize.width || effectiveFw,
        roiFrameSize.height || effectiveFh,
        activeSessionId,
        isCameraSource ? 'camera' : 'video'
      );
      setSavedRoiPoints(res.points || roiPoints);
      showToast('success', 'ROI saved & applied. Tracker totals recalibrated for the new region.');
      // Automatically close editor mode after saving: hides toolbar and handles, shows clean line only
      setIsRoiActive(false);
      // Reset counters in store to prevent jump in totals from looking like a bug
      useInferenceStore.getState().setStats({
        ...backendStats,
        heads_total: 0,
        tails_total: 0,
      });
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to save ROI';
      showToast('error', msg);
    } finally {
      setIsSavingRoi(false);
    }
  };

  const handleClearRoi = async () => {
    try {
      await roiService.saveRoi(
        [],
        roiFrameSize.width || effectiveFw,
        roiFrameSize.height || effectiveFh,
        activeSessionId,
        isCameraSource ? 'camera' : 'video'
      );
      setRoiPoints([]);
      setSavedRoiPoints([]);
      showToast('info', 'ROI deleted. Full frame is now being inspected.');
      useInferenceStore.getState().setStats({
        ...backendStats,
        heads_total: 0,
        tails_total: 0,
      });
      // Close the ROI editor
      setIsRoiActive(false);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to delete ROI on server';
      showToast('error', msg);
    }
  };

  // Reset zoom whenever stream or video source changes
  useEffect(() => {
    resetZoom();
  }, [sourceType, customVideoUrl, cameraRecordSessionId, resetZoom]);

  // Ensure camera stream reconnects cleanly whenever running state changes while streaming or camera reconnects
  useEffect(() => {
    if (isCameraSource && isStreaming) {
      setStreamCacheBuster(Date.now());
    }
  }, [isRunning, isCameraSource, isStreaming, isCameraConnected]);

  // Auto-retry reconnect loop if stream encountered an error or connection was interrupted
  useEffect(() => {
    if (streamError && ((isCameraSource && isStreaming) || (isRunning && (videoSessionId || cameraRecordSessionId)))) {
      const timer = setTimeout(() => {
        setStreamCacheBuster(Date.now());
      }, 2500);
      return () => clearTimeout(timer);
    }
  }, [streamError, isCameraSource, isStreaming, isRunning, videoSessionId, cameraRecordSessionId, streamCacheBuster]);

  const effectiveFramesProcessed = framesProcessed || backendStats?.frames_processed || 0;
  const hasInferenceResult = (effectiveFramesProcessed > 0 || tubes.length > 0) && tubes.length > 0;

  const hasDetections = Boolean(
    tubes.length > 0 ||
    (backendStats?.detections && backendStats.detections.length > 0) ||
    ((backendStats?.heads_count || 0) > 0) ||
    ((backendStats?.tails_count || 0) > 0) ||
    ((backendStats?.heads_total || 0) > 0) ||
    ((backendStats?.tails_total || 0) > 0)
  );

  // Reset overlay state whenever inference stops so new runs start with the overlay
  useEffect(() => {
    if (!isRunning) {
      setIsFirstFrameLoaded(false);
    }
  }, [isRunning, videoSessionId, cameraRecordSessionId]);

  // Keep "Inference is Running..." visible until initial detections arrive, then reveal the frame!
  useEffect(() => {
    if (isRunning && !isFirstFrameLoaded) {
      if (hasDetections) {
        setIsFirstFrameLoaded(true);
      } else if (effectiveFramesProcessed >= 15) {
        // Fallback: If 15 frames processed without detections (e.g. empty background), reveal frame
        setIsFirstFrameLoaded(true);
      }
    }
  }, [isRunning, isFirstFrameLoaded, hasDetections, effectiveFramesProcessed]);

  // Safety fallback: if no tubes exist in the scene, reveal video after 3.5s
  useEffect(() => {
    if (isRunning && !isFirstFrameLoaded) {
      const timer = setTimeout(() => {
        setIsFirstFrameLoaded(true);
      }, 3500);
      return () => clearTimeout(timer);
    }
  }, [isRunning, isFirstFrameLoaded]);

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
      className={`relative w-full flex-1 h-full min-h-[350px] lg:min-h-0 overflow-hidden border select-none group transition-colors duration-200 ${isFullscreen ? 'rounded-none border-none' : 'rounded-3xl'
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
                  if (hasDetections || effectiveFramesProcessed >= 15 || !isRunning) {
                    setIsFirstFrameLoaded(true);
                  }
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
              <div className={`absolute inset-0 ${isRoiActive ? 'pointer-events-none opacity-40' : 'pointer-events-auto'}`}>
                <BoundingBoxOverlay
                  tubes={tubes}
                  selectedTubeId={selectedTubeId}
                  onSelectTube={onSelectTube}
                  labelMode={labelMode}
                />
              </div>
            )}

            {/* Passive Saved ROI Reference Outline — only shown after the user draws & saves an ROI */}
            {!isRoiActive && savedRoiPoints.length >= 3 && (
              <svg
                className="absolute inset-0 z-20 w-full h-full pointer-events-none"
                style={{ zIndex: 20 }}
                viewBox={`0 0 ${roiFrameSize.width || effectiveFw} ${roiFrameSize.height || effectiveFh}`}
                preserveAspectRatio="none"
              >
                <polygon
                  points={savedRoiPoints.map(([x, y]) => `${x},${y}`).join(' ')}
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2"
                  strokeDasharray="6 4"
                />
              </svg>
            )}

            {/* Interactive ROI Mask Drawing & Editing Overlay (Active when user toggles [ 🎯 ]) */}
            {isRoiActive && (
              <RoiDrawingOverlay
                points={roiPoints}
                onChange={setRoiPoints}
                activeTool={activeRoiTool}
                onSelectTool={setActiveRoiTool}
                frameWidth={roiFrameSize.width || effectiveFw}
                frameHeight={roiFrameSize.height || effectiveFh}
              />
            )}
          </div>
        </div>
      )}

      {/* Floating Left-Center Vertical Tool Palette for ROI Drawing */}
      {isRoiActive && isMediaActive && !isOverlayShowing && (
        <RoiLeftToolbar
          activeTool={activeRoiTool}
          onSelectTool={setActiveRoiTool}
          onClear={handleClearRoi}
          onSave={handleSaveRoi}
          isSaving={isSavingRoi}
          hasChanges={hasRoiChanges}
          hasRoi={roiPoints.length > 0}
        />
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
          isRoiActive={isRoiActive}
          onToggleRoi={() => {
            setIsRoiActive((prev) => {
              if (!prev) {
                // Opening editor: pre-populate with any previously saved ROI so user can edit it
                setRoiPoints(savedRoiPoints.length > 0 ? savedRoiPoints : []);
              } else {
                // Closing editor without saving: discard unsaved edits
                setRoiPoints([]);
              }
              return !prev;
            });
          }}
        />
      )}

      {/* {!isOverlayShowing && showHUD && !isCameraOffline && (isRunning || isStarting || hasInferenceResult) && (
        <StatusBar
          anomalyStatus={anomalyStatus}
          fps={fps}
          backendStatus={backendStats?.status}
          tubes={tubes}
        />
      )} */}

      {/* Floating Bottom-Right Corner Zoom Controls: Always prominent and accessible on canvas */}
      {!isOverlayShowing && showHUD && isMediaActive && !isRoiActive && (
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
