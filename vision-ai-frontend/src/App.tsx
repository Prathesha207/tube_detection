import React, { useState, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import type { ThemeMode, StreamSourceType } from './types';
import { LandingScreen } from './components/LandingScreen';
import { lockScroll, unlockScroll } from './utils/scrollLock';
import { Header } from './components/Header';
import { SourceSelector } from './components/SourceSelector';
import { DetectionCanvas } from './components/DetectionCanvas';
import { DetectionDrawer } from './components/DetectionDrawer';
import { CameraSettingsModal } from './components/CameraSettingsModal';
import { HelpModal } from './components/HelpModal';
import { Modal, Toast, Button } from './components/ui';
import { useInferenceStore } from './store/inferenceStore';
import { playWaterDropSound, setSoundEnabled, isSoundEnabled } from './utils/audio';
import { AlertTriangle } from 'lucide-react';
import { getApiBaseUrl } from './lib/api';
import { cameraService } from './components/service/cameraService';
import { resetBBoxCache, mapDetectionsToDucks } from './utils/mlDataMapper';
import { DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT } from './utils/constants';
import { loadSessionState, saveSessionState, clearSessionState } from './utils/sessionPersistence';

// Extracted hooks — each one owns a clean slice of state + effects
import { useBackendHealth } from './hooks/useBackendHealth';
import { useToastAndLogs } from './hooks/useToastAndLogs';
import { useCameraStatus } from './hooks/useCameraStatus';
import { useVideoPipeline } from './hooks/useVideoPipeline';
import { useInferenceLoop } from './hooks/useInferenceLoop';
import { useAnomalyStatus } from './hooks/useAnomalyStatus';

export default function App() {
  // ─── 1. System Health ──────────────────────────────────────────────
  const { systemInitialized, setSystemInitialized, isBackendConnected, handleInitializeSystem } = useBackendHealth();

  // ─── 2. Toast Notifications & Activity Logs ────────────────────────
  const { toast, setToast, showToast, logs, addLog } = useToastAndLogs();

  // ─── 3. Theme & UI Shell ───────────────────────────────────────────
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('visionmonitor-theme') as ThemeMode | null;
      if (saved && (saved === 'pond-light' || saved === 'pond-dark' || saved === 'nature')) {
        return saved;
      }
    }
    return 'pond-light';
  });

  const handleThemeChange = (newTheme: ThemeMode) => {
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-theme', newTheme);
      document.body.setAttribute('data-theme', newTheme);
      const themeClass = newTheme === 'pond-dark' ? 'theme-pond-dark dark' : newTheme === 'nature' ? 'theme-nature dark' : 'theme-pond-light';
      document.documentElement.className = themeClass;
      document.body.className = themeClass;
      try { localStorage.setItem('visionmonitor-theme', newTheme); } catch { }
    }
    setTheme(newTheme);
  };

  const [feedMode, setFeedMode] = useState<'raw' | 'inference'>('inference');
  const [soundActive, setSoundActive] = useState<boolean>(() => isSoundEnabled());
  const [drawerOpen, setDrawerOpen] = useState<boolean>(true);
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [helpOpen, setHelpOpen] = useState<boolean>(false);
  const [selectedDuckId, setSelectedDuckId] = useState<string | null>(null);

  const handleToggleSound = () => {
    const next = !soundActive;
    setSoundActive(next);
    setSoundEnabled(next);
    if (next) playWaterDropSound();
  };

  // ─── 4. Source Mode Coordination ───────────────────────────────────
  // Seed sourceType from session so refresh preserves the active source
  const [sourceType, setSourceType] = useState<StreamSourceType>(() => {
    const s = loadSessionState();
    return (s?.sourceType as StreamSourceType) ?? 'uploaded-video';
  });
  const [pendingSourceSwitch, setPendingSourceSwitch] = useState<StreamSourceType | null>(null);

  // ─── 5. Camera Hardware ────────────────────────────────────────────
  const camera = useCameraStatus(addLog, showToast);

  // ─── Shared Pipeline State ─────────────────────────────────────────
  const initialSession = useMemo(() => loadSessionState(), []);
  // Seed isRunning and all metrics from session so Ctrl+R keeps stats and ducks visible
  const [isRunning, setIsRunning] = useState<boolean>(() => {
    return initialSession?.isRunning ?? false;
  });
  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(() => {
    if (initialSession?.sourceType === 'oak-camera' || initialSession?.sourceType === 'webcam') return 0;
    return initialSession?.fps ?? 0;
  });
  const [framesProcessed, setFramesProcessed] = useState<number>(() => {
    if (initialSession?.sourceType === 'oak-camera' || initialSession?.sourceType === 'webcam') return 0;
    return initialSession?.framesProcessed ?? 0;
  });
  const [uptimeSeconds, setUptimeSeconds] = useState<number>(() => {
    if (initialSession?.sourceType === 'oak-camera' || initialSession?.sourceType === 'webcam') return 0;
    return initialSession?.uptimeSeconds ?? 0;
  });
  const [expectedDucks, setExpectedDucks] = useState<number>(() => initialSession?.expectedDucks ?? 18);
  const [ducks, setDucks] = useState<import('./types').DuckEntity[]>(() => {
    if (initialSession?.sourceType === 'oak-camera' || initialSession?.sourceType === 'webcam') return [];
    return initialSession?.ducks ?? [];
  });
  const [lastCameraFrame, setLastCameraFrame] = useState<string | undefined>(() => initialSession?.lastCameraFrame);
  const [lastVideoFrame, setLastVideoFrame] = useState<string | undefined>(() => initialSession?.lastVideoFrame);

  // Restore inference store stats from session on mount
  useEffect(() => {
    if (initialSession?.stats && initialSession.stats.status !== 'idle') {
      if (initialSession.sourceType !== 'oak-camera' && initialSession.sourceType !== 'webcam') {
        useInferenceStore.getState().replaceStats(initialSession.stats);
      }
    }
  }, [initialSession]);

  // Maintain fresh refs for session state snapshots
  const ducksRef = useRef(ducks);
  ducksRef.current = ducks;
  const framesProcessedRef = useRef(framesProcessed);
  framesProcessedRef.current = framesProcessed;
  const fpsRef = useRef(fps);
  fpsRef.current = fps;
  const uptimeSecondsRef = useRef(uptimeSeconds);
  uptimeSecondsRef.current = uptimeSeconds;
  const expectedDucksRef = useRef(expectedDucks);
  expectedDucksRef.current = expectedDucks;
  const sourceTypeRef = useRef(sourceType);
  sourceTypeRef.current = sourceType;

  // Keep sessionStorage in sync with live inference state & results
  useEffect(() => {
    const isCamera = sourceType === 'oak-camera' || sourceType === 'webcam';
    saveSessionState({
      isRunning,
      sourceType,
      expectedDucks,
      ...(ducks.length > 0 && !isCamera ? { ducks } : {}),
      ...(framesProcessed > 0 && !isCamera ? { framesProcessed, fps, uptimeSeconds } : {}),
      stats: isCamera ? undefined : useInferenceStore.getState().stats,
      lastCameraFrame: isCamera ? lastCameraFrame : undefined,
      lastVideoFrame: !isCamera ? lastVideoFrame : undefined,
    });
  }, [isRunning, sourceType, expectedDucks, ducks, framesProcessed, fps, uptimeSeconds, lastCameraFrame, lastVideoFrame]);

  // Snapshot cache to preserve complete run state across source toggling
  interface SourceStateSnapshot {
    ducks: import('./types').DuckEntity[];
    stats: import('./store/inferenceStore').InferenceStats;
    framesProcessed: number;
    fps: number;
    uptimeSeconds: number;
    selectedDuckId: string | null;
    videoDimensions: { width: number; height: number } | null;
    lastCameraFrame?: string;
    lastVideoFrame?: string;
  }

  const sourceStateCache = React.useRef<{
    video: SourceStateSnapshot | null;
    camera: SourceStateSnapshot | null;
  }>({
    video: null,
    camera: null,
  });

  // ─── 6. Derived Source States ──────────────────────────────────────
  const backendStats = useInferenceStore(state => state.stats);
  const isVideoSource = sourceType === 'sample-pond' || sourceType === 'uploaded-video';
  const isCameraSource = sourceType === 'oak-camera' || sourceType === 'webcam';

  // ─── 7. Video Pipeline ─────────────────────────────────────────────
  const video = useVideoPipeline({
    showToast,
    addLog,
    expectedDucks,
    sourceType,
    isRunning,
    setIsRunning,
    setDucks,
    setFramesProcessed,
    setFps,
    setCameraStartingState: camera.setCameraStartingState,
    setSourceType,
    setIsStarting,
  });

  const hasActiveVideo = isVideoSource && Boolean(video.customVideoUrl || video.customVideoName);

  // ─── 8. Camera Pipeline (startCameraPipeline / stopCameraPipeline) ─
  const startCameraPipeline = async (): Promise<boolean> => {
    setDucks([]);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();
    camera.setCameraStartingState('waking_camera');
    addLog('Step 1/3: Starting OAK Camera device (POST /oak/start)...', 'info');

    try {
      await fetch(`${getApiBaseUrl()}/oak/inference/update_expected/live`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: expectedDucks })
      }).catch(err => console.error("Failed to sync expected ducks before start", err));

      const startRes = await cameraService.start();
      if (startRes?.status === 'error') throw new Error(startRes.message || 'Camera start failed');

      const streamRes = await cameraService.startStream();
      if (streamRes?.status === 'error') throw new Error(streamRes.message || 'Stream start failed');
      camera.setIsStreaming(true);

      const inferenceRes = await cameraService.startLiveInference('live');
      if (inferenceRes?.status === 'error') throw new Error(inferenceRes.message || 'Inference start failed');

      camera.setCameraStartingState('waiting_frame');
      addLog('Step 2/3: Stream started • Waiting for camera sensor to warm up...', 'info');

      await new Promise((resolve) => setTimeout(resolve, 1500));

      camera.setCameraStartingState('ready');
      setIsRunning(true);
      showToast('success', 'Camera inference started successfully');
      addLog('Step 3/3: First live frame received (1080p). Starting inference • YOLOv8 active.', 'success');
      return true;
    } catch (error) {
      console.error('Failed to start camera pipeline:', error);
      camera.setCameraStartingState('ready');
      setIsRunning(false);
      showToast('error', 'Failed to start camera device');
      addLog('Error: Failed to connect to OAK stream or live inference.', 'error');
      return false;
    }
  };

  const stopCameraPipeline = async () => {
    camera.setCameraStartingState('ready');
    setIsRunning(false);
    setDucks([]);
    try {
      await cameraService.stopStream();
      camera.setIsStreaming(false);
    } catch (err) {
      console.warn('Stream stop call failed:', err);
    }
  };

  // ─── 9. Inference Loop (owns transport and sets states) ────────────
  const inference = useInferenceLoop({
    sourceType,
    videoSessionId: video.videoSessionId,
    cameraRecordSessionId: video.cameraRecordSessionId,
    expectedDucks,
    showToast,
    addLog,
    startCameraPipeline,
    setDucks,
    setVideoDimensions: video.setVideoDimensions,
    cameraService,
    setCameraIsStreaming: camera.setIsStreaming,
    isRunning,
    setIsRunning,
    isStarting,
    setIsStarting,
    fps,
    setFps,
    framesProcessed,
    setFramesProcessed,
    uptimeSeconds,
    setUptimeSeconds,
    setLastCameraFrame,
  });

  // Recalculate derived values that depend on inference state.
  // NOTE: isRunning=true means inference is active regardless of stream state flags —
  // this handles the post-refresh window where isRunning is seeded true but camera/video
  // flags haven't fully settled yet.
  const hasActiveStream = isRunning || (isVideoSource && hasActiveVideo) || (isCameraSource && camera.isCameraDeviceActive && camera.cameraStartingState === 'ready');
  const isStandby = !hasActiveStream && !isRunning && ducks.length === 0;

  // ─── Auto-resume after page refresh ───────────────────────────────
  // On mount: isRunning, videoSessionId, and sourceType are already seeded
  // from sessionStorage. useInferenceLoop will automatically start polling
  // because its effect depends on [isRunning, videoSessionId, sourceType] —
  // all of which are already set correctly on mount.
  //
  // This effect only handles side-effects the polling loop can't do itself:
  //   • Log the reconnect message
  //   • Restore camera.isStreaming for camera sources
  //   • Stop isRunning ONLY if the video session is definitively gone (404)
  const didAutoResume = useRef(false);
  useEffect(() => {
    if (didAutoResume.current) return;
    didAutoResume.current = true;
    const saved = loadSessionState();
    if (!saved) return;

    const st = (saved.sourceType || sourceType) as StreamSourceType;
    const isVideo = st === 'uploaded-video' || st === 'sample-pond';
    const isCamera = st === 'oak-camera' || st === 'webcam';

    if (isVideo && saved.videoSessionId) {
      const sessionId = saved.videoSessionId;
      fetch(`${getApiBaseUrl()}/video/status/${sessionId}`, { method: 'GET', cache: 'no-store' })
        .then(async (res) => {
          if (res.status === 404) {
            if (saved.isRunning) {
              saveSessionState({ isRunning: false });
              setIsRunning(false);
              addLog('Video session no longer exists on backend — inference stopped.', 'info');
            }
            return;
          }
          if (!res.ok) return;
          const data = await res.json();
          if (!data) return;

          // Sync backend authoritative stats and detections
          useInferenceStore.getState().setStats(data);
          if (data.fps) setFps(data.fps);
          if (data.frames_processed) {
            setFramesProcessed(data.frames_processed);
            setUptimeSeconds(Math.floor(data.frames_processed / (data.fps || 30)));
          }
          if (data.video_width && data.video_height) {
            video.setVideoDimensions({ width: data.video_width, height: data.video_height });
          }

          const vw = data.video_width || DEFAULT_VIDEO_WIDTH;
          const vh = data.video_height || DEFAULT_VIDEO_HEIGHT;
          const incomingDucks = mapDetectionsToDucks(data, vw, vh, expectedDucks);
          if (data.status !== 'HAND' && !data.hand_detected && incomingDucks.length > 0) {
            setDucks(incomingDucks);
          }

          if (saved.isRunning) {
            addLog('🔄 Page refreshed — reconnecting to active video inference session...', 'info');
          } else if (incomingDucks.length > 0 || (data.frames_processed || 0) > 0) {
            addLog('🔄 Page refreshed — restored inference stats and detections.', 'info');
          }
        })
        .catch(() => {
          // Network error — leave state as restored from sessionStorage
        });
    } else if (isCamera && saved.isRunning) {
      addLog('🔄 Page refreshed — reconnecting to live camera inference stream...', 'info');
      camera.setIsStreaming(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── 10. Anomaly Detection (computed exactly once per render) ──────
  const anomalyFinal = useAnomalyStatus({
    hasActiveStream,
    isRunning,
    isStarting,
    isCameraSource,
    framesProcessed,
    backendStats,
    addLog,
    ducks,
    expectedDucks,
  });

  // ─── 10. Mode Switching ────────────────────────────────────────────
  const executeSwitchMode = async (targetType: StreamSourceType) => {
    const isCurrentCamera = sourceType === 'oak-camera' || sourceType === 'webcam';
    const isTargetCamera = targetType === 'oak-camera' || targetType === 'webcam';
    const currentKey = isCurrentCamera ? 'camera' : 'video';
    const targetKey = isTargetCamera ? 'camera' : 'video';

    // 1. Snapshot current source state before switching away
    sourceStateCache.current[currentKey] = {
      ducks,
      stats: useInferenceStore.getState().stats,
      framesProcessed,
      fps,
      uptimeSeconds,
      selectedDuckId,
      videoDimensions: video.videoDimensions,
      lastCameraFrame: isCurrentCamera ? lastCameraFrame : undefined,
      lastVideoFrame: !isCurrentCamera ? lastVideoFrame : undefined,
    };

    // 2. Stop running stream/inference on previous source
    if (isCurrentCamera) {
      if (isRunning) { setIsRunning(false); cameraService.stopLiveInference().catch(() => { }); }
      if (camera.isStreaming) { cameraService.stopStream().catch(() => { }); camera.setIsStreaming(false); }
    } else if (isRunning && video.videoSessionId) {
      setIsRunning(false);
      fetch(`${getApiBaseUrl()}/video/stop/${video.videoSessionId}`, { method: 'POST' }).catch(() => { });
    }

    // 3. Switch source
    setSourceType(targetType);
    setPendingSourceSwitch(null);
    setSelectedDuckId(null);

    // 4. Restore target source state if previously cached
    const cached = sourceStateCache.current[targetKey];
    if (cached && (cached.ducks.length > 0 || cached.framesProcessed > 0)) {
      setDucks(cached.ducks);
      useInferenceStore.getState().replaceStats(cached.stats);
      setFramesProcessed(cached.framesProcessed);
      setFps(cached.fps);
      setUptimeSeconds(cached.uptimeSeconds);
      setSelectedDuckId(cached.selectedDuckId);
      if (cached.videoDimensions) video.setVideoDimensions(cached.videoDimensions);
      if (cached.lastCameraFrame && isTargetCamera) setLastCameraFrame(cached.lastCameraFrame);
      if (cached.lastVideoFrame && !isTargetCamera) setLastVideoFrame(cached.lastVideoFrame);
    } else {
      // Clean slate for new un-run source
      setDucks([]);
      useInferenceStore.getState().resetStats();
      resetBBoxCache();
      setFramesProcessed(0);
      setFps(0);
      setUptimeSeconds(0);
      setSelectedDuckId(null);
    }

    if (isTargetCamera) {
      camera.setCameraStartingState('ready');
      camera.setIsStreaming(false);
      addLog(`Stream source switched to: ${targetType.toUpperCase()}`, 'info');
      showToast('info', 'Switched to OAK Camera mode');
      // No auto-start — CameraStandbyCard renders until user clicks Start Stream.
    } else {
      camera.setCameraStartingState('ready');
      if (video.customVideoUrl) {
        showToast('info', 'Switched to Video mode • Press Start Inference to evaluate');
      } else {
        showToast('info', 'Switched to Video mode');
      }
      addLog(`Stream source switched to: ${targetType.toUpperCase()}`, 'info');
    }
  };

  const handleRequestSwitchMode = (targetType: StreamSourceType) => {
    if (targetType === sourceType) return;
    const isCurrentCamera = sourceType === 'oak-camera' || sourceType === 'webcam';
    const isTargetCamera = targetType === 'oak-camera' || targetType === 'webcam';
    if (inference.isRunning && isCurrentCamera !== isTargetCamera) {
      setPendingSourceSwitch(targetType);
      return;
    }
    executeSwitchMode(targetType);
  };

  const handleConfirmSwitchMode = () => {
    if (!pendingSourceSwitch) return;
    playWaterDropSound();
    executeSwitchMode(pendingSourceSwitch);
  };

  // ─── 11. Metrics ──────────────────────────────────────────────────
  const metrics = useMemo(() => {
    const totalConf = anomalyFinal.activeDucks.reduce((acc, d) => acc + d.confidence, 0);
    const avgConfidence = anomalyFinal.activeDucks.length > 0 ? totalConf / anomalyFinal.activeDucks.length : 0;
    const speciesCounts: Record<string, number> = {};
    anomalyFinal.activeDucks.forEach((d) => { speciesCounts[d.species] = (speciesCounts[d.species] || 0) + 1; });
    return {
      fps: inference.fps,
      inferenceTimeMs: inference.fps > 0 ? 1000 / inference.fps : 0,
      framesProcessed: inference.framesProcessed,
      uptimeSeconds: inference.uptimeSeconds,
      avgConfidence,
      speciesCounts,
    };
  }, [anomalyFinal.activeDucks, inference.fps, inference.framesProcessed, inference.uptimeSeconds]);

  const isRecording = useInferenceStore((state) => state.isRecording);

  // ─── 12. Misc Handlers ────────────────────────────────────────────
  const handleRestart = () => {
    playWaterDropSound();
    setSelectedDuckId(null);
    addLog('Pipeline reset triggered. Reconnecting to camera stream...', 'info');
    inference.setFramesProcessed(0);
    inference.setUptimeSeconds(0);
    setTimeout(() => addLog('Camera re-connected • YOLOv8 model inference active', 'success'), 400);
  };

  const handleTakeSnapshot = () => {
    playWaterDropSound();
    addLog(`Snapshot captured at ${new Date().toLocaleTimeString()} (Frame #${inference.framesProcessed})`, 'success');
    const heroEl = document.getElementById('detection-hero-viewport');
    if (heroEl) {
      heroEl.classList.add('ring-4', 'ring-white');
      setTimeout(() => heroEl.classList.remove('ring-4', 'ring-white'), 300);
    }
  };

  const uploadTriggerRef = React.useRef<(() => void) | null>(null);

  const handleClearCustomVideo = () => {
    sourceStateCache.current.video = null;
    setLastVideoFrame(undefined);
    video.handleClearVideo();
    showToast('info', 'Video cleared. Select or upload a new video.');
  };

  const handleResetCamera = async () => {
    playWaterDropSound();
    setSelectedDuckId(null);
    setIsRunning(false);
    setDucks([]);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();
    inference.setFramesProcessed(0);
    inference.setFps(0);
    inference.setUptimeSeconds(0);
    sourceStateCache.current.camera = null;
    setLastCameraFrame(undefined);
    video.clearCameraRecording();
    camera.setCameraStartingState('ready');

    try {
      await cameraService.stopLiveInference();
    } catch { }

    showToast('info', 'Detection cards and details reset');
    addLog('Camera reset • Detection cards, counts, and metrics cleared.', 'info');
  };

  const handleStopStream = async () => {
    playWaterDropSound();
    setIsRunning(false);
    setSelectedDuckId(null);
    setDucks([]);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();
    inference.setFramesProcessed(0);
    inference.setFps(0);
    inference.setUptimeSeconds(0);
    setLastCameraFrame(undefined);
    sourceStateCache.current.camera = null;
    camera.setCameraStartingState('ready');
    camera.setIsStreaming(false);

    try {
      await cameraService.stopLiveInference();
    } catch { }
    try {
      await cameraService.stopStream();
    } catch { }

    showToast('info', 'Camera stream stopped • Cleared for fresh stream');
    addLog('Camera stream stopped • Canvas, overlays, and drawer cleared for fresh stream.', 'info');
  };

  const handleClearCameraRecord = () => {
    video.clearCameraRecording();
    camera.setCameraStartingState('ready');
    camera.setIsStreaming(false);
    cameraService.stopStream().catch(() => { });
  };

  const handleResetVideo = async () => {
    playWaterDropSound();
    setSelectedDuckId(null);
    setIsRunning(false);
    setDucks([]);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();
    inference.setFramesProcessed(0);
    inference.setFps(0);
    inference.setUptimeSeconds(0);
    sourceStateCache.current.video = null;
    setLastVideoFrame(undefined);
    // Stop and clear backend session if one is active
    if (video.videoSessionId) {
      try {
        await fetch(`${getApiBaseUrl()}/video/clear/${video.videoSessionId}`, { method: 'POST' });
      } catch { }
    }
    // Clear the video pipeline state so the upload card shows again
    video.setCustomVideoUrl(undefined);
    video.setLocalPreviewUrl(undefined);
    video.setVideoSessionId(null);
    video.setCustomVideoName(undefined);
    // Clear persisted session so a refresh after reset shows the upload card, not auto-resume
    clearSessionState();
    showToast('info', 'Video reset • Upload a new video to begin');
    addLog('Video cleared • Ready for a new upload.', 'info');
  };

  // Wrap toggle/stop/resume to pass startVideoInference
  const handleToggleRunning = async () => {
    if (sourceType === 'uploaded-video' || sourceType === 'sample-pond') {
      if (!video.videoSessionId) {
        uploadTriggerRef.current?.();
        return;
      }
    }
    if (sourceType === 'oak-camera' && video.cameraRecordSessionId) {
      if (!isRunning) {
        // GPU Contention Safety: release live camera claims before starting video inference on recording
        await cameraService.stopLiveInference().catch(() => { });
        await cameraService.stopStream().catch(() => { });
        camera.setIsStreaming(false);
      }
    }
    if ((sourceType === 'oak-camera' || sourceType === 'webcam') && !video.cameraRecordSessionId) {
      if (!isRunning && !camera.isStreaming) {
        // Stream not started yet: start both stream and inference together!
        playWaterDropSound();
        await startCameraPipeline();
        return;
      }
    }
    await inference.handleToggleRunning(video.startVideoInference);
  };
  const handleStopInference = async () => {
    camera.setCameraStartingState('ready');
    setSelectedDuckId(null);
    await inference.handleStopInference();
  };
  const handleResumeInference = async () => {
    if (sourceType === 'uploaded-video' || sourceType === 'sample-pond') {
      if (!video.videoSessionId) {
        uploadTriggerRef.current?.();
        return;
      }
      playWaterDropSound();
      void video.startVideoInference();
      return;
    }
    if ((sourceType === 'oak-camera' || sourceType === 'webcam') && !video.cameraRecordSessionId) {
      if (!camera.isStreaming) {
        // User clicked Start Inference directly without clicking Start Stream first:
        // Automatically start both stream and inference!
        playWaterDropSound();
        await startCameraPipeline();
        return;
      }
    }
    inference.handleResumeInference(video.startVideoInference);
  };

  // ─── 13. Keyboard Shortcuts (using refs to avoid stale closures) ──
  const handleToggleRunningRef = React.useRef(handleToggleRunning);
  const handleTakeSnapshotRef = React.useRef(handleTakeSnapshot);
  const isCameraSourceRef = React.useRef(isCameraSource);
  useEffect(() => { handleToggleRunningRef.current = handleToggleRunning; });
  useEffect(() => { handleTakeSnapshotRef.current = handleTakeSnapshot; });
  useEffect(() => { isCameraSourceRef.current = isCameraSource; });

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === 'Space') { e.preventDefault(); handleToggleRunningRef.current(); }
      else if (e.code === 'KeyI') {
        if (isCameraSourceRef.current) {
          setFeedMode(m => m === 'raw' ? 'inference' : 'raw');
        }
      }
      else if (e.code === 'KeyS') { handleTakeSnapshotRef.current(); }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // ─── Desktop-app close/reload guard ──────────────────────────────
  // When inference is running, intercept Ctrl+R and window close attempts.
  // This prevents accidentally leaving the inference page mid-session.
  const isRunningRef = React.useRef(isRunning);
  useEffect(() => { isRunningRef.current = isRunning; }, [isRunning]);

  // Notify Electron main process of inference state for native close dialog
  useEffect(() => {
    try {
      (window as any).electronAPI?.setInferenceRunning?.(isRunning);
    } catch { }
  }, [isRunning]);

  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      saveSessionState({
        isRunning: isRunningRef.current,
        sourceType: sourceTypeRef.current,
        expectedDucks: expectedDucksRef.current,
        ducks: ducksRef.current,
        framesProcessed: framesProcessedRef.current,
        fps: fpsRef.current,
        uptimeSeconds: uptimeSecondsRef.current,
        stats: useInferenceStore.getState().stats,
      });
      if (isRunningRef.current) {
        e.preventDefault();
        // Modern browsers require returnValue to be set for the dialog to show
        e.returnValue = 'Inference is still running. Are you sure you want to leave?';
        return e.returnValue;
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  // ─── 14. Theme Sync ───────────────────────────────────────────────
  useLayoutEffect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);
    document.documentElement.classList.remove('theme-pond-light', 'theme-pond-dark', 'theme-nature', 'dark');
    document.body.classList.remove('theme-pond-light', 'theme-pond-dark', 'theme-nature', 'dark');
    if (theme === 'pond-dark') {
      document.documentElement.classList.add('theme-pond-dark', 'dark');
      document.body.classList.add('theme-pond-dark', 'dark');
    } else if (theme === 'nature') {
      document.documentElement.classList.add('theme-nature', 'dark');
      document.body.classList.add('theme-nature', 'dark');
    } else {
      document.documentElement.classList.add('theme-pond-light');
      document.body.classList.add('theme-pond-light');
    }
    try { localStorage.setItem('visionmonitor-theme', theme); } catch { }
  }, [theme]);

  // ─── 15. Modal Scroll Lock ────────────────────────────────────────
  useEffect(() => {
    if (pendingSourceSwitch || settingsOpen || helpOpen) {
      lockScroll();
      return () => { unlockScroll(); };
    }
  }, [pendingSourceSwitch, settingsOpen, helpOpen]);

  // ═══════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════
  if (!systemInitialized) {
    return (
      <div
        data-theme={theme}
        className={`w-full min-w-full min-h-screen bg-[var(--bg-page)] text-[var(--text-primary)] ${theme === 'pond-dark' ? 'theme-pond-dark dark' : theme === 'nature' ? 'theme-nature dark' : 'theme-pond-light'
          }`}
      >
        <LandingScreen cameraConnected={camera.cameraConnected} onInitialize={handleInitializeSystem} />
      </div>
    );
  }

  return (
    <div
      data-theme={theme}
      className={`min-h-screen lg:h-screen lg:max-h-screen w-full min-w-full flex flex-col bg-[var(--bg-page)] text-[var(--text-primary)] ${theme === 'pond-dark' ? 'theme-pond-dark dark' : theme === 'nature' ? 'theme-nature dark' : 'theme-pond-light'}`}
    >
      <div className="fixed inset-0 pointer-events-none opacity-20 bg-[radial-gradient(var(--accent-pond)_0.8px,transparent_0.8px)] [background-size:24px_24px]" />

      <Header
        theme={theme}
        onThemeChange={handleThemeChange}
        cameraConfig={camera.effectiveCameraConfig}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenHelp={() => setHelpOpen(true)}
        fps={fps}
        anomalyDetected={anomalyFinal.anomalyStatus.isAnomaly}
        onExitToLanding={() => { clearSessionState(); setSystemInitialized(false); }}
        soundActive={soundActive}
        onToggleSound={handleToggleSound}
      />

      <div className="relative w-full max-w-[1720px] 2xl:max-w-[1920px] mx-auto px-3 sm:px-5 lg:px-6 pt-2 sm:pt-3 pb-2 sm:pb-3 flex flex-col flex-1 min-h-0 gap-2.5 sm:gap-3">
        <SourceSelector
          sourceType={sourceType}
          onSourceChange={(st) => { setSourceType(st); addLog(`Stream source switched to: ${st.toUpperCase()}`, 'info'); }}
          onRequestSwitchMode={handleRequestSwitchMode}
          isRunning={isRunning}
          isStarting={isStarting}
          isRecording={isRecording}
          onToggleRunning={handleToggleRunning}
          onStopInference={handleStopInference}
          onResumeInference={handleResumeInference}
          isStreaming={camera.isStreaming}
          onStartStream={camera.startCameraStream}
          onStopStream={handleStopStream}
          expectedDucks={expectedDucks}
          onExpectedDucksChange={(count) => {
            setExpectedDucks(count);
            addLog(`Expected duck count set to: ${count}`, 'info');
            if (video.videoSessionId) {
              fetch(`${getApiBaseUrl()}/video/update_expected/${video.videoSessionId}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count })
              }).catch(err => console.error("Failed to update expected ducks", err));
            }
            if (sourceType === 'oak-camera' || sourceType === 'webcam') {
              fetch(`${getApiBaseUrl()}/oak/inference/update_expected/live`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count })
              }).catch(err => console.error("Failed to update expected ducks for live camera", err));
            }
          }}
          onOpenSettings={() => setSettingsOpen(true)}
          customVideoName={video.customVideoName}
          onCustomVideoUploaded={video.handleVideoUploaded}
          customVideoUrl={video.customVideoUrl}
          videoSessionId={video.videoSessionId}
          hasActiveVideo={Boolean(video.customVideoUrl || video.customVideoName)}
          onClearCustomVideo={handleClearCustomVideo}
          onResetVideo={handleResetVideo}
          onResetCamera={handleResetCamera}
          isCameraConnected={camera.effectiveCameraConfig.connected}
          cameraStartingState={camera.cameraStartingState}
          cameraRecordSessionId={video.cameraRecordSessionId}
          onClearCameraRecord={handleClearCameraRecord}
          hasDetections={ducks.length > 0 || framesProcessed > 0}
        />

        <div className="w-full flex flex-col lg:flex-row items-stretch flex-1 min-h-0 gap-4">
          <main className="flex-1 w-full min-w-0 min-h-0 flex flex-col">
            <DetectionCanvas
              ducks={anomalyFinal.activeDucks}
              anomalyStatus={anomalyFinal.anomalyStatus}
              feedMode={feedMode}
              onFeedModeChange={setFeedMode}
              isRunning={isRunning}
              isStarting={isStarting}
              onToggleRunning={handleToggleRunning}
              onStopInference={handleStopInference}
              onResumeInference={handleResumeInference}
              isStreaming={camera.isStreaming}
              onRequestSwitchMode={handleRequestSwitchMode}
              fps={fps}
              sourceType={sourceType}
              customVideoUrl={video.customVideoUrl}
              videoSessionId={video.videoSessionId}
              selectedDuckId={selectedDuckId}
              onSelectDuck={setSelectedDuckId}
              onCustomVideoUploaded={video.handleVideoUploaded}
              cameraStartingState={camera.cameraStartingState}
              onCameraDeviceChange={camera.setIsCameraDeviceActive}
              expectedDucks={expectedDucks}
              videoDimensions={video.videoDimensions}
              isCameraConnected={camera.effectiveCameraConfig.connected}
              initialUploadFile={video.initialUploadFile}
              isBackendConnected={isBackendConnected}
              onRegisterTriggerUpload={(fn) => { uploadTriggerRef.current = fn; }}
              lastCameraFrame={lastCameraFrame}
              lastVideoFrame={lastVideoFrame}
              onCaptureVideoFrame={setLastVideoFrame}
              onRetryConnection={camera.startCameraStream}
              onStartStream={camera.startCameraStream}
              framesProcessed={inference.framesProcessed}
              cameraRecordSessionId={video.cameraRecordSessionId}
              cameraRecordUrl={video.cameraRecordUrl}
              cameraRecordName={video.cameraRecordName}
              onClearCameraRecord={handleClearCameraRecord}
              cameraTargetFps={camera.effectiveCameraConfig.targetFps || 30}
              recordingFormat={camera.effectiveCameraConfig.recordingFormat || 'MP4'}
            />
          </main>

          <DetectionDrawer
            isOpen={drawerOpen}
            onToggle={() => setDrawerOpen(!drawerOpen)}
            anomalyStatus={anomalyFinal.anomalyStatus}
            ducks={anomalyFinal.activeDucks}
            metrics={metrics}
            selectedDuckId={selectedDuckId}
            onSelectDuck={setSelectedDuckId}
            isStandby={isStandby}
            logs={logs}
            isCameraSource={isCameraSource}
          />
        </div>
      </div>

      {/* Safeguard modal for switching modes during active inference */}
      <Modal
        isOpen={Boolean(pendingSourceSwitch)}
        onClose={() => setPendingSourceSwitch(null)}
        title="Active Inference Running"
        description="Safeguard: Cannot switch mode while active"
        icon={<AlertTriangle className="w-5 h-5 text-[var(--accent-duck)]" />}
        footer={
          <div className="flex flex-col-reverse sm:flex-row sm:items-center sm:justify-end gap-2 w-full">
            <Button variant="secondary" onClick={() => { playWaterDropSound(); setPendingSourceSwitch(null); }}>Cancel</Button>
            <Button variant="primary" onClick={handleConfirmSwitchMode}>
              Stop &amp; Switch to {pendingSourceSwitch === 'oak-camera' ? 'Camera' : 'Video'}
            </Button>
          </div>
        }
      >
        <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
          Inference is currently active on the stream. To switch between <span className="font-bold text-[var(--text-primary)]">Offline (Video)</span> and <span className="font-bold text-[var(--text-primary)]">Online (Camera)</span> modes, please stop the active inference process first.
        </p>
      </Modal>

      {toast && <Toast type={toast.type} message={toast.message} onClose={() => setToast(null)} />}

      <CameraSettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        config={camera.effectiveCameraConfig}
        onSaveConfig={async (cfg) => {
          camera.setCameraConfig(cfg);
          try {
            const payload = {
              name: cfg.sourceName || 'OAK Camera', ip_address: cfg.ipAddress || undefined,
              resolution: cfg.resolution, fps: cfg.targetFps,
              rotation_angle: cfg.rotationAngle ?? 0, control_mode: cfg.controlMode ?? 'auto',
              exposure: cfg.exposure, gain: cfg.iso ?? cfg.gain, focus: cfg.focus,
              brightness: cfg.brightness, contrast: cfg.contrast,
              auto_focus: cfg.autoFocus, auto_exposure: cfg.autoExposure ?? true,
              recording_format: cfg.recordingFormat || 'MP4',
            };
            const savedCamera = cfg.id
              ? await cameraService.updateCamera(cfg.id, payload)
              : await cameraService.createCamera(payload);
            if (savedCamera && savedCamera.id) {
              camera.setCameraConfig((prev) => ({ ...prev, ...cfg, id: savedCamera.id }));
            }
            showToast('success', 'Camera settings saved to database');
          } catch (e) {
            console.error(e);
            showToast('error', 'Failed to save camera to database');
          }
          addLog(`Camera configuration updated [${cfg.resolution} @ ${cfg.targetFps}fps]`, 'info');
        }}
        onReconnect={async (cfg) => {
          const targetConfig = cfg || camera.effectiveCameraConfig;
          const targetFps = targetConfig.targetFps || 30;
          const targetResolution = targetConfig.resolution || '1920x1080';
          const ipAddress = targetConfig.ipAddress || '';

          addLog(`Saving & connecting OAK camera at ${ipAddress || 'USB'} [${targetResolution} @ ${targetFps}fps]...`, 'info');

          try {
            // 1. Save latest config to database first so DB always has latest resolution, FPS, and IP
            const payload = {
              name: targetConfig.sourceName || 'OAK Camera',
              ip_address: ipAddress || undefined,
              resolution: targetResolution,
              fps: targetFps,
              rotation_angle: targetConfig.rotationAngle ?? 0,
              control_mode: targetConfig.controlMode ?? 'auto',
              exposure: targetConfig.exposure,
              gain: targetConfig.iso ?? targetConfig.gain,
              focus: targetConfig.focus,
              brightness: targetConfig.brightness,
              contrast: targetConfig.contrast,
              auto_focus: targetConfig.autoFocus,
              auto_exposure: targetConfig.autoExposure ?? true,
              is_enabled: true,
            };

            const savedCamera = targetConfig.id
              ? await cameraService.updateCamera(targetConfig.id, payload)
              : await cameraService.createCamera(payload);

            const updatedId = savedCamera?.id || targetConfig.id;

            camera.setCameraConfig((prev) => ({
              ...prev,
              ...targetConfig,
              id: updatedId,
              targetFps,
              resolution: targetResolution,
            }));

            // 2. Stop running stream/pipeline if already active to rebuild cleanly with new resolution & FPS
            try {
              await cameraService.stopLiveInference();
            } catch { }
            try {
              await cameraService.stopStream();
            } catch { }
            try {
              await cameraService.stop();
            } catch { }

            // 3. Connect & start pipeline with updated camera settings
            const startRes = await cameraService.start({
              camera_id: updatedId,
              ip_address: ipAddress || undefined,
            });

            if (startRes?.status === 'error') {
              throw new Error(startRes.message || 'Camera failed to connect');
            }

            camera.setIsCameraDeviceActive(true);

            // 4. Start the live video stream
            const streamRes = await cameraService.startStream();
            if (streamRes?.status === 'error') {
              throw new Error(streamRes.message || 'Failed to start stream');
            }

            camera.setIsStreaming(true);
            camera.setCameraStartingState('ready');
            camera.setCameraConnected(true);

            showToast('success', `Connected to OAK Camera [${targetResolution} @ ${targetFps} FPS]`);
            addLog(`OAK Camera connected successfully [${targetResolution} @ ${targetFps} FPS]`, 'success');
          } catch (err: any) {
            console.error('Camera connection error:', err);
            camera.setIsCameraDeviceActive(false);
            camera.setIsStreaming(false);
            camera.setCameraConnected(false);
            showToast('error', err.message || 'Failed to connect to camera');
            addLog(`Failed to connect to camera: ${err.message || 'Unknown error'}`, 'error');
            throw err;
          }
        }}
      />

      <HelpModal isOpen={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
