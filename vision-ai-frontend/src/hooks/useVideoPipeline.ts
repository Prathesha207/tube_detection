import { useState, useEffect } from 'react';
import type { StreamSourceType, LogEntry } from '../types';
import { getApiBaseUrl } from '../lib/api';
import { useInferenceStore } from '../store/inferenceStore';
import { resetBBoxCache } from '../utils/mlDataMapper';
import { loadSessionState, saveSessionState } from '../utils/sessionPersistence';

export function useVideoPipeline({
  showToast,
  addLog,
  expectedDucks,
  sourceType,
  isRunning,
  setIsRunning,
  setDucks,
  setFramesProcessed,
  setFps,
  setCameraStartingState,
  setSourceType,
  setIsStarting
}: {
  showToast: (type: 'error' | 'success' | 'info', message: string) => void;
  addLog: (message: string, level?: LogEntry['level']) => void;
  expectedDucks: number;
  sourceType: StreamSourceType;
  isRunning: boolean;
  setIsRunning: (running: boolean) => void;
  setDucks: (ducks: any[]) => void;
  setFramesProcessed: (count: number) => void;
  setFps: (fps: number) => void;
  setCameraStartingState: (state: any) => void;
  setSourceType: (type: StreamSourceType) => void;
  setIsStarting: (starting: boolean) => void;
}) {
  const _session = loadSessionState();
  const [customVideoUrl, setCustomVideoUrl] = useState<string | undefined>(_session?.customVideoUrl);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | undefined>(_session?.customVideoUrl);
  const [videoSessionId, setVideoSessionId] = useState<string | null>(_session?.videoSessionId ?? null);
  const [customVideoName, setCustomVideoName] = useState<string | undefined>(_session?.customVideoName);
  const [autoStartRecordedInference, setAutoStartRecordedInference] = useState(false);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [initialUploadFile, setInitialUploadFile] = useState<File | undefined>();

  // Dedicated state for camera recording sessions (kept inside Camera Inference Area)
  const [cameraRecordSessionId, setCameraRecordSessionId] = useState<string | null>(null);
  const [cameraRecordUrl, setCameraRecordUrl] = useState<string | undefined>(undefined);
  const [cameraRecordName, setCameraRecordName] = useState<string | undefined>(undefined);

  const clearCameraRecording = () => {
    if (cameraRecordSessionId) {
      fetch(`${getApiBaseUrl()}/video/clear/${cameraRecordSessionId}`, { method: 'POST' }).catch(() => {});
    }
    setCameraRecordSessionId(null);
    setCameraRecordUrl(undefined);
    setCameraRecordName(undefined);
    setIsRunning(false);
    setDucks([]);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();
    addLog('Camera recording cleared • Returned to live camera feed.', 'info');
  };

  // Persist video session state on every change so Ctrl+R can reconnect
  useEffect(() => {
    saveSessionState({ videoSessionId: videoSessionId ?? null, customVideoUrl, customVideoName });
  }, [videoSessionId, customVideoUrl, customVideoName]);

  // Custom uploaded video handler (Directly in video canvas)
  const handleVideoUploaded = async (
    url: string,
    name: string,
    sessionId?: string,
    isCameraRecording?: boolean
  ): Promise<void> => {
    const isRecordedStream = Boolean(isCameraRecording);

    if (isRecordedStream) {
      if (sessionId) {
        setVideoSessionId(sessionId);
      }
      setLocalPreviewUrl(url);
      setCustomVideoUrl(url);
      setCustomVideoName(name);
      setSourceType('uploaded-video');
      setCameraRecordSessionId(null);
      setCameraRecordUrl(undefined);
      setCameraRecordName(undefined);
      setCameraStartingState('ready');
      setDucks([]);
      setFramesProcessed(0);
      setFps(0);
      useInferenceStore.getState().resetStats();
      resetBBoxCache();
      if (sessionId) {
        await startVideoInference(sessionId);
      }
      return;
    }

    // If online camera is currently active, block upload with alert as per requirement (unless it's a recorded stream)
    const isCameraActive = (sourceType === 'oak-camera' || sourceType === 'webcam') && isRunning;
    if (isCameraActive) {
      showToast('error', 'Disable camera before uploading video');
      addLog('Blocked upload: Disable camera before uploading video', 'anomaly');
      return;
    }

    if (sessionId) {
      setVideoSessionId(sessionId);
    }

    if (name.toLowerCase().includes('_annotated') || name.toLowerCase().startsWith('annotated_')) {
      showToast('info', `Note: "${name}" is an already-annotated video. For clean inference, upload raw videos from D:/ml/test-videos/`);
      addLog(`Warning: "${name}" contains baked-in annotations from a previous run.`, 'anomaly');
    }

    // Always update local preview and customVideoUrl so the video frame shows immediately
    const isStreamUrl = url.includes('/video/stream/');
    setLocalPreviewUrl(url);
    setCustomVideoUrl(url);
    setCustomVideoName(name);
    setSourceType('uploaded-video');
    setAutoStartRecordedInference(false);
    setCameraStartingState('ready');

    // Clean slate: clear prior detections, frame counts, and stats for the new video
    setDucks([]);
    setFramesProcessed(0);
    setFps(0);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();

    if (sessionId) {
      setVideoSessionId(sessionId);
      await startVideoInference(sessionId);
    } else {
      setIsRunning(false);
      showToast('success', `Video ready. Click "Start Inference" to evaluate.`);
      addLog(`Video loaded: "${name}". Ready for inference.`, 'info');
    }
  };

  const handleClearVideo = () => {
    // If backend video was running, signal stop and clear temporary session files
    if (videoSessionId) {
      fetch(`${getApiBaseUrl()}/video/clear/${videoSessionId}`, { method: 'POST' })
        .catch(() => {});
    }
    setCustomVideoUrl(undefined);
    setLocalPreviewUrl(undefined);
    setVideoSessionId(null);
    setCustomVideoName(undefined);
    setSourceType('uploaded-video');
    setIsRunning(false);
    setDucks([]);
    setFramesProcessed(0);
    setFps(0);
    useInferenceStore.getState().resetStats();
    showToast('info', 'Video cleared.');
    addLog('Reset video canvas and cleared all detections.', 'info');
  };

  const startVideoInference = async (customSessionId?: string) => { 
    setDucks([]); 
    useInferenceStore.getState().resetStats(); 
    resetBBoxCache();
    let sid = customSessionId || videoSessionId;
    if (!sid && (customVideoUrl || localPreviewUrl)) {
      const urlToTest = customVideoUrl || localPreviewUrl || '';
      const match = urlToTest.match(/\/video\/(?:stream|last_frame|raw)\/([a-zA-Z0-9_-]+)/);
      if (match && match[1]) {
        sid = match[1];
        setVideoSessionId(sid);
      }
    }
    if (!sid) {
      showToast('error', 'Upload a video before starting inference.');
      return;
    }
    setFramesProcessed(0);
    setFps(0);
    setDucks([]);
    useInferenceStore.getState().resetStats();
    resetBBoxCache();
    setIsStarting(true);
    try {
      // Sync the expected count to the backend RIGHT BEFORE starting inference!
      // This guarantees the backend always warms up with the number currently shown on screen.
      await fetch(`${getApiBaseUrl()}/video/update_expected/${sid}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: expectedDucks })
      }).catch(err => console.error("Failed to sync expected ducks before start", err));

      const response = await fetch(`${getApiBaseUrl()}/video/start/${sid}`, { method: 'POST' });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.message || 'Unable to start inference');
      }
      setIsRunning(true);
      showToast('success', 'Inference started');
      addLog('Inference active on uploaded video.', 'success');
    } catch (error) {
      setIsRunning(false);
      showToast('error', error instanceof Error ? error.message : 'Unable to start inference');
    } finally {
      setIsStarting(false);
    }
  };

  // A camera recording is already a completed input video. Start its normal
  // uploaded-video inference flow automatically after the upload returns a
  // backend session ID, so Record -> review -> inference is one workflow.
  useEffect(() => {
    if (!autoStartRecordedInference || !videoSessionId || sourceType !== 'uploaded-video') return;
    setAutoStartRecordedInference(false);
    void startVideoInference();
  }, [autoStartRecordedInference, videoSessionId, sourceType]);

  return {
    customVideoUrl,
    setCustomVideoUrl,
    localPreviewUrl,
    setLocalPreviewUrl,
    videoSessionId,
    setVideoSessionId,
    customVideoName,
    setCustomVideoName,
    autoStartRecordedInference,
    setAutoStartRecordedInference,
    videoDimensions,
    setVideoDimensions,
    initialUploadFile,
    setInitialUploadFile,
    handleVideoUploaded,
    handleClearVideo,
    startVideoInference,
    cameraRecordSessionId,
    setCameraRecordSessionId,
    cameraRecordUrl,
    setCameraRecordUrl,
    cameraRecordName,
    setCameraRecordName,
    clearCameraRecording,
  };
}
