import { useState, useEffect, useMemo } from 'react';
import type { CameraConfig, LogEntry } from '../types';
import { DEFAULT_CAMERA_CONFIG } from '../data/cameraData';
import { cameraService } from '../components/service/cameraService';
import { getApiBaseUrl } from '../lib/api';

export function useCameraStatus(
  addLog: (message: string, level?: LogEntry['level']) => void,
  showToast: (type: 'error' | 'success' | 'info', message: string) => void
) {
  const [cameraConfig, setCameraConfig] = useState<CameraConfig>({
    ...DEFAULT_CAMERA_CONFIG,
    connected: false,
  });
  const [isCameraDeviceActive, setIsCameraDeviceActive] = useState<boolean>(false);
  const [cameraStartingState, setCameraStartingState] = useState<'idle' | 'waking_camera' | 'waiting_frame' | 'ready'>('idle');
  const [cameraConnected, setCameraConnected] = useState<boolean | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Automatically fetch latest saved camera from database on mount / refresh
  useEffect(() => {
    async function loadSavedCamera() {
      try {
        const cameras = await cameraService.checkCamera();
        if (Array.isArray(cameras) && cameras.length > 0) {
          const activeCamera =
            [...cameras].sort((a: any, b: any) => (b.id ?? 0) - (a.id ?? 0)).find((c: any) => c.is_enabled) ||
            cameras[0];
          if (activeCamera) {
            setCameraConfig((prev) => ({
              ...prev,
              id: activeCamera.id,
              sourceName: activeCamera.name || prev.sourceName,
              ipAddress: activeCamera.ip_address || '',
              resolution: (activeCamera.resolution as any) || prev.resolution,
              targetFps: activeCamera.fps || prev.targetFps,
              rotationAngle: activeCamera.rotation_angle ?? prev.rotationAngle,
              controlMode: (activeCamera.control_mode as any) ?? prev.controlMode,
              exposure: activeCamera.exposure ?? prev.exposure,
              gain: activeCamera.gain ?? prev.gain,
              iso: activeCamera.gain ?? prev.iso,
              focus: activeCamera.focus ?? prev.focus,
              brightness: activeCamera.brightness ?? prev.brightness,
              contrast: activeCamera.contrast ?? prev.contrast,
              autoFocus: activeCamera.auto_focus ?? prev.autoFocus,
              autoExposure: activeCamera.auto_exposure ?? prev.autoExposure,
              recordingFormat: (activeCamera.recording_format as any) || prev.recordingFormat || 'MP4',
            }));
            addLog(`Loaded camera "${activeCamera.name}" from database [${activeCamera.resolution || '1080p'} @ ${activeCamera.fps || 30}fps]`, 'info');
          }
        }
      } catch (err) {
        console.error("Could not fetch camera from database:", err);
      }
    }
    loadSavedCamera();
  }, [addLog]);

  // Periodically check live camera hardware connection status
  useEffect(() => {
    let isMounted = true;
    async function checkCameraStatus() {
      try {
        const baseUrl = getApiBaseUrl();
        const res = await fetch(`${baseUrl}/oak/health`);
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            const isOnline = Boolean(data.connected || data.running || data.streaming);
            setCameraConfig((prev) => (prev.connected !== isOnline ? { ...prev, connected: isOnline } : prev));
            setIsCameraDeviceActive(isOnline);
            setCameraConnected(isOnline);
            if (data.last_error) {
              setCameraError(data.last_error);
            } else if (isOnline) {
              setCameraError(null);
            }
            // Only force false if the hardware is completely offline
            if (!isOnline) {
              setIsStreaming(false);
            }
          }
        } else {
          if (isMounted) {
            setCameraConfig((prev) => (prev.connected ? { ...prev, connected: false } : prev));
            setIsCameraDeviceActive(false);
            setCameraConnected(false);
            setIsStreaming(false);
          }
        }
      } catch {
        if (isMounted) {
          setCameraConfig((prev) => (prev.connected ? { ...prev, connected: false } : prev));
          setIsCameraDeviceActive(false);
          setCameraConnected(false);
          setIsStreaming(false);
        }
      }
    }

    checkCameraStatus();
    const interval = setInterval(checkCameraStatus, 5000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Dynamic camera config that accurately reflects hardware connection
  const effectiveCameraConfig = useMemo(() => ({
    ...cameraConfig,
    connected: isCameraDeviceActive,
  }), [cameraConfig, isCameraDeviceActive]);

  const startCameraStream = async () => {
    setCameraStartingState('waking_camera');
    try {
      setCameraError(null);
      if (!isCameraDeviceActive) {
        const startRes = await cameraService.start();
        if (startRes?.status === 'error') throw new Error(startRes.message || 'Camera start failed');
        setIsCameraDeviceActive(true);
      }
      const streamRes = await cameraService.startStream();
      if (streamRes?.status === 'error') throw new Error(streamRes.message || 'Stream start failed');
      setIsStreaming(true);
      setCameraStartingState('ready');
      setCameraError(null);
      showToast('success', 'Camera stream started');
      addLog('Camera stream started. Ready for display or inference.', 'success');
    } catch (error: any) {
      setCameraStartingState('ready');
      const errMsg = error?.response?.data?.message || error?.response?.data?.detail || error?.message || 'Unable to start camera stream';
      setCameraError(errMsg);
      showToast('error', errMsg);
      addLog(`Error: ${errMsg}`, 'error');
    }
  };

  const stopCameraStream = async (
    _isRunning?: boolean,
    setTubes?: (tubes: any[]) => void,
    setIsRunning?: (r: boolean) => void
  ) => {
    try { await cameraService.stopLiveInference(); } catch { /* stream can still stop */ }
    setIsRunning?.(false);
    setTubes?.([]);
    try {
      await cameraService.stopStream();
      setIsStreaming(false);
      showToast('info', 'Camera stream stopped');
      addLog('Camera stream stopped • Ready for fresh stream.', 'info');
    } catch {
      showToast('error', 'Unable to stop camera stream');
    }
  };

  return {
    cameraConfig,
    setCameraConfig,
    isCameraDeviceActive,
    setIsCameraDeviceActive,
    cameraStartingState,
    setCameraStartingState,
    cameraConnected,
    setCameraConnected,
    isStreaming,
    setIsStreaming,
    effectiveCameraConfig,
    startCameraStream,
    stopCameraStream,
    cameraError,
    setCameraError
  };
}
