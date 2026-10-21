import { useState, useEffect, useMemo, useRef } from 'react';
import type { CameraConfig, LogEntry, StreamSourceType } from '../types';
import { DEFAULT_CAMERA_CONFIG } from '../data/cameraData';
import { cameraService } from '../components/service/cameraService';
import { getApiBaseUrl } from '../lib/api';

export function useCameraStatus(
  addLog: (message: string, level?: LogEntry['level']) => void,
  showToast: (type: 'error' | 'success' | 'info', message: string) => void,
  sourceType?: StreamSourceType
) {
  const [cameraConfig, setCameraConfig] = useState<CameraConfig>({
    ...DEFAULT_CAMERA_CONFIG,
    connected: false,
  });
  const [isCameraDeviceActive, setIsCameraDeviceActive] = useState<boolean>(false);
  const [isDeviceDetected, setIsDeviceDetected] = useState<boolean>(false);
  const [cameraStartingState, setCameraStartingState] = useState<'idle' | 'waking_camera' | 'waiting_frame' | 'ready'>('ready');
  const [cameraConnected, setCameraConnected] = useState<boolean | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const autoConnectAttemptedRef = useRef<boolean>(false);

  // Automatically fetch latest saved camera from database on mount / refresh
  useEffect(() => {
    async function loadSavedCamera() {
      try {
        const storageRes = await cameraService.getStoragePath().catch(() => null);
        if (storageRes?.storage_path) {
          setCameraConfig((prev) => ({
            ...prev,
            recordingVideoPath: storageRes.storage_path,
          }));
        }

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
              autoFocus: false,
              autoExposure: false,
              recordingFormat: (activeCamera.recording_format as any) || prev.recordingFormat || 'MP4',
              recordingVideoPath: activeCamera.recording_video_path || storageRes?.storage_path || prev.recordingVideoPath,
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
            // STRICT: Only online if the hardware connection is actually verified as true
            const isOnline = Boolean(data.connected);
            const isDetected = Boolean(data.device_detected);
            setIsDeviceDetected(isDetected);

            if (!isDetected) {
              autoConnectAttemptedRef.current = false;
            }

            const settings = data.camera_settings;
            setCameraConfig((prev) => {
              const wasOffline = !prev.connected;
              return {
                ...prev,
                connected: isOnline,
                // Only overwrite controls when transitioning from offline to online,
                // so user slider edits don't snap back every 5 seconds!
                ...(settings && wasOffline ? {
                  controlMode: settings.control_mode,
                  exposure: settings.exposure,
                  gain: settings.gain,
                  iso: settings.gain,
                  focus: settings.focus ?? undefined,
                  focusAvailable: Boolean(settings.focus_available),
                  brightness: settings.brightness,
                  contrast: settings.contrast,
                  autoFocus: false,
                  autoExposure: false,
                } : (settings ? { focusAvailable: Boolean(settings.focus_available) } : {})),
              };
            });
            setIsCameraDeviceActive(isOnline);
            setCameraConnected(isOnline);
            if (data.last_error) {
              setCameraError(data.last_error);
            } else if (isOnline) {
              setCameraError(null);
            }
            // Force offline states if the hardware is not connected
            if (!isOnline) {
              setIsStreaming(false);
              setCameraStartingState((prev) => prev !== 'idle' ? 'idle' : prev);

              // Auto-connect when hardware is detected in camera mode and no error occurred
              // Auto-connect and start stream when hardware is detected in camera mode and no error occurred
              const isCameraMode = sourceType === 'oak-camera' || sourceType === 'webcam';
              if (isDetected && isCameraMode && !autoConnectAttemptedRef.current && !data.last_error) {
                autoConnectAttemptedRef.current = true;
                cameraService.start().then(async (startRes) => {
                  if (startRes?.status === 'started' || startRes?.status === 'already_running') {
                    setIsCameraDeviceActive(true);
                    setCameraConnected(true);
                    setCameraError(null);
                    // Automatically stream immediately so the user sees live video right away
                    try {
                      await cameraService.startStream();
                      setIsStreaming(true);
                      setCameraStartingState('ready');
                    } catch (streamErr) {
                      console.warn('Auto-stream on device detection failed:', streamErr);
                    }
                  } else if (startRes?.status === 'error') {
                    setIsCameraDeviceActive(false);
                    setCameraConnected(false);
                    setCameraError(startRes.message || 'Camera connection failed');
                    setCameraStartingState('ready');
                  }
                }).catch((err) => {
                  console.warn('Auto-connect on device detection failed:', err);
                  setIsCameraDeviceActive(false);
                  setCameraConnected(false);
                  setCameraStartingState('ready');
                });
              }
            }
          }
        } else {
          if (isMounted) {
            setCameraConfig((prev) => (prev.connected ? { ...prev, connected: false } : prev));
            setIsCameraDeviceActive(false);
            setCameraConnected(false);
            setIsStreaming(false);
            setIsDeviceDetected(false);
            setCameraStartingState('ready');
          }
        }
      } catch {
        if (isMounted) {
          setCameraConfig((prev) => (prev.connected ? { ...prev, connected: false } : prev));
          setIsCameraDeviceActive(false);
          setCameraConnected(false);
          setIsStreaming(false);
          setIsDeviceDetected(false);
          setCameraStartingState('ready');
        }
      }
    }

    checkCameraStatus();
    const interval = setInterval(checkCameraStatus, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [sourceType]);

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
        setCameraConnected(true);
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
      setIsStreaming(false);
      setIsCameraDeviceActive(false);
      setCameraConnected(false);
      const errMsg = error?.response?.data?.message || error?.response?.data?.detail || error?.message || 'Unable to start camera stream';
      setCameraError(errMsg);
      showToast('error', errMsg);
      addLog(`Error: ${errMsg}`, 'error');
      throw error;
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
    isDeviceDetected,
    setIsDeviceDetected,
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
