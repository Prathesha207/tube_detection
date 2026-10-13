import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingOverlayProps {
  isStarting?: boolean;
  isCameraSource: boolean;
  isVideoSource: boolean;
  hasCameraRecording?: boolean;
  cameraStartingState?: 'idle' | 'waking_camera' | 'waiting_frame' | 'ready';
  hasActiveVideo: boolean;
  isRunning: boolean;
  isFirstFrameLoaded: boolean;
  isCameraConnected?: boolean;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = () => {
  // Never block the canvas with dark overlays; show video and camera frames directly
  return null;
};
