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

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isStarting,
  isCameraSource,
  isVideoSource,
  hasCameraRecording,
  cameraStartingState = 'ready',
  hasActiveVideo,
  isRunning,
  isFirstFrameLoaded,
  isCameraConnected = false,
}) => {
  // Never show loading overlay if camera mode is active but camera is offline/disconnected (unless a recording is loaded)
  if (isCameraSource && !hasCameraRecording && !isCameraConnected) {
    return null;
  }

  return (
    <>
      {isStarting && (
        <div className="absolute inset-0 z-25 bg-black/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center animate-in fade-in duration-300">
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-emerald-950/80 border border-emerald-700/60 flex items-center justify-center mb-4 text-emerald-400">
            <Loader2 className="w-7 h-7 sm:w-8 sm:h-8 animate-spin" />
          </div>
          <h3 className="text-base sm:text-lg font-bold text-white mb-1">
            Starting Inference...
          </h3>
          <p className="text-xs text-emerald-300/80 max-w-md">
            Please wait while we connect to the video stream and initialize the AI model.
          </p>
        </div>
      )}
      
      {!isStarting && isCameraSource && cameraStartingState !== 'ready' && (
        <div className="absolute inset-0 z-25 bg-black/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center animate-in fade-in duration-300">
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-emerald-950/80 border border-emerald-700/60 flex items-center justify-center mb-4 text-emerald-400">
            <Loader2 className="w-7 h-7 sm:w-8 sm:h-8 animate-spin" />
          </div>
          <h3 className="text-base sm:text-lg font-bold text-white mb-1">
            {cameraStartingState === 'waking_camera' && 'Waking up OAK Camera Stream...'}
            {cameraStartingState === 'waiting_frame' && 'Receiving First Live Frame (1080p)...'}
          </h3>
          <p className="text-xs text-emerald-300/80 max-w-md font-mono">
            {cameraStartingState === 'waking_camera' && 'Executing POST /oak/start -> Initializing sensor pipeline'}
            {cameraStartingState === 'waiting_frame' && 'Executing waitForFirstFrame() -> Connecting YOLOv8 inference buffer'}
          </p>
        </div>
      )}

      {((isVideoSource && hasActiveVideo) || hasCameraRecording) && isRunning && !isFirstFrameLoaded && (
        <div className="absolute inset-0 z-25 bg-black/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center animate-in fade-in duration-300">
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-emerald-950/80 border border-emerald-700/60 flex items-center justify-center mb-4 text-emerald-400">
            <Loader2 className="w-7 h-7 sm:w-8 sm:h-8 animate-spin" />
          </div>
          <h3 className="text-base sm:text-lg font-bold text-white mb-1">
            Inference is Running...
          </h3>
          <p className="text-xs text-emerald-300/80 max-w-md font-mono">
            Connecting to stream &amp; processing initial video frames...
          </p>
        </div>
      )}
    </>
  );
};
