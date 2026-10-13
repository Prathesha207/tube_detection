import React from 'react';
import { Camera, CameraOff, Video, RotateCcw, Loader2 } from 'lucide-react';

interface CameraOfflineCardProps {
  onSwitchToVideo: () => void;
  onRetryConnection?: () => void;
  onCanvasClick: (e: React.MouseEvent<HTMLDivElement>) => void;
  errorMessage?: string | null;
  isDeviceDetected?: boolean;
  isConnecting?: boolean;
}

export const CameraOfflineCard: React.FC<CameraOfflineCardProps> = ({
  onSwitchToVideo,
  onRetryConnection,
  onCanvasClick,
  errorMessage,
  isDeviceDetected = false,
  isConnecting = false,
}) => {
  const isDeviceInUse = Boolean(
    errorMessage?.toLowerCase().includes('already used') || 
    errorMessage?.toLowerCase().includes('in use')
  );

  // Determine presentation based on detection & error status
  const isDetected = isDeviceDetected && !errorMessage;

  const badgeText = isConnecting
    ? 'CONNECTING TO OAK CAMERA...'
    : isDeviceInUse
    ? 'DEVICE IN USE BY ANOTHER PROCESS'
    : errorMessage
    ? 'CAMERA UNAVAILABLE'
    : isDetected
    ? 'OAK CAMERA DETECTED'
    : 'NO OAK CAMERA DETECTED';

  const titleText = isConnecting
    ? 'Connecting to Camera...'
    : isDeviceInUse
    ? 'Camera Already In Use'
    : errorMessage
    ? 'Camera Connection Failed'
    : isDetected
    ? 'OAK Camera Detected'
    : 'OAK-D Hardware Offline';

  const descriptionText = isConnecting
    ? 'Initializing DepthAI hardware pipeline and establishing video connection...'
    : isDeviceInUse
    ? 'Device is already used by another application/process. Make sure to close all applications/processes using the device before starting a new one.'
    : errorMessage
    ? errorMessage
    : isDetected
    ? 'Luxonis OAK-D camera hardware is connected and detected. Click Connect Camera to start the live video feed.'
    : 'No Luxonis OAK-D / USB camera is currently connected. Connect a DepthAI device or switch back to Video mode to run inference.';

  const isGreenState = isDetected || isConnecting;

  return (
    <div 
      onClick={onCanvasClick}
      className="absolute inset-0 w-full h-full flex flex-col items-center justify-center text-center p-6 rounded-3xl z-10"
      style={{ backgroundColor: 'var(--bg-card)' }}
    >
      {/* Standalone Icon */}
      <div className={`mb-2 sm:mb-3 shrink-0 ${isGreenState ? 'text-emerald-500' : 'text-[var(--status-anomaly-text)]'}`}>
        {isConnecting ? (
          <Loader2 className="w-10 h-10 sm:w-12 sm:h-12 lg:w-14 lg:h-14 stroke-[1.75] animate-spin" />
        ) : isGreenState ? (
          <Camera className="w-10 h-10 sm:w-12 sm:h-12 lg:w-14 lg:h-14 stroke-[1.75]" />
        ) : (
          <CameraOff className="w-10 h-10 sm:w-12 sm:h-12 lg:w-14 lg:h-14 stroke-[1.75]" />
        )}
      </div>
      
      {/* Status Badge */}
      <div 
        className={`inline-flex items-center gap-1.5 px-2.5 sm:px-3.5 py-1 rounded-lg text-[10px] sm:text-xs font-mono font-semibold mb-2.5 shadow-xs border ${
          isGreenState
            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
            : 'bg-[var(--status-anomaly-bg)] border-[var(--status-anomaly-border)] text-[var(--status-anomaly-text)]'
        }`}
      >
        <span 
          className={`w-1.5 h-1.5 rounded-full ${
            isGreenState ? 'bg-emerald-500 animate-pulse' : 'bg-[var(--status-anomaly-text)] animate-pulse'
          }`} 
        />
        {badgeText}
      </div>

      <h3 className="text-lg sm:text-xl lg:text-2xl font-bold text-[var(--text-primary)] mb-1 sm:mb-2 shrink-0">
        {titleText}
      </h3>
      <p className="text-xs sm:text-sm lg:text-base text-[var(--text-secondary)] max-w-md mb-5 leading-relaxed font-medium">
        {descriptionText}
      </p>

      <div className="flex flex-wrap items-center justify-center gap-2.5">
        {isDetected ? (
          <>
            {onRetryConnection && (
              <button
                type="button"
                disabled={isConnecting}
                onClick={(e) => {
                  e.stopPropagation();
                  onRetryConnection();
                }}
                className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-bold text-xs shadow-xs transition-all cursor-pointer active:scale-95"
              >
                {isConnecting ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Connecting...</span>
                  </>
                ) : (
                  <>
                    <Camera className="w-3.5 h-3.5" />
                    <span>Connect Camera</span>
                  </>
                )}
              </button>
            )}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onSwitchToVideo();
              }}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--btn-secondary-border)] text-[var(--btn-secondary-text)] font-semibold text-xs transition-all cursor-pointer active:scale-95"
            >
              <Video className="w-3.5 h-3.5" />
              <span>Switch to Video Mode</span>
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onSwitchToVideo();
              }}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] font-bold text-xs shadow-xs transition-all cursor-pointer active:scale-95"
            >
              <Video className="w-3.5 h-3.5" />
              <span>Switch to Video Mode</span>
            </button>
            {onRetryConnection && (
              <button
                type="button"
                disabled={isConnecting}
                onClick={(e) => {
                  e.stopPropagation();
                  onRetryConnection();
                }}
                className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--btn-secondary-border)] text-[var(--btn-secondary-text)] font-semibold text-xs transition-all cursor-pointer active:scale-95"
              >
                {isConnecting ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RotateCcw className="w-3.5 h-3.5" />
                )}
                <span>{isConnecting ? 'Connecting...' : 'Retry Connection'}</span>
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
};
