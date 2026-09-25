import React from 'react';
import { CameraOff, Video, RotateCcw } from 'lucide-react';

interface CameraOfflineCardProps {
  onSwitchToVideo: () => void;
  onRetryConnection?: () => void;
  onCanvasClick: (e: React.MouseEvent<HTMLDivElement>) => void;
}

export const CameraOfflineCard: React.FC<CameraOfflineCardProps> = ({
  onSwitchToVideo,
  onRetryConnection,
  onCanvasClick
}) => {
  return (
    <div 
      onClick={onCanvasClick}
      className="absolute inset-0 w-full h-full flex flex-col items-center justify-center text-center p-6 rounded-3xl z-10"
      style={{ backgroundColor: 'var(--bg-card)' }}
    >
      {/* Standalone Icon matching Image 1 style - NO box */}
      <div className="text-[var(--status-anomaly-text)] mb-2 sm:mb-3 shrink-0">
        <CameraOff className="w-10 h-10 sm:w-12 sm:h-12 lg:w-14 lg:h-14 stroke-[1.75]" />
      </div>
      
      {/* Status Badge: theme style, not a round pill */}
      <div className="inline-flex items-center gap-1.5 px-2.5 sm:px-3.5 py-1 rounded-lg bg-[var(--status-anomaly-bg)] border border-[var(--status-anomaly-border)] text-[var(--status-anomaly-text)] text-[10px] sm:text-xs font-mono font-semibold mb-2.5 shadow-xs">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--status-anomaly-text)] animate-pulse" />
        NO OAK CAMERA DETECTED
      </div>

      <h3 className="text-lg sm:text-xl lg:text-2xl font-bold text-[var(--text-primary)] mb-1 sm:mb-2 shrink-0">
        OAK-D Hardware Offline
      </h3>
      <p className="text-xs sm:text-sm lg:text-base text-[var(--text-secondary)] max-w-md mb-5 leading-relaxed font-medium">
        No Luxonis OAK-D / USB camera is currently connected. Connect a DepthAI device or switch back to Video mode to run inference.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-2.5">
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
            onClick={(e) => {
              e.stopPropagation();
              onRetryConnection();
            }}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--btn-secondary-border)] text-[var(--btn-secondary-text)] font-semibold text-xs transition-all cursor-pointer active:scale-95"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Retry Connection</span>
          </button>
        )}
      </div>
    </div>
  );
};
