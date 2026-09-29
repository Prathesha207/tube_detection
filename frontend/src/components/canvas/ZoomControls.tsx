import React from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Move } from 'lucide-react';

interface ZoomControlsProps {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  onSetZoom: (zoom: number) => void;
  className?: string;
  isCompact?: boolean;
}

export const ZoomControls: React.FC<ZoomControlsProps> = ({
  zoom,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  onSetZoom,
  className = '',
  isCompact = false,
}) => {
  const percentage = Math.round(zoom * 100);
  const isZoomed = zoom > 1.02;

  const cycleZoom = () => {
    if (zoom < 1.4) {
      onSetZoom(1.5);
    } else if (zoom < 1.9) {
      onSetZoom(2.0);
    } else if (zoom < 2.9) {
      onSetZoom(3.0);
    } else {
      onResetZoom();
    }
  };

  return (
    <div
      className={`pointer-events-auto flex items-center gap-1 sm:gap-1.5 p-1 rounded-2xl bg-[var(--bg-card)]/95 dark:bg-slate-900/90 backdrop-blur-md border border-[var(--border-color)] dark:border-slate-700/80 shadow-xl select-none z-30 transition-all ${className}`}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Zoom Out Button */}
      <button
        type="button"
        onClick={onZoomOut}
        disabled={zoom <= 1.0}
        title="Zoom Out (-) [Mouse wheel down]"
        aria-label="Zoom Out"
        className={`w-7 h-7 sm:w-8 sm:h-8 flex items-center justify-center rounded-xl transition-all cursor-pointer ${
          zoom <= 1.0
            ? 'opacity-35 cursor-not-allowed text-slate-400 dark:text-slate-600'
            : 'text-[var(--text-primary)] hover:bg-cyan-500/15 hover:text-cyan-500 dark:hover:text-cyan-400 active:scale-95'
        }`}
      >
        <ZoomOut className="w-3.5 h-3.5 sm:w-4 sm:h-4 stroke-[2.2]" />
      </button>

      {/* Clickable Zoom Percentage Badge */}
      <button
        type="button"
        onClick={cycleZoom}
        title="Click to cycle zoom (100% -> 150% -> 200% -> 300% -> 100%)"
        aria-label={`Zoom level: ${percentage} percent`}
        className={`h-7 sm:h-8 px-2 sm:px-2.5 rounded-xl font-mono text-[11px] sm:text-xs font-bold tracking-wide flex items-center gap-1 cursor-pointer transition-all ${
          isZoomed
            ? 'bg-cyan-500/20 border border-cyan-500/50 text-cyan-500 dark:text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.35)]'
            : 'text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
        }`}
      >
        <span>{percentage}%</span>
        {isZoomed && <Move className="w-2.5 h-2.5 opacity-80 animate-pulse text-cyan-400" />}
      </button>

      {/* Zoom In Button */}
      <button
        type="button"
        onClick={onZoomIn}
        disabled={zoom >= 5.0}
        title="Zoom In (+) [Mouse wheel up or double-click]"
        aria-label="Zoom In"
        className={`w-7 h-7 sm:w-8 sm:h-8 flex items-center justify-center rounded-xl transition-all cursor-pointer ${
          zoom >= 5.0
            ? 'opacity-35 cursor-not-allowed text-slate-400 dark:text-slate-600'
            : 'text-[var(--text-primary)] hover:bg-cyan-500/15 hover:text-cyan-500 dark:hover:text-cyan-400 active:scale-95'
        }`}
      >
        <ZoomIn className="w-3.5 h-3.5 sm:w-4 sm:h-4 stroke-[2.2]" />
      </button>

      {/* Quick Reset / 1x Button (shown when zoomed) */}
      {isZoomed && (
        <button
          type="button"
          onClick={onResetZoom}
          title="Reset Zoom to 100% (0) [Double-click]"
          aria-label="Reset Zoom"
          className="h-7 sm:h-8 px-2 flex items-center gap-1 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/50 text-amber-500 dark:text-amber-400 font-bold text-[10px] sm:text-[11px] transition-all active:scale-95 cursor-pointer shadow-xs animate-in fade-in zoom-in-95 duration-150"
        >
          <RotateCcw className="w-3 h-3 stroke-[2.2]" />
          {!isCompact && <span>1x</span>}
        </button>
      )}
    </div>
  );
};
