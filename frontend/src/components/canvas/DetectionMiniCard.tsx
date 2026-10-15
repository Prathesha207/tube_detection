import React, { useState } from 'react';
import type { TubeEntity } from '../../types';
import { Layers, ChevronDown, ChevronUp } from 'lucide-react';

interface DetectionMiniCardProps {
  tubes: TubeEntity[];
  selectedTubeId?: string | null;
  onSelectTube?: (id: string | null) => void;
  isRunning?: boolean;
  className?: string;
}

export const DetectionMiniCard: React.FC<DetectionMiniCardProps> = ({
  tubes,
  selectedTubeId,
  onSelectTube,
  isRunning = false,
  className = '',
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // If there are no detections and inference is not running, hide the card
  if (tubes.length === 0 && !isRunning) {
    return null;
  }

  const headsCount = tubes.filter((t) => t.role === 'HEAD').length;
  const tailsCount = tubes.filter((t) => t.role === 'TAIL').length;

  const formatWidth = (d: TubeEntity): string => {
    if (typeof d.width_mm === 'number' && d.width_mm > 0) {
      return `${d.width_mm.toFixed(1)}mm`;
    }
    if (typeof d.width_px === 'number' && d.width_px > 0) {
      return `${d.width_px.toFixed(1)}px`;
    }
    return 'null';
  };

  return (
    <div
      className={`pointer-events-auto w-fit min-w-[155px] max-w-[185px] rounded-xl bg-[var(--bg-card)]/95 dark:bg-slate-900/90 backdrop-blur-md border border-[var(--border-color)] dark:border-slate-700/80 shadow-xl overflow-hidden select-none transition-all duration-200 animate-in fade-in slide-in-from-bottom-2 ${className}`}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header Bar with Counts */}
      <div
        className="flex items-center justify-between gap-1.5 px-2.5 py-1.5 border-b border-[var(--border-color)]/70 dark:border-slate-800/80 bg-[var(--bg-card)]/60 cursor-pointer select-none"
        onClick={() => setIsCollapsed(!isCollapsed)}
        title={isCollapsed ? 'Expand detection card' : 'Collapse detection card'}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <Layers className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-secondary)] font-mono whitespace-nowrap">
            ENDS ({tubes.length})
          </span>
        </div>

        <div className="flex items-center gap-1 text-[9.5px] font-mono shrink-0">
          <span className="inline-flex items-center px-1 py-0.2 rounded bg-sky-500/15 text-sky-400 font-bold">
            {headsCount}H
          </span>
          <span className="text-[var(--border-color)]">•</span>
          <span className="inline-flex items-center px-1 py-0.2 rounded bg-indigo-500/15 text-indigo-400 font-bold">
            {tailsCount}T
          </span>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setIsCollapsed(!isCollapsed);
            }}
            className="ml-0.5 p-0.5 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--border-color)]/50 transition-colors"
            aria-label={isCollapsed ? 'Expand' : 'Collapse'}
          >
            {isCollapsed ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Body: Compact List of detected tube ends */}
      {!isCollapsed && (
        <div className="p-1 max-h-44 overflow-y-auto invisible-scrollbar flex flex-col gap-1">
          {tubes.length === 0 ? (
            <div className="px-2 py-1.5 text-[10px] font-mono text-[var(--text-secondary)] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse shrink-0" />
              <span>Scanning...</span>
            </div>
          ) : (
            tubes.map((d, idx) => {
              const isHead = d.role === 'HEAD';
              const isSelected = selectedTubeId != null && String(d.id) === String(selectedTubeId);
              const widthStr = formatWidth(d);
              const hasWidth = widthStr !== 'null';

              return (
                <div
                  key={`mini-det-${d.id ?? idx}`}
                  onClick={() => onSelectTube?.(isSelected ? null : String(d.id))}
                  role="button"
                  tabIndex={0}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-lg font-mono text-[11px] border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-cyan-500/20 border-cyan-400/90 text-[var(--text-primary)] shadow-xs ring-1 ring-cyan-400/50'
                      : 'bg-[var(--bg-card)]/60 hover:bg-[var(--bg-card)] border-[var(--border-color)]/70 text-[var(--text-primary)] hover:border-cyan-500/50'
                  }`}
                >
                  {/* ID displayed directly first (no serial number) */}
                  <span className="text-[10px] font-bold text-[var(--text-secondary)] font-mono shrink-0">
                    ID:{d.displayId ?? d.id}
                  </span>

                  {/* Role badge */}
                  <span
                    className={`px-1.5 py-0.2 rounded text-[9px] font-black uppercase tracking-wider leading-tight shrink-0 ${
                      isHead ? 'bg-sky-500 text-white shadow-2xs' : 'bg-indigo-500 text-white shadow-2xs'
                    }`}
                  >
                    {d.role || 'END'}
                  </span>

                  {/* Width minimized right next to it with size tag */}
                  <div className="flex items-center gap-1 ml-auto shrink-0 text-[10px] font-mono">
                    {d.size === 'BIGGER' && (
                      <span className="px-1 py-0.2 rounded text-[7.5px] font-black uppercase tracking-wider bg-rose-500/20 text-rose-400 border border-rose-500/30">
                        BIG
                      </span>
                    )}
                    {d.size === 'SMALLER' && (
                      <span className="px-1 py-0.2 rounded text-[7.5px] font-black uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/30">
                        SML
                      </span>
                    )}
                    {d.size === 'SAME' && (
                      <span className="px-1 py-0.2 rounded text-[7.5px] font-black uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        EQ
                      </span>
                    )}
                    <span className="text-[var(--text-secondary)] opacity-65">w:</span>
                    <strong
                      className={`font-mono ${
                        hasWidth
                          ? 'text-emerald-500 dark:text-emerald-400 font-bold'
                          : 'text-[var(--text-secondary)] opacity-60 font-normal'
                      }`}
                    >
                      {widthStr}
                    </strong>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
};
