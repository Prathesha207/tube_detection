import React from 'react';
import { MousePointer2, Square, Hexagon, Trash2, Save, Check } from 'lucide-react';

export type RoiToolMode = 'pointer' | 'rectangle' | 'polygon';

interface RoiLeftToolbarProps {
  activeTool: RoiToolMode;
  onSelectTool: (tool: RoiToolMode) => void;
  onClear: () => void;
  onSave: () => void;
  isSaving?: boolean;
  hasChanges?: boolean;
  hasRoi?: boolean;
}

export const RoiLeftToolbar: React.FC<RoiLeftToolbarProps> = ({
  activeTool,
  onSelectTool,
  onClear,
  onSave,
  isSaving = false,
  hasChanges = false,
  hasRoi = false,
}) => {
  return (
    <div
      className="absolute left-3.5 sm:left-4 top-1/2 -translate-y-1/2 z-30 flex flex-col items-center gap-1.5 p-1.5 rounded-2xl bg-[var(--bg-card)]/90 backdrop-blur-md border border-[var(--border-color)] shadow-2xl pointer-events-auto select-none transition-all duration-200 animate-in fade-in zoom-in-95"
      role="toolbar"
      aria-label="ROI Drawing Tools"
    >
      {/* 1. Pointer / Select & Adjust Tool */}
      <button
        onClick={() => onSelectTool('pointer')}
        title="Select & Move Tool (V) — Click & Drag Vertices"
        className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all cursor-pointer ${
          activeTool === 'pointer'
            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-[0_0_12px_rgba(6,182,212,0.4)] scale-105'
            : 'text-[var(--text-secondary)] hover:text-white hover:bg-[var(--btn-secondary-hover)]'
        }`}
      >
        <MousePointer2 className="w-4 h-4" />
      </button>

      {/* 2. Rectangle Tool */}
      <button
        onClick={() => onSelectTool('rectangle')}
        title="Rectangle Tool (R) — Click & Drag Diagonal"
        className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all cursor-pointer ${
          activeTool === 'rectangle'
            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-[0_0_12px_rgba(6,182,212,0.4)] scale-105'
            : 'text-[var(--text-secondary)] hover:text-white hover:bg-[var(--btn-secondary-hover)]'
        }`}
      >
        <Square className="w-4 h-4" />
      </button>

      {/* 3. Polygon Tool */}
      <button
        onClick={() => onSelectTool('polygon')}
        title="Polygon Tool (P) — Click Sequential Vertices, Double-Click to Close"
        className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all cursor-pointer ${
          activeTool === 'polygon'
            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 shadow-[0_0_12px_rgba(6,182,212,0.4)] scale-105'
            : 'text-[var(--text-secondary)] hover:text-white hover:bg-[var(--btn-secondary-hover)]'
        }`}
      >
        <Hexagon className="w-4 h-4" />
      </button>

      {/* Divider */}
      <div className="w-5 h-[1px] bg-[var(--border-color)] my-1 shrink-0" />

      {/* 4. Clear Tool */}
      <button
        onClick={onClear}
        disabled={!hasRoi}
        title={hasRoi ? 'Clear ROI Mask (Reset to Full Frame)' : 'No ROI to clear'}
        className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
          hasRoi
            ? 'text-rose-400 hover:text-rose-200 hover:bg-rose-500/20 active:scale-95 cursor-pointer'
            : 'text-zinc-600 cursor-not-allowed opacity-40'
        }`}
      >
        <Trash2 className="w-4 h-4" />
      </button>

      {/* 5. Save Tool */}
      <button
        onClick={onSave}
        disabled={isSaving}
        title="Save ROI & Hot-Swap Live Inference"
        className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all cursor-pointer ${
          hasChanges
            ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 shadow-[0_0_12px_rgba(16,185,129,0.4)] animate-pulse'
            : 'text-emerald-400 hover:text-emerald-200 hover:bg-emerald-500/20'
        } ${isSaving ? 'cursor-wait opacity-60' : 'active:scale-95'}`}
      >
        {isSaving ? <Check className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
      </button>
    </div>
  );
};
