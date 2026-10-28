import React from 'react';

export interface DepthTelemetry {
  depth_width?: number;
  depth_height?: number;
  valid_pct?: number;
  center_z_mm?: number | null;
  colormap_min_mm?: number;
  colormap_max_mm?: number;
  aligned_to_rgb?: boolean;
}

export interface DepthHoverPoint {
  x_px: number;
  y_px: number;
  z_mm: number | null;
  valid_ratio?: number;
}

interface DepthHudProps {
  telemetry: DepthTelemetry | null;
  hoverPoint: DepthHoverPoint | null;
  isHovering: boolean;
  className?: string;
}

export const DepthHud: React.FC<DepthHudProps> = ({
  telemetry,
  hoverPoint,
  isHovering,
  className = '',
}) => {
  // Center distance formatted
  const centerText =
    telemetry?.center_z_mm !== null && telemetry?.center_z_mm !== undefined
      ? `${telemetry.center_z_mm.toFixed(0)} mm`
      : 'no depth';

  // Valid percentage formatted
  const validText =
    typeof telemetry?.valid_pct === 'number'
      ? `${telemetry.valid_pct.toFixed(0)}%`
      : '—%';

  // Colormap range formatted
  const colormapText = `${telemetry?.colormap_min_mm ?? 300}-${telemetry?.colormap_max_mm ?? 1500} mm`;

  return (
    <div
      className={`pointer-events-auto flex flex-col items-start gap-1.5 select-none animate-in fade-in slide-in-from-bottom-2 duration-200 ${className}`}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Row 1: Hover Probe Box (Matches the grey tip badge requested by the operator) */}
      <div className="flex items-center justify-center min-w-[76px] h-[28px] px-3 py-1 rounded-lg bg-zinc-700/90 dark:bg-zinc-800/95 backdrop-blur-md border border-zinc-500/60 dark:border-zinc-600/70 shadow-lg font-mono text-xs transition-all duration-150">
        {isHovering && hoverPoint ? (
          <div className="flex items-center gap-2 whitespace-nowrap">
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                hoverPoint.z_mm !== null ? 'bg-cyan-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            {hoverPoint.z_mm !== null ? (
              <>
                <span className="font-bold text-white text-[12px] tracking-tight">
                  {hoverPoint.z_mm.toFixed(0)} mm
                </span>
                <span className="text-cyan-300 text-[11px] font-medium">
                  ({(hoverPoint.z_mm / 10).toFixed(1)} cm)
                </span>
              </>
            ) : (
              <span className="text-amber-300 font-semibold text-[11.5px]">
                — (no depth)
              </span>
            )}
            <span className="text-[10px] text-zinc-300 dark:text-zinc-400 border-l border-zinc-500/60 pl-2">
              X: {hoverPoint.x_px} Y: {hoverPoint.y_px}
            </span>
          </div>
        ) : (
          <span className="text-white/90 font-bold text-[14px] leading-none tracking-widest px-2">
            —
          </span>
        )}
      </div>

      {/* Row 2: Stream Telemetry Numbers (Center, Valid %, Colormap range — Always visible, never hidden) */}
      <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-black/80 backdrop-blur-md border border-white/10 text-[10.5px] font-mono text-zinc-300 shadow-md">
        <div className="flex items-center gap-1">
          <span className="text-zinc-400">center</span>
          <span
            className={`font-semibold ${
              telemetry?.center_z_mm !== null && telemetry?.center_z_mm !== undefined
                ? 'text-cyan-300'
                : 'text-amber-400'
            }`}
          >
            {centerText}
          </span>
        </div>

        <span className="text-white/20">|</span>

        <div className="flex items-center gap-1">
          <span className="text-zinc-400">valid</span>
          <span className="font-semibold text-emerald-400">{validText}</span>
        </div>

        <span className="text-white/20">|</span>

        <div className="flex items-center gap-1">
          <span className="text-zinc-400">colormap</span>
          <span className="font-semibold text-slate-200">{colormapText}</span>
        </div>
      </div>
    </div>
  );
};
