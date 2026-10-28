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
      {/* Row 1: Hover Probe Box (Uses active theme background & border colors) */}
      <div className="flex items-center justify-center min-w-[76px] h-[28px] px-3 py-1 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-md font-mono text-xs transition-all duration-150">
        {isHovering && hoverPoint ? (
          <div className="flex items-center gap-2 whitespace-nowrap">
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                hoverPoint.z_mm !== null
                  ? 'bg-[var(--accent-pond)] animate-pulse shadow-[0_0_6px_var(--accent-pond)]'
                  : 'bg-amber-500'
              }`}
            />
            {hoverPoint.z_mm !== null ? (
              <>
                <span className="font-bold text-[var(--text-primary)] text-[12px] tracking-tight">
                  {hoverPoint.z_mm.toFixed(0)} mm
                </span>
                <span className="text-[var(--accent-pond)] text-[11px] font-semibold">
                  ({(hoverPoint.z_mm / 10).toFixed(1)} cm)
                </span>
              </>
            ) : (
              <span className="text-amber-600 dark:text-amber-400 font-semibold text-[11.5px]">
                — (no depth)
              </span>
            )}
            <span className="text-[10px] text-[var(--text-secondary)] border-l border-[var(--border-color)] pl-2">
              X: {hoverPoint.x_px} Y: {hoverPoint.y_px}
            </span>
          </div>
        ) : (
          <span className="text-[var(--text-secondary)] font-bold text-[14px] leading-none tracking-widest px-2">
            —
          </span>
        )}
      </div>

      {/* Row 2: Stream Telemetry Numbers (Uses active theme background, border & text) */}
      <div className="flex items-center gap-2 px-2.5 py-1 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] text-[10.5px] font-mono shadow-md">
        <div className="flex items-center gap-1">
          <span className="text-[var(--text-secondary)]">center</span>
          <span
            className={`font-semibold ${
              telemetry?.center_z_mm !== null && telemetry?.center_z_mm !== undefined
                ? 'text-[var(--accent-pond)] font-bold'
                : 'text-amber-600 dark:text-amber-400'
            }`}
          >
            {centerText}
          </span>
        </div>

        <span className="text-[var(--border-color)]">|</span>

        <div className="flex items-center gap-1">
          <span className="text-[var(--text-secondary)]">valid</span>
          <span className="font-semibold text-emerald-600 dark:text-emerald-400">{validText}</span>
        </div>

        <span className="text-[var(--border-color)]">|</span>

        <div className="flex items-center gap-1">
          <span className="text-[var(--text-secondary)]">colormap</span>
          <span className="font-semibold text-[var(--text-primary)]">{colormapText}</span>
        </div>
      </div>
    </div>
  );
};
