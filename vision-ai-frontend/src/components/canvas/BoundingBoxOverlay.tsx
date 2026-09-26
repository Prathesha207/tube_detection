import React, { useState, useMemo } from 'react';
import type { TubeEntity, LabelMode } from '../../types';

interface BoundingBoxOverlayProps {
  tubes: TubeEntity[];
  selectedTubeId: string | null;
  onSelectTube: (id: string | null) => void;
  labelMode?: LabelMode;
}


interface BadgePlacement {
  side: 'left' | 'right' | 'top' | 'bottom';
  containerStyle: React.CSSProperties;
  stemStyle: React.CSSProperties;
  zIndex: number;
}

function createPlacement(
  side: 'left' | 'right' | 'top' | 'bottom',
  role: string,
  offsetPx: number = 0,
  stemLengthPx: number = 18
): BadgePlacement {
  const isHead = role === 'HEAD';
  const stemColor = isHead ? 'rgba(6, 182, 212, 1)' : 'rgba(245, 158, 11, 1)';
  const glow = isHead ? '0 0 4px rgba(6, 182, 212, 0.7)' : '0 0 4px rgba(245, 158, 11, 0.7)';

  // Strictly clamp stem length between 14px and 22px to prevent huge stems or hiding behind toolbar
  const safeStem = Math.min(22, Math.max(14, stemLengthPx));

  if (side === 'left') {
    return {
      side: 'left',
      containerStyle: {
        position: 'absolute',
        right: `calc(100% + ${safeStem}px)`,
        top: '50%',
        transform: `translateY(calc(-50% + ${offsetPx}px))`,
      },
      stemStyle: {
        position: 'absolute',
        right: `-${safeStem}px`,
        top: '50%',
        transform: `translateY(calc(-50% - ${offsetPx}px))`,
        width: `${safeStem}px`,
        height: '2px',
        backgroundColor: stemColor,
        boxShadow: glow,
      },
      zIndex: 40,
    };
  }

  if (side === 'right') {
    return {
      side: 'right',
      containerStyle: {
        position: 'absolute',
        left: `calc(100% + ${safeStem}px)`,
        top: '50%',
        transform: `translateY(calc(-50% + ${offsetPx}px))`,
      },
      stemStyle: {
        position: 'absolute',
        left: `-${safeStem}px`,
        top: '50%',
        transform: `translateY(calc(-50% - ${offsetPx}px))`,
        width: `${safeStem}px`,
        height: '2px',
        backgroundColor: stemColor,
        boxShadow: glow,
      },
      zIndex: 40,
    };
  }

  if (side === 'top') {
    return {
      side: 'top',
      containerStyle: {
        position: 'absolute',
        bottom: `calc(100% + ${safeStem}px)`,
        left: '50%',
        transform: `translateX(calc(-50% + ${offsetPx}px))`,
      },
      stemStyle: {
        position: 'absolute',
        bottom: `-${safeStem}px`,
        left: '50%',
        transform: `translateX(calc(-50% - ${offsetPx}px))`,
        width: '2px',
        height: `${safeStem}px`,
        backgroundColor: stemColor,
        boxShadow: glow,
      },
      zIndex: 40,
    };
  }

  // side === 'bottom'
  return {
    side: 'bottom',
    containerStyle: {
      position: 'absolute',
      top: `calc(100% + ${safeStem}px)`,
      left: '50%',
      transform: `translateX(calc(-50% + ${offsetPx}px))`,
    },
    stemStyle: {
      position: 'absolute',
      top: `-${safeStem}px`,
      left: '50%',
      transform: `translateX(calc(-50% - ${offsetPx}px))`,
      width: '2px',
      height: `${safeStem}px`,
      backgroundColor: stemColor,
      boxShadow: glow,
    },
    zIndex: 40,
  };
}

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  tubes,
  selectedTubeId,
  onSelectTube,
  labelMode = 'compact',
}) => {
  const [hoveredTubeId, setHoveredTubeId] = useState<string | null>(null);

  const validTubes = useMemo(() => {
    return tubes.filter((tube) => {
      return (
        Number.isFinite(tube.x) &&
        Number.isFinite(tube.y) &&
        Number.isFinite(tube.width) &&
        Number.isFinite(tube.height) &&
        tube.width > 0 &&
        tube.height > 0
      );
    });
  }, [tubes]);

  // Radial Outward Quadrant Callout Algorithm:
  // 1. Calculates the mean cluster center (avgCx, avgCy).
  // 2. Each tube projects directly OUTWARD into its nearest open quadrant (North, South, East, West).
  // 3. Tubes on the left NEVER shoot across to the right; tubes on the right NEVER shoot across to the left!
  // 4. Stems are strictly compact (16px - 20px) so labels NEVER shoot across the screen or hide behind toolbars!
  // 5. If two tubes share a sector, they are staggered with horizontal/vertical offsets so labels never collide.
  const placementMap = useMemo(() => {
    const result: Record<string, BadgePlacement> = {};
    if (validTubes.length === 0) return result;

    // 1. Calculate cluster center
    const centers = validTubes.map((t) => ({
      tube: t,
      cx: t.x + t.width / 2,
      cy: t.y + t.height / 2,
    }));

    const avgCx = centers.reduce((sum, c) => sum + c.cx, 0) / centers.length;
    const avgCy = centers.reduce((sum, c) => sum + c.cy, 0) / centers.length;

    // 2. Calculate outward vector & polar angle from center
    const withAngles = centers.map((item) => {
      const dx = item.cx - avgCx;
      const dy = item.cy - avgCy;
      const angle = Math.atan2(dy, dx);
      return { ...item, dx, dy, angle };
    });

    // Walk around the cluster perimeter
    withAngles.sort((a, b) => a.angle - b.angle);

    const sideCounts: Record<'left' | 'right' | 'top' | 'bottom', number> = {
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
    };

    withAngles.forEach((item) => {
      const { tube, dx, dy, angle } = item;
      let side: 'left' | 'right' | 'top' | 'bottom';

      // Natural outward compass direction based on polar sector:
      // North: -135° to -45°
      // East:  -45° to +45°
      // South: +45° to +135°
      // West:  <-135° or >+135°
      if (angle >= -Math.PI * 0.75 && angle < -Math.PI * 0.25) {
        side = 'top';
      } else if (angle >= -Math.PI * 0.25 && angle < Math.PI * 0.25) {
        side = 'right';
      } else if (angle >= Math.PI * 0.25 && angle < Math.PI * 0.75) {
        side = 'bottom';
      } else {
        side = 'left';
      }

      // If the natural side is already occupied, try the secondary outward direction
      if (sideCounts[side] > 0) {
        if (side === 'top' && dx < 0 && sideCounts['left'] === 0) side = 'left';
        else if (side === 'top' && dx >= 0 && sideCounts['right'] === 0) side = 'right';
        else if (side === 'bottom' && dx < 0 && sideCounts['left'] === 0) side = 'left';
        else if (side === 'bottom' && dx >= 0 && sideCounts['right'] === 0) side = 'right';
        else if (side === 'left' && dy < 0 && sideCounts['top'] === 0) side = 'top';
        else if (side === 'left' && dy >= 0 && sideCounts['bottom'] === 0) side = 'bottom';
        else if (side === 'right' && dy < 0 && sideCounts['top'] === 0) side = 'top';
        else if (side === 'right' && dy >= 0 && sideCounts['bottom'] === 0) side = 'bottom';
      }

      // Canvas boundary safety guard
      if (tube.y < 8 && side === 'top') side = 'bottom';
      if (tube.y > 88 && side === 'bottom') side = 'top';
      if (tube.x < 10 && side === 'left') side = 'right';
      if (tube.x > 88 && side === 'right') side = 'left';

      const count = sideCounts[side];
      sideCounts[side] += 1;

      // If multiple badges share the same side, stagger them so they never overlap
      const offsetPx = count === 0 ? 0 : (count % 2 === 1 ? -42 * count : 42 * count);
      const stemLen = count === 0 ? 18 : 22;

      result[tube.id] = createPlacement(side, tube.role, offsetPx, stemLen);
    });

    return result;
  }, [validTubes]);

  return (
    <div className="absolute inset-0 w-full h-full pointer-events-none z-20">
      {/* High-Visibility Bounding Boxes */}
      {validTubes.map((tube, idx) => {
        const isSelected = tube.id === selectedTubeId;
        const isHovered = tube.id === hoveredTubeId;
        const isHead = tube.role === 'HEAD';
        const placement = placementMap[tube.id] || createPlacement('left', tube.role);

        // Vibrant high-contrast styling:
        // HEAD = Electric Cyan (High visibility on dark connectors)
        // TAIL = Electric Amber/Gold (High visibility on clear tubes & white background)
        const borderColor = isHead
          ? (isSelected ? 'border-cyan-300' : 'border-cyan-400')
          : (isSelected ? 'border-amber-300' : 'border-amber-400');

        const glowShadow = isSelected
          ? (isHead ? 'shadow-[0_0_14px_rgba(6,182,212,0.8)]' : 'shadow-[0_0_14px_rgba(245,158,11,0.8)]')
          : isHovered
            ? (isHead ? 'shadow-[0_0_10px_rgba(6,182,212,0.6)]' : 'shadow-[0_0_10px_rgba(245,158,11,0.6)]')
            : (isHead ? 'shadow-[0_0_6px_rgba(6,182,212,0.4)]' : 'shadow-[0_0_6px_rgba(245,158,11,0.4)]');

        const bracketColor = isHead ? 'border-cyan-200' : 'border-amber-200';
        const effectiveZIndex = isSelected || isHovered ? 50 : placement.zIndex;
        const isCoasting = tube.status === 'COASTED' || Boolean(tube.is_coasting);

        return (
          <div
            key={`bbox-${idx}-${tube.id}`}
            role="button"
            tabIndex={0}
            aria-label={`Select tube end ${tube.id}`}
            onClick={(e) => {
              e.stopPropagation();
              onSelectTube(tube.id === selectedTubeId ? null : tube.id);
            }}
            onMouseEnter={() => setHoveredTubeId(tube.id)}
            onMouseLeave={() => setHoveredTubeId(null)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.stopPropagation();
                e.preventDefault();
                onSelectTube(tube.id === selectedTubeId ? null : tube.id);
              }
            }}
            style={{
              left: `${tube.x}%`,
              top: `${tube.y}%`,
              width: `${tube.width}%`,
              height: `${tube.height}%`,
              zIndex: effectiveZIndex,
            }}
            className={`absolute border-2 rounded-sm pointer-events-auto cursor-pointer transition-all duration-150 ${borderColor} ${glowShadow} bg-transparent hover:bg-white/5 ${
              isCoasting ? 'border-dashed opacity-70' : 'border-solid'
            }`}
          >
            {/* Corner Crosshair Brackets (High contrast for pinpoint accuracy) */}
            <span className={`absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 ${bracketColor}`} />
            <span className={`absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 ${bracketColor}`} />
            <span className={`absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 ${bracketColor}`} />
            <span className={`absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 ${bracketColor}`} />

            {/* Subtle center reticle dot */}
            <span
              className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-1 rounded-full ${
                isHead ? 'bg-cyan-300/80 shadow-[0_0_4px_cyan]' : 'bg-amber-300/80 shadow-[0_0_4px_orange]'
              }`}
            />

            {/* 3. PINS MODE: Minimalist, non-intrusive numbered circle pins */}
            {labelMode === 'pins' && (
              <div
                className={`absolute -top-2.5 -left-2.5 w-5 h-5 rounded-full flex items-center justify-center font-mono font-black text-[10px] shadow-md border pointer-events-none transition-transform ${
                  isHead
                    ? 'bg-cyan-500 text-slate-950 border-cyan-200'
                    : 'bg-amber-500 text-slate-950 border-amber-200'
                } ${isHovered || isSelected ? 'scale-125 z-50' : ''}`}
              >
                {tube.displayId ?? (typeof tube.id === 'string' && tube.id.includes('-') ? tube.id.split('-')[1] : tube.id)}
              </div>
            )}

            {/* 4. QUADRANT COMPACT MODE: Badges placed in North, South, East, West with guide stems */}
            {labelMode === 'compact' && (
              <div
                style={placement.containerStyle}
                className={`px-1.5 py-0.5 rounded-md text-[9.5px] font-mono whitespace-nowrap flex items-center gap-1.5 backdrop-blur-md shadow-md pointer-events-none border transition-all ${
                  isHead
                    ? 'bg-slate-950/90 text-cyan-200 border-cyan-500/70'
                    : 'bg-slate-950/90 text-amber-200 border-amber-500/70'
                } ${isHovered || isSelected ? 'scale-105 border-white text-white z-50 shadow-lg' : ''}`}
              >
                {/* Connecting guide line straight to the box edge */}
                <span style={placement.stemStyle} />

                <span className="font-bold flex items-center gap-1">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isHead ? 'bg-cyan-400' : 'bg-amber-400'
                    }`}
                  />
                  <span>#{tube.displayId ?? (typeof tube.id === 'string' && tube.id.includes('-') ? tube.id.split('-')[1] : tube.id)}</span>
                  <span className={isHead ? 'text-cyan-300 font-extrabold' : 'text-amber-300 font-extrabold'}>
                    {tube.role}
                  </span>
                  {tube.width_px != null && (
                    <span className="text-white font-mono opacity-90">
                      {tube.width_px.toFixed(0)}px
                    </span>
                  )}
                  {tube.pair_id != null && (
                    <span className="text-emerald-400 font-bold">
                      T{tube.pair_id}
                    </span>
                  )}
                </span>
              </div>
            )}

            {/* 5. HOVER TOOLTIP: Rich popover when hovering or clicking a tube */}
            {(isHovered || isSelected) && (
              <div
                className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 p-2 rounded-xl bg-slate-950/95 backdrop-blur-lg border border-white/20 text-white shadow-xl z-50 pointer-events-none min-w-[130px] flex flex-col gap-1 text-[10px] font-mono animate-in fade-in zoom-in-95 duration-100"
              >
                <div className="flex items-center justify-between pb-1 border-b border-white/10 font-bold">
                  <span className="flex items-center gap-1">
                    <span className={`w-2 h-2 rounded-full ${isHead ? 'bg-cyan-400' : 'bg-amber-400'}`} />
                    <span>Tube End #{tube.displayId ?? (typeof tube.id === 'string' && tube.id.includes('-') ? tube.id.split('-')[1] : tube.id)}</span>
                  </span>
                  <span className={isHead ? 'text-cyan-300' : 'text-amber-300'}>
                    {tube.role}
                  </span>
                </div>

                <div className="flex items-center justify-between text-slate-300">
                  <span>Width:</span>
                  <span className="font-bold text-white">
                    {tube.width_px ? `${tube.width_px.toFixed(1)}px` : '-'}
                    {tube.width_mm ? ` (${tube.width_mm.toFixed(2)}mm)` : ''}
                  </span>
                </div>

                <div className="flex items-center justify-between text-slate-300">
                  <span>Pair:</span>
                  <span className={tube.pair_id != null ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
                    {tube.pair_id != null ? `Tube #${tube.pair_id}` : 'Unpaired'}
                  </span>
                </div>

                {tube.size && (
                  <div className="flex items-center justify-between text-slate-300">
                    <span>Size:</span>
                    <span className={`font-bold ${tube.size === 'BIGGER' ? 'text-rose-400' : tube.size === 'SMALLER' ? 'text-yellow-400' : 'text-emerald-400'}`}>
                      {tube.size}
                    </span>
                  </div>
                )}

                {tube.status && tube.status !== 'OK' && (
                  <div className="mt-0.5 pt-0.5 border-t border-amber-500/30 text-[9px] text-amber-300">
                    {tube.status}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
