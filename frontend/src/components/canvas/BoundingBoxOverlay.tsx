import React, { useState, useMemo } from 'react';
import type { TubeEntity, TubeMask, TubeWrapper, TubeTrace, LabelMode } from '../../types';

interface BoundingBoxOverlayProps {
  tubes: TubeEntity[];
  selectedTubeId: string | null;
  onSelectTube: (id: string | null) => void;
  labelMode?: LabelMode;
  boxesOnly?: boolean;
  tubeMasks?: TubeMask[];
  wrapper?: TubeWrapper | null;
  traces?: TubeTrace[] | null;
  frameWidth?: number;
  frameHeight?: number;
}


interface BadgePlacement {
  side: 'left' | 'right' | 'top' | 'bottom';
  containerStyle: React.CSSProperties;
  stemStyle: React.CSSProperties;
  zIndex: number;
}

function createPlacement(
  side: 'left' | 'right' | 'top' | 'bottom',
  tubeColor?: string | null,
  offsetPx: number = 0,
  stemLengthPx: number = 18
): BadgePlacement {
  const isBlue = String(tubeColor || '').toLowerCase() === 'blue';
  const stemColor = isBlue ? 'rgba(56, 189, 248, 1)' : 'rgba(255, 255, 255, 0.95)';
  const glow = isBlue ? '0 0 4px rgba(56, 189, 248, 0.7)' : '0 0 4px rgba(255, 255, 255, 0.7)';

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
  labelMode = 'hidden',
  boxesOnly = false,
  tubeMasks = [],
  wrapper,
  traces,
  frameWidth = 0,
  frameHeight = 0,
}) => {
  const [hoveredTubeId, setHoveredTubeId] = useState<string | null>(null);

  const validTubes = useMemo(() => {
    return tubes.filter((tube) => {
      return (
        !(tube as any).held &&
        tube.status !== 'COASTED' &&
        !tube.is_coasting &&
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

    // 1. Anchor outward placement relative to frame center (50%, 50%)
    // This guarantees label quadrant placement is completely stable and
    // never jumps sides when another tube appears or disappears!
    const centers = validTubes.map((t) => ({
      tube: t,
      cx: t.x + t.width / 2,
      cy: t.y + t.height / 2,
    }));

    const anchorCx = 50;
    const anchorCy = 50;

    // 2. Calculate outward vector & polar angle from frame center
    const withAngles = centers.map((item) => {
      const dx = item.cx - anchorCx;
      const dy = item.cy - anchorCy;
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

      result[tube.id] = createPlacement(side, tube.tube_color, offsetPx, stemLen);
    });

    return result;
  }, [validTubes]);

  // Stable multi-polygon SVG path compilation for tube segmentation masks.
  // Grouping by tube color into two permanent layers prevents React key recycling
  // and eliminates color-flipping flickers between frames.
  const { whiteTubePath, blueTubePath } = useMemo(() => {
    const whitePolys: number[][][] = [];
    const bluePolys: number[][][] = [];

    for (const mask of tubeMasks) {
      if ((mask as any).held) continue; // Never render ghost/held masks
      const color = String(mask.tube_color || '').toLowerCase();
      if (color !== 'blue' && color !== 'white') continue;
      const polys = (mask.polygons && mask.polygons.length > 0)
        ? mask.polygons
        : (mask.polygon && mask.polygon.length >= 3 ? [mask.polygon] : []);

      for (const p of polys) {
        if (p && p.length >= 3) {
          if (color === 'blue') {
            bluePolys.push(p);
          } else {
            whitePolys.push(p);
          }
        }
      }
    }

    const toPathD = (polys: number[][][]): string => {
      if (!polys.length) return '';
      return polys
        .map((poly) => {
          const start = poly[0];
          const lines = poly.slice(1).map(([x, y]) => `L ${x} ${y}`).join(' ');
          return `M ${start[0]} ${start[1]} ${lines} Z`;
        })
        .join(' ');
    };

    return {
      whiteTubePath: toPathD(whitePolys),
      blueTubePath: toPathD(bluePolys),
    };
  }, [tubeMasks]);

  return (
    <div className="absolute inset-0 w-full h-full pointer-events-none z-20">
      {frameWidth > 0 && frameHeight > 0 && (
        <svg
          className="absolute inset-0 w-full h-full overflow-visible"
          viewBox={`0 0 ${frameWidth} ${frameHeight}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {/* WHITE TUBES: Neat clean outline + subtle translucent fill */}
          {whiteTubePath && (
            <g key="layer-white-tubes">
              <path
                d={whiteTubePath}
                fill="rgba(255, 255, 255, 0.12)"
                stroke="#f8fafc"
                strokeOpacity={0.95}
                strokeWidth={1.2}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )}

          {/* BLUE TUBES: Neat clean outline + subtle translucent fill */}
          {blueTubePath && (
            <g key="layer-blue-tubes">
              <path
                d={blueTubePath}
                fill="rgba(56, 189, 248, 0.12)"
                stroke="#38bdf8"
                strokeOpacity={0.95}
                strokeWidth={1.2}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          )}
          {/* WRAPPER BAND: Distinct sleek polygon overlay when detected */}
          {wrapper?.polygon && wrapper.polygon.length >= 3 && (
            <g key="layer-wrapper">
              <polygon
                points={wrapper.polygon.map(([x, y]: [number, number]) => `${x},${y}`).join(' ')}
                fill="rgba(168, 85, 247, 0.15)"
                stroke="#c084fc"
                strokeWidth={1.2}
                strokeDasharray="4 3"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
              {wrapper.center && (
                <text
                  x={wrapper.center[0]}
                  y={wrapper.center[1]}
                  fill="#f3e8ff"
                  fontSize={Math.max(9, frameWidth / 160)}
                  fontWeight="bold"
                  fontFamily="monospace"
                  textAnchor="middle"
                  dominantBaseline="central"
                  style={{ textShadow: '0 1px 3px rgba(0,0,0,0.9)' }}
                >
                  WRAPPER
                </text>
              )}
            </g>
          )}

          {/* TRACE PATHS: Ultra-thin, precise centerline traces out of wrapper */}
          {traces && traces.length > 0 && (
            <g key="layer-traces">
              {traces.map((tr: TubeTrace, trIdx: number) => {
                if (!tr.path || tr.path.length < 2) return null;
                const pathD = tr.path.reduce((acc: string, pt: [number, number], i: number) => `${acc} ${i === 0 ? 'M' : 'L'} ${pt[0]} ${pt[1]}`, '');
                const isBlue = String(tr.tube_color || '').toLowerCase() === 'blue';
                const traceColor = isBlue ? '#38bdf8' : '#f8fafc';
                const endPt = tr.path[tr.path.length - 1];
                return (
                  <g key={`trace-${tr.tube_id ?? 'untracked'}-${trIdx}`}>
                    <path
                      d={pathD}
                      fill="none"
                      stroke="rgba(0, 0, 0, 0.55)"
                      strokeWidth={1.8}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      vectorEffect="non-scaling-stroke"
                    />
                    <path
                      d={pathD}
                      fill="none"
                      stroke={traceColor}
                      strokeWidth={1.1}
                      strokeDasharray="3 3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      vectorEffect="non-scaling-stroke"
                    />
                    {endPt && (
                      <circle
                        cx={endPt[0]}
                        cy={endPt[1]}
                        r={2.2}
                        fill={traceColor}
                        stroke="#0f172a"
                        strokeWidth="1"
                      />
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {tubes.map((end) => {
            if ((end as any).held || !end.polygon || end.polygon.length < 3) return null;
            const isBlue = String(end.tube_color || '').toLowerCase() === 'blue';
            // Blue tube = Sky Blue; White tube = Crisp White
            const colorStroke = isBlue ? '#38bdf8' : '#f8fafc';
            const colorFill = isBlue ? 'rgba(56, 189, 248, 0.14)' : 'rgba(255, 255, 255, 0.14)';
            return (
              <polygon
                key={`end-mask-${end.id}`}
                points={end.polygon.map(([x, y]: [number, number]) => `${x},${y}`).join(' ')}
                fill={colorFill}
                stroke={colorStroke}
                strokeWidth={1.0}
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      )}
      {/* High-Visibility Bounding Boxes */}
      {validTubes.map((tube, idx) => {
        const isSelected = tube.id === selectedTubeId;
        const isHovered = tube.id === hoveredTubeId;
        const isHead = tube.role === 'HEAD';
        const placement = placementMap[tube.id] || createPlacement('left', tube.tube_color);

        // Color coding: Blue tube = Sky Blue; White tube = Crisp White
        const isBlue = String(tube.tube_color || '').toLowerCase() === 'blue';
        const borderColor = isBlue
          ? (isSelected ? 'border-sky-300' : 'border-sky-400')
          : (isSelected ? 'border-white' : 'border-slate-100');

        const glowShadow = isBlue
          ? (isSelected ? 'shadow-[0_0_8px_rgba(56,189,248,0.7)]' : isHovered ? 'shadow-[0_0_5px_rgba(56,189,248,0.45)]' : '')
          : (isSelected ? 'shadow-[0_0_8px_rgba(255,255,255,0.7)]' : isHovered ? 'shadow-[0_0_5px_rgba(255,255,255,0.45)]' : '');

        const bracketColor = isBlue ? 'border-sky-300' : 'border-white';
        const effectiveZIndex = isSelected || isHovered ? 50 : placement.zIndex;
        const isCoasting = tube.status === 'COASTED' || Boolean(tube.is_coasting);
        const tooltipBelow = tube.y < 20;

        return (
          <div
            key={`bbox-${tube.id}`}
            role={boxesOnly ? undefined : "button"}
            tabIndex={boxesOnly ? -1 : 0}
            aria-label={`Select tube end ${tube.id}${tube.tube_color ? `, ${tube.tube_color}` : ''}${tube.class ? `, ${tube.class}` : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              onSelectTube(tube.id === selectedTubeId ? null : tube.id);
            }}
            onMouseEnter={() => setHoveredTubeId(tube.id)}
            onMouseLeave={(event) => {
              // The popup is an overflowing child above the box. Keep it open while the
              // pointer moves from the box into that child; clear hover only on a true exit.
              const nextTarget = event.relatedTarget;
              if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
                setHoveredTubeId(null);
              }
            }}
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
            className={`absolute border rounded-xs transition-colors duration-75 ${
              boxesOnly ? 'pointer-events-none' : 'pointer-events-auto cursor-pointer hover:bg-white/5'
            } ${borderColor} ${glowShadow} bg-transparent ${isCoasting ? 'border-dashed opacity-70' : 'border-solid'
              }`}
          >
            {/* Corner Crosshair Brackets (Slim, high-precision outer corners) */}
            <span className={`absolute -top-0.5 -left-0.5 w-2 h-2 border-t-[1.5px] border-l-[1.5px] ${bracketColor}`} />
            <span className={`absolute -top-0.5 -right-0.5 w-2 h-2 border-t-[1.5px] border-r-[1.5px] ${bracketColor}`} />
            <span className={`absolute -bottom-0.5 -left-0.5 w-2 h-2 border-b-[1.5px] border-l-[1.5px] ${bracketColor}`} />
            <span className={`absolute -bottom-0.5 -right-0.5 w-2 h-2 border-b-[1.5px] border-r-[1.5px] ${bracketColor}`} />

            {/* 3. PINS MODE: Minimalist, non-intrusive numbered circle pins */}
            {!boxesOnly && labelMode === 'pins' && (
              <div
                className={`absolute -top-2.5 -left-2.5 min-w-[20px] h-5 px-1 rounded-full flex items-center justify-center font-mono font-black text-[9.5px] shadow-md border pointer-events-none transition-transform ${
                  isBlue
                    ? 'bg-sky-500 text-white border-sky-300'
                    : 'bg-white text-slate-950 border-slate-300'
                } ${isHovered || isSelected ? 'scale-125 z-50' : ''}`}
              >
                {tube.displayId ?? (typeof tube.id === 'string' && tube.id.includes('-') ? tube.id.split('-')[1] : tube.id)}
              </div>
            )}

            {/* 4. QUADRANT COMPACT MODE: Badges placed in North, South, East, West with guide stems */}
            {!boxesOnly && labelMode === 'compact' && (
              <div
                style={placement.containerStyle}
                className={`px-1.5 py-0.5 rounded-md text-[9.5px] font-mono whitespace-nowrap flex items-center gap-1.5 backdrop-blur-md shadow-md pointer-events-none border transition-colors ${
                  isBlue
                    ? 'bg-slate-950/90 text-sky-200 border-sky-500/70'
                    : 'bg-slate-950/90 text-slate-100 border-white/70'
                } ${isHovered || isSelected ? 'scale-105 border-white text-white z-50 shadow-lg' : ''}`}
              >
                {/* Connecting guide line straight to the box edge */}
                <span style={placement.stemStyle} />

                <span className="font-bold flex items-center gap-1">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isBlue ? 'bg-sky-400' : 'bg-white'
                    }`}
                  />
                  <span>#{tube.displayId ?? (typeof tube.id === 'string' && tube.id.includes('-') ? tube.id.split('-')[1] : tube.id)}</span>
                  <span className={isBlue ? 'text-sky-300 font-extrabold' : 'text-slate-100 font-extrabold'}>
                    {tube.role}
                  </span>
                  {tube.tube_color && (
                    <span className={`px-1 py-0.2 rounded text-[8px] font-black uppercase ${
                      isBlue ? 'bg-sky-500 text-white' : 'bg-white text-slate-900'
                    }`}>
                      {tube.tube_color}
                    </span>
                  )}
                  {(tube.width_mm_depth != null || tube.width_mm != null || tube.width_px != null) && (
                    <span className="text-white font-mono opacity-90 font-bold">
                      {tube.width_mm_depth != null
                        ? `Ø${tube.width_mm_depth.toFixed(1)}mm${tube.width_px != null ? ` (${tube.width_px.toFixed(0)}px)` : ''}`
                        : tube.width_mm != null
                          ? `Ø${tube.width_mm.toFixed(1)}mm${tube.width_px != null ? ` (${tube.width_px.toFixed(0)}px)` : ''}`
                          : `${tube.width_px?.toFixed(0)}px`}
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
            {!boxesOnly && (isHovered || isSelected) && (
              <div
                onMouseEnter={() => setHoveredTubeId(tube.id)}
                onMouseLeave={() => setHoveredTubeId(null)}
                className={`absolute left-1/2 -translate-x-1/2 ${tooltipBelow ? 'top-full' : 'bottom-full'} p-2 rounded-xl bg-slate-950/95 backdrop-blur-lg border border-white/20 text-white shadow-xl z-50 pointer-events-auto min-w-[170px] flex flex-col gap-1 text-[10px] font-mono animate-in fade-in zoom-in-95 duration-100`}
              >
                <div className="flex items-center justify-between pb-1 border-b border-white/10 font-bold">
                  <span className="flex items-center gap-1">
                    <span className={`w-2 h-2 rounded-full ${isBlue ? 'bg-sky-400' : 'bg-white'}`} />
                    <span>{tube.role} #{tube.displayId ?? (typeof tube.id === 'string' && tube.id.includes('-') ? tube.id.split('-')[1] : tube.id)}</span>
                  </span>
                  {tube.tube_color && (
                    <span className={`px-1.5 py-0.5 rounded text-[8px] font-black uppercase ${
                      isBlue ? 'bg-sky-500 text-white' : 'bg-white text-slate-900'
                    }`}>
                      {tube.tube_color}
                    </span>
                  )}
                </div>

                <div className="flex items-center justify-between text-slate-300">
                  <span className="whitespace-nowrap">Width:</span>
                  <span className="font-bold text-white font-mono">
                    {typeof tube.width_mm_depth === 'number'
                      ? `${tube.width_mm_depth.toFixed(2)}mm${tube.width_px != null ? ` (${tube.width_px.toFixed(1)}px)` : ''}`
                      : typeof tube.width_mm === 'number'
                        ? `${tube.width_mm.toFixed(2)}mm${tube.width_px != null ? ` (${tube.width_px.toFixed(1)}px)` : ''}`
                        : typeof tube.width_px === 'number'
                          ? `${tube.width_px.toFixed(1)}px`
                          : 'null'}
                  </span>
                </div>

                {tube.z_mm != null && (
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="whitespace-nowrap">Depth (Z):</span>
                    <span className="font-bold text-cyan-300 font-mono">
                      {tube.z_mm.toFixed(0)}mm
                    </span>
                  </div>
                )}

                {/* <div className="flex items-center justify-between text-slate-300">
                  <span>Pair:</span>
                  <span className={tube.pair_id != null ? 'text-emerald-400 font-bold font-mono' : 'text-slate-400 font-mono'}>
                    {tube.pair_id != null ? `Tube #${tube.pair_id}` : 'null'}
                  </span>
                </div> */}

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
