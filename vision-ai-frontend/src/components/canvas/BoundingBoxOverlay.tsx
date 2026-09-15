import React from 'react';
import type { DuckEntity } from '../../types';
import { playDuckQuackSound } from '../../utils/audio';

interface BoundingBoxOverlayProps {
  ducks: DuckEntity[];
  selectedDuckId: string | null;
  onSelectDuck: (id: string | null) => void;
  showAllBoxes: boolean;
  isHandPresent: boolean;
  isSceneAnomaly?: boolean;
  isCountMismatch?: boolean;
}

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  ducks,
  selectedDuckId,
  onSelectDuck,
  showAllBoxes,
  isHandPresent,
  isSceneAnomaly = false,
  isCountMismatch = false,
}) => {
  if (isHandPresent) return null;

  const showConfidence = true;

  return (
    <div className="absolute inset-0 w-full h-full pointer-events-none z-20">
      {ducks
        .filter((duck) => {
          if (duck.species === 'Hand' || duck.handDetected) return false;
          const isMissing = duck.statusEvent === 'missing';
          // In "Anomalies Only" mode (showAllBoxes === false), ONLY show anomalous ducks (excess duck, missing duck, foreign toy, etc.)
          const isAnomalyDuck =
            !duck.provisional &&
            (duck.isAnomaly ||
              isMissing ||
              duck.species !== 'Duck' ||
              duck.statusEvent === 'added' ||
              duck.statusEvent === 'other_present');
          const isVisible = showAllBoxes || isAnomalyDuck;
          return isVisible && Number.isFinite(duck.x) && Number.isFinite(duck.y) && Number.isFinite(duck.width) && Number.isFinite(duck.height) && duck.width > 0 && duck.height > 0;
        })
        .map((duck, idx) => {
          const isProvisional = duck.provisional;
          const isMissing = duck.statusEvent === 'missing';
          // Only actual anomalous ducks (foreign species, excess duck, under-count frame, or explicit anomaly flag) are red
          const isIndividualAnomaly =
            !isProvisional &&
            !isMissing &&
            (duck.isAnomaly ||
              duck.species !== 'Duck');
          const isSelected = duck.id === selectedDuckId;

          // Default: Normal detected duck (Clean Emerald Green)
          let borderColor = isSelected ? 'border-emerald-400' : 'border-emerald-400/80';
          let bgColor = isSelected ? 'bg-emerald-500/30' : 'bg-emerald-500/5';
          let bracketColor = 'border-emerald-300';
          let tagStyle = isSelected
            ? 'bg-emerald-950/95 text-emerald-200 border border-emerald-400 font-bold shadow-xs'
            : 'bg-black/75 text-emerald-300 border border-emerald-500/30';
          let confColor = 'text-emerald-400';

          if (isMissing) {
            // Missing duck: YELLOW / AMBER dashed bounding box with alert tag
            borderColor = isSelected ? 'border-dashed border-amber-400' : 'border-dashed border-amber-500';
            bgColor = isSelected ? 'bg-amber-500/35 animate-pulse' : 'bg-amber-500/15 animate-pulse';
            bracketColor = 'border-amber-400';
            tagStyle = 'bg-amber-950/90 text-amber-200 border border-amber-500/80 font-bold shadow-xs';
            confColor = 'text-amber-300 font-bold';
          } else if (isProvisional) {
            // Provisional / Warming up duck: Amber/Yellow
            borderColor = isSelected ? 'border-amber-300' : 'border-amber-400/90';
            bgColor = isSelected ? 'bg-amber-500/30' : 'bg-amber-500/10';
            bracketColor = 'border-amber-300';
            tagStyle = 'bg-amber-950/90 text-amber-200 border border-amber-500/50';
            confColor = 'text-amber-300 font-bold';
          } else if (isIndividualAnomaly) {
            // Individual Anomaly (foreign object, unexpected toy, anomaly flag): Red / Rose
            borderColor = isSelected ? 'border-rose-400' : 'border-rose-500';
            bgColor = isSelected ? 'bg-rose-500/30' : 'bg-rose-500/10';
            bracketColor = 'border-rose-400';
            tagStyle = 'bg-rose-950/90 text-rose-200 border border-rose-500/80 font-bold shadow-xs';
            confColor = 'text-rose-300 font-bold';
          }

          let extraHighlight = '';
          if (isSelected) {
            // No extra large white border: use same color screen and opacity
            if (isIndividualAnomaly) {
              extraHighlight = 'shadow-[0_0_12px_rgba(244,63,94,0.35)]';
            } else if (isMissing || isProvisional) {
              extraHighlight = 'shadow-[0_0_12px_rgba(245,158,11,0.35)]';
            } else {
              extraHighlight = 'shadow-[0_0_12px_rgba(52,211,153,0.35)]';
            }
          }

          return (
            <div
              key={`bbox-${idx}-${duck.id}`}
              role="button"
              tabIndex={0}
              aria-label={`Select duck ${duck.id}`}
              onClick={(e) => {
                e.stopPropagation();
                playDuckQuackSound();
                onSelectDuck(duck.id === selectedDuckId ? null : duck.id);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.stopPropagation();
                  e.preventDefault();
                  playDuckQuackSound();
                  onSelectDuck(duck.id === selectedDuckId ? null : duck.id);
                }
              }}
              style={{
                left: `${duck.x}%`,
                top: `${duck.y}%`,
                width: `${duck.width}%`,
                height: `${duck.height}%`,
              }}
              className={`absolute border rounded pointer-events-auto cursor-pointer transition-colors duration-150 ${borderColor} ${bgColor} ${extraHighlight}`}
            >
              <span className={`absolute -top-0.5 -left-0.5 w-1.5 h-1.5 border-t-2 border-l-2 ${bracketColor}`} />
              <span className={`absolute -top-0.5 -right-0.5 w-1.5 h-1.5 border-t-2 border-r-2 ${bracketColor}`} />
              <span className={`absolute -bottom-0.5 -left-0.5 w-1.5 h-1.5 border-b-2 border-l-2 ${bracketColor}`} />
              <span className={`absolute -bottom-0.5 -right-0.5 w-1.5 h-1.5 border-b-2 border-r-2 ${bracketColor}`} />

              <div
                className={`absolute -top-5 left-0 px-1.5 py-0.5 rounded text-[9px] font-mono font-medium whitespace-nowrap flex items-center gap-1 backdrop-blur-xs pointer-events-none ${tagStyle}`}
              >
                <span>
                  {isMissing ? (
                    <span className="font-black text-amber-200">
                      #{duck.id} MISSING
                    </span>
                  ) : isProvisional ? (
                    'WARMING_UP'
                  ) : isIndividualAnomaly ? (
                    <span className="font-black text-rose-200">
                      {duck.species === 'Duck'
                        ? (duck.statusEvent === 'added' || duck.id.startsWith('unbound') || duck.id.startsWith('extra')
                            ? (duck.id.startsWith('unbound') || duck.id.startsWith('extra') ? 'EXCESS DUCK' : `#${duck.id} EXCESS`)
                            : `#${duck.id}`)
                        : (duck.species || 'Unknown')}
                    </span>
                  ) : (
                    `#${duck.id}`
                  )}
                </span>
                {/* {showConfidence && !isMissing && (
                  <span className={`text-[8.5px] opacity-80 ${confColor}`}>
                    {(duck.confidence * 100).toFixed(0)}%
                  </span>
                )} */}
              </div>
            </div>
          );
        })}
    </div>
  );
};
