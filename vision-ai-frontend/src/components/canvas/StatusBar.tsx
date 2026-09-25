import React from 'react';
import type { AnomalyStatus, TubeEntity } from '../../types';
import { useInferenceStore } from '../../store/inferenceStore';

interface StatusBarProps {
  anomalyStatus: AnomalyStatus;
  fps: number;
  backendStatus?: string;
  tubes?: TubeEntity[];
}

export const StatusBar: React.FC<StatusBarProps> = ({
  anomalyStatus,
  fps: _fps,
  backendStatus: _backendStatus,
}) => {
  const stats = useInferenceStore((state) => state.stats);
  const headsCount = anomalyStatus?.headsCount ?? stats?.heads_count ?? 0;
  const tailsCount = anomalyStatus?.tailsCount ?? stats?.tails_count ?? 0;

  return (
    <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 z-30 pointer-events-auto animate-in fade-in slide-in-from-bottom-2 duration-300 max-w-[calc(100%-24px)]">
      <div className="flex items-center gap-3 sm:gap-5 px-3.5 sm:px-5 py-2 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-lg text-[var(--text-primary)]">

        {/* Tube Heads */}
        <div className="flex flex-col items-center">
          <span className="text-[9.5px] text-cyan-400 uppercase tracking-wider font-semibold">
            Heads
          </span>
          <div className="flex items-center gap-1 mt-0.5">
            <span className="text-base sm:text-lg font-black text-cyan-300">
              {headsCount}
            </span>
          </div>
        </div>

        <div className="h-6 w-[1px] bg-[var(--border-color)]" />

        {/* Tube Tails */}
        <div className="flex flex-col items-center">
          <span className="text-[9.5px] text-emerald-400 uppercase tracking-wider font-semibold">
            Tails
          </span>
          <div className="flex items-center gap-1 mt-0.5">
            <span className="text-base sm:text-lg font-black text-emerald-300">
              {tailsCount}
            </span>
          </div>
        </div>

        <div className="h-6 w-[1px] bg-[var(--border-color)]" />

        {/* Tube Ends Status / Balance */}
        <div className="flex flex-col items-center">
          <span className="text-[9.5px] text-[var(--text-secondary)] uppercase tracking-wider font-semibold">
            Pair Match
          </span>
          <div className="mt-0.5">
            {headsCount === tailsCount ? (
              <span className="text-xs sm:text-sm font-bold text-emerald-400">
                MATCHED ({headsCount})
              </span>
            ) : (
              <span className="text-xs sm:text-sm font-bold text-amber-400">
                UNPAIRED ({Math.abs(headsCount - tailsCount)})
              </span>
            )}
          </div>
        </div>

        {stats?.bigger_tube && (
          <>
            <div className="h-6 w-[1px] bg-[var(--border-color)]" />
            <div className="flex flex-col items-center">
              <span className="text-[9.5px] text-purple-400 uppercase tracking-wider font-semibold">
                Bigger Tube
              </span>
              <span className="text-xs sm:text-sm font-bold font-mono text-purple-300">
                {stats.bigger_tube.width_mm
                  ? `${stats.bigger_tube.width_mm.toFixed(1)}mm`
                  : `${stats.bigger_tube.width_px}px`}
              </span>
            </div>
          </>
        )}

      </div>
    </div>
  );
};
