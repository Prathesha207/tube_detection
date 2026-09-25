import React, { useState, useEffect, useRef, useCallback } from 'react';
import { SlidersHorizontal, Sparkles, RotateCcw, Check, CheckCircle2 } from 'lucide-react';
import { CameraConfig } from '../types';
import { cameraService } from './service/cameraService';

interface CameraImageAdjustmentsCardProps {
  cameraId?: number | string | null;
  config?: CameraConfig;
  onUpdateConfig?: (newConfig: Partial<CameraConfig>) => void;
  isLive?: boolean;
}

export const CameraImageAdjustmentsCard: React.FC<CameraImageAdjustmentsCardProps> = ({
  cameraId,
  config,
  onUpdateConfig,
  isLive = true,
}) => {
  const [brightness, setBrightness] = useState<number>(config?.brightness ?? 0);
  const [contrast, setContrast] = useState<number>(config?.contrast ?? 50);
  const [exposure, setExposure] = useState<number>(config?.exposure ?? 50);
  const [autoFocus, setAutoFocus] = useState<boolean>(config?.autoFocus ?? true);
  const [syncStatus, setSyncStatus] = useState<'synced' | 'adjusting'>('synced');

  const debounceTimerRef = useRef<any>(null);
  const isMountedRef = useRef(false);

  // Sync internal state when external config loads or changes
  useEffect(() => {
    if (config) {
      if (typeof config.brightness === 'number') setBrightness(config.brightness);
      if (typeof config.contrast === 'number') setContrast(config.contrast);
      if (typeof config.exposure === 'number') setExposure(config.exposure);
      if (typeof config.autoFocus === 'boolean') setAutoFocus(config.autoFocus);
    }
  }, [config?.id]);

  // Live auto-apply function to backend
  const applyLiveControls = useCallback(
    async (b: number, c: number, exp: number, af: boolean) => {
      setSyncStatus('adjusting');
      try {
        await cameraService.updateLiveControls(cameraId || config?.id, {
          brightness: b,
          contrast: c,
          exposure: exp,
          auto_focus: af,
          autoFocus: af,
        });
        onUpdateConfig?.({
          brightness: b,
          contrast: c,
          exposure: exp,
          autoFocus: af,
        });
        setSyncStatus('synced');
      } catch (err) {
        console.warn('[CameraImageAdjustments] Failed to apply live controls:', err);
        setSyncStatus('synced');
      }
    },
    [cameraId, config?.id, onUpdateConfig]
  );

  // Real-time automatic adjustment when any control changes
  useEffect(() => {
    if (!isMountedRef.current) {
      isMountedRef.current = true;
      return;
    }

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    setSyncStatus('adjusting');
    debounceTimerRef.current = setTimeout(() => {
      applyLiveControls(brightness, contrast, exposure, autoFocus);
    }, 70);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [brightness, contrast, exposure, autoFocus, applyLiveControls]);

  // Automatic one-click optimal calibration
  const handleAutoAdjust = () => {
    const optimalBrightness = 0;
    const optimalContrast = 50;
    const optimalExposure = 45;
    const optimalAutoFocus = true;

    setBrightness(optimalBrightness);
    setContrast(optimalContrast);
    setExposure(optimalExposure);
    setAutoFocus(optimalAutoFocus);

    applyLiveControls(optimalBrightness, optimalContrast, optimalExposure, optimalAutoFocus);
  };

  // Reset to default hardware settings
  const handleReset = () => {
    setBrightness(0);
    setContrast(50);
    setExposure(50);
    setAutoFocus(true);
    applyLiveControls(0, 50, 50, true);
  };

  const getSliderStyle = (val: number, min: number, max: number) => {
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    return {
      background: `linear-gradient(to right, var(--accent-pond) 0%, var(--accent-pond) ${pct}%, var(--border-color) ${pct}%, var(--border-color) 100%)`,
      accentColor: 'var(--accent-pond)',
    };
  };

  return (
    <div className="p-3.5 rounded-2xl border border-[var(--border-color)] bg-[var(--bg-card)] shadow-md flex flex-col w-full shrink-0 overflow-hidden transition-all duration-200">
      {/* 1. Header */}
      <div className="flex items-center justify-between pb-2.5 border-b border-[var(--border-color)]">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-[var(--accent-pond)]" />
          <span className="font-semibold text-xs tracking-wider uppercase text-[var(--text-primary)]">
            Image Adjustments
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] font-mono font-bold bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] border border-[var(--border-color)]">
            {syncStatus === 'adjusting' ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                ADJUSTING
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                LIVE SYNC
              </>
            )}
          </span>

          <button
            type="button"
            onClick={handleReset}
            title="Reset to default image adjustments"
            className="p-1 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)] transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 2. Adjustment Sliders */}
      <div className="pt-3 flex flex-col gap-3">
        {/* Brightness */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs font-semibold text-[var(--text-primary)]">
            <span>Brightness</span>
            <span className="font-mono text-xs font-bold text-[var(--accent-pond)] bg-[var(--accent-pond-subtle)] px-1.5 py-0.5 rounded">
              {brightness > 0 ? `+${brightness}` : brightness}
            </span>
          </div>
          <input
            type="range"
            min={-50}
            max={50}
            value={brightness}
            onChange={(e) => setBrightness(parseInt(e.target.value, 10))}
            style={getSliderStyle(brightness, -50, 50)}
            className="w-full h-1.5 rounded cursor-pointer transition-all"
          />
        </div>

        {/* Contrast */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs font-semibold text-[var(--text-primary)]">
            <span>Contrast</span>
            <span className="font-mono text-xs font-bold text-[var(--accent-pond)] bg-[var(--accent-pond-subtle)] px-1.5 py-0.5 rounded">
              {contrast}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={contrast}
            onChange={(e) => setContrast(parseInt(e.target.value, 10))}
            style={getSliderStyle(contrast, 0, 100)}
            className="w-full h-1.5 rounded cursor-pointer transition-all"
          />
        </div>

        {/* Exposure Time */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-xs font-semibold text-[var(--text-primary)]">
            <span>Exposure Time</span>
            <span className="font-mono text-xs font-bold text-[var(--accent-pond)] bg-[var(--accent-pond-subtle)] px-1.5 py-0.5 rounded">
              {exposure} ms
            </span>
          </div>
          <input
            type="range"
            min={10}
            max={100}
            value={exposure}
            onChange={(e) => setExposure(parseInt(e.target.value, 10))}
            style={getSliderStyle(exposure, 10, 100)}
            className="w-full h-1.5 rounded cursor-pointer transition-all"
          />
        </div>

        {/* Continuous Auto-Focus Toggle */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--border-color)]">
          <div>
            <span className="text-xs font-semibold text-[var(--text-primary)] block">
              Continuous Auto-Focus
            </span>
            <span className="text-[10px] text-[var(--text-secondary)]">
              Automatic lens focus adjustment
            </span>
          </div>
          <button
            type="button"
            onClick={() => setAutoFocus(!autoFocus)}
            aria-pressed={autoFocus}
            title={autoFocus ? 'Auto-focus enabled' : 'Auto-focus disabled'}
            className={`w-9 h-5 flex items-center rounded-full p-0.5 transition-colors cursor-pointer ${
              autoFocus ? 'bg-[var(--accent-pond)]' : 'bg-[var(--border-color)]'
            }`}
          >
            <span
              className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                autoFocus ? 'translate-x-4' : 'translate-x-0'
              }`}
            />
          </button>
        </div>

        {/* 3. Action Buttons: Auto Adjust */}
        <div className="pt-2 flex items-center gap-2 border-t border-[var(--border-color)]">
          <button
            type="button"
            onClick={handleAutoAdjust}
            className="flex-1 h-7 rounded-xl bg-[var(--accent-pond-subtle)] hover:bg-[var(--accent-pond)]/20 border border-[var(--border-color)] text-[var(--accent-pond)] text-[11px] font-bold flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Auto Adjust All
          </button>

          <button
            type="button"
            onClick={handleReset}
            className="h-7 px-2.5 rounded-xl border border-[var(--border-color)] hover:bg-[var(--btn-secondary-hover)] text-[var(--text-secondary)] text-[11px] font-medium flex items-center justify-center transition-colors cursor-pointer"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
};
