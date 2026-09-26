import React, { useState, useEffect, useRef, useCallback } from 'react';
import { SlidersHorizontal, Sparkles, RotateCcw, Focus, Sun, Zap, Camera } from 'lucide-react';
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
  const [exposure, setExposure] = useState<number>(config?.exposure ?? 10);
  const [gain, setGain] = useState<number>(config?.gain ?? 100);
  const [focus, setFocus] = useState<number>(config?.focus ?? 120);
  const [autoExposure, setAutoExposure] = useState<boolean>(config?.autoExposure ?? false);
  const [autoFocus, setAutoFocus] = useState<boolean>(config?.autoFocus ?? false);
  const [brightness, setBrightness] = useState<number>(config?.brightness ?? 0);
  const [contrast, setContrast] = useState<number>(config?.contrast ?? 50);
  const [syncStatus, setSyncStatus] = useState<'synced' | 'adjusting'>('synced');

  // Network queuing & pacing refs for ultra-smooth responsiveness
  const inFlightRef = useRef(false);
  const pendingUpdateRef = useRef<any>(null);
  const throttleTimerRef = useRef<any>(null);
  const lastSentTimeRef = useRef<number>(0);
  const isMountedRef = useRef(false);

  // Sync internal state when external config loads or changes
  useEffect(() => {
    if (config) {
      if (typeof config.exposure === 'number') setExposure(config.exposure);
      if (typeof config.gain === 'number') setGain(config.gain);
      if (typeof config.focus === 'number') setFocus(config.focus);
      if (typeof config.autoExposure === 'boolean') setAutoExposure(config.autoExposure);
      if (typeof config.autoFocus === 'boolean') setAutoFocus(config.autoFocus);
      if (typeof config.brightness === 'number') setBrightness(config.brightness);
      if (typeof config.contrast === 'number') setContrast(config.contrast);
    }
  }, [config?.id]);

  // Actual network dispatch with in-flight queuing
  const dispatchToBackend = useCallback(async (payload: any) => {
    if (inFlightRef.current) {
      pendingUpdateRef.current = payload;
      return;
    }
    inFlightRef.current = true;
    lastSentTimeRef.current = Date.now();
    setSyncStatus('adjusting');

    try {
      await cameraService.updateLiveControls(cameraId || config?.id, payload);
      setSyncStatus('synced');
    } catch (err) {
      console.warn('[CameraImageAdjustments] Control send error:', err);
      setSyncStatus('synced');
    } finally {
      inFlightRef.current = false;
      if (pendingUpdateRef.current) {
        const next = pendingUpdateRef.current;
        pendingUpdateRef.current = null;
        dispatchToBackend(next);
      }
    }
  }, [cameraId, config?.id]);

  // Throttled schedule: immediate on first event, paced at ~40ms while dragging, guaranteed trailing
  const scheduleDispatch = useCallback((payload: any) => {
    const now = Date.now();
    const elapsed = now - lastSentTimeRef.current;
    const THROTTLE_MS = 40; // 25 updates/sec max rate for silky smooth feel

    if (elapsed >= THROTTLE_MS && !inFlightRef.current) {
      if (throttleTimerRef.current) {
        clearTimeout(throttleTimerRef.current);
        throttleTimerRef.current = null;
      }
      dispatchToBackend(payload);
    } else {
      pendingUpdateRef.current = payload;
      if (!throttleTimerRef.current) {
        throttleTimerRef.current = setTimeout(() => {
          throttleTimerRef.current = null;
          if (pendingUpdateRef.current) {
            const next = pendingUpdateRef.current;
            pendingUpdateRef.current = null;
            dispatchToBackend(next);
          }
        }, Math.max(10, THROTTLE_MS - elapsed));
      }
    }
  }, [dispatchToBackend]);

  // Handler: Change Exposure
  const handleExposureChange = (newExp: number) => {
    setExposure(newExp);
    setAutoExposure(false); // Automatically switch to manual exposure mode
    onUpdateConfig?.({ exposure: newExp, autoExposure: false });
    scheduleDispatch({
      exposure: newExp,
      gain,
      focus,
      brightness,
      contrast,
      auto_exposure: false,
      auto_focus: autoFocus,
    });
  };

  // Handler: Change Gain (Sensor ISO)
  const handleGainChange = (newGain: number) => {
    setGain(newGain);
    setAutoExposure(false); // Changing sensor gain automatically switches out of auto exposure
    onUpdateConfig?.({ gain: newGain, autoExposure: false });
    scheduleDispatch({
      exposure,
      gain: newGain,
      focus,
      brightness,
      contrast,
      auto_exposure: false,
      auto_focus: autoFocus,
    });
  };

  // Handler: Change Focus (Lens Position)
  const handleFocusChange = (newFocus: number) => {
    setFocus(newFocus);
    setAutoFocus(false); // Changing lens position automatically switches out of auto focus
    onUpdateConfig?.({ focus: newFocus, autoFocus: false });
    scheduleDispatch({
      exposure,
      gain,
      focus: newFocus,
      brightness,
      contrast,
      auto_exposure: autoExposure,
      auto_focus: false,
    });
  };

  // Handler: Toggle Auto Exposure
  const handleToggleAutoExposure = () => {
    const nextAe = !autoExposure;
    setAutoExposure(nextAe);
    onUpdateConfig?.({ autoExposure: nextAe });
    scheduleDispatch({
      exposure,
      gain,
      focus,
      brightness,
      contrast,
      auto_exposure: nextAe,
      auto_focus: autoFocus,
    });
  };

  // Handler: Toggle Auto Focus
  const handleToggleAutoFocus = () => {
    const nextAf = !autoFocus;
    setAutoFocus(nextAf);
    onUpdateConfig?.({ autoFocus: nextAf });
    scheduleDispatch({
      exposure,
      gain,
      focus,
      brightness,
      contrast,
      auto_exposure: autoExposure,
      auto_focus: nextAf,
    });
  };

  // Handler: Change Brightness
  const handleBrightnessChange = (newB: number) => {
    setBrightness(newB);
    onUpdateConfig?.({ brightness: newB });
    scheduleDispatch({
      exposure,
      gain,
      focus,
      brightness: newB,
      contrast,
      auto_exposure: autoExposure,
      auto_focus: autoFocus,
    });
  };

  // Handler: Change Contrast
  const handleContrastChange = (newC: number) => {
    setContrast(newC);
    onUpdateConfig?.({ contrast: newC });
    scheduleDispatch({
      exposure,
      gain,
      focus,
      brightness,
      contrast: newC,
      auto_exposure: autoExposure,
      auto_focus: autoFocus,
    });
  };

  // Auto Calibrate / Reset
  const handleAutoAdjust = () => {
    const defaultExp = 10;
    const defaultGain = 100;
    const defaultFocus = 120;
    const defaultB = 0;
    const defaultC = 50;

    setExposure(defaultExp);
    setGain(defaultGain);
    setFocus(defaultFocus);
    setAutoExposure(true);
    setAutoFocus(true);
    setBrightness(defaultB);
    setContrast(defaultC);

    onUpdateConfig?.({
      exposure: defaultExp,
      gain: defaultGain,
      focus: defaultFocus,
      autoExposure: true,
      autoFocus: true,
      brightness: defaultB,
      contrast: defaultC,
    });

    scheduleDispatch({
      exposure: defaultExp,
      gain: defaultGain,
      focus: defaultFocus,
      brightness: defaultB,
      contrast: defaultC,
      auto_exposure: true,
      auto_focus: true,
    });
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
            Camera Controls
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] font-mono font-bold bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] border border-[var(--border-color)]">
            {syncStatus === 'adjusting' ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                APPLYING
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
            onClick={handleAutoAdjust}
            title="Reset to default calibration"
            className="p-1 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)] transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 2. Primary Camera Controls (Exposure, Gain, Focus) */}
      <div className="pt-3 flex flex-col gap-3">
        {/* --- 1. SENSOR GAIN (ISO) --- */}
        <div className="flex flex-col gap-1.5 p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              <span className="text-xs font-bold text-[var(--text-primary)]">Sensor Gain</span>
            </div>
            <div className="flex items-center gap-1.5">
              {autoExposure && (
                <span className="text-[9px] font-mono font-bold px-1.5 py-0.2 rounded bg-amber-500/15 text-amber-600 dark:text-amber-400">
                  AE ACTIVE
                </span>
              )}
              <span className="font-mono text-xs font-bold text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded">
                ISO {gain}
              </span>
            </div>
          </div>
          <input
            type="range"
            min={100}
            max={1600}
            step={10}
            value={gain}
            onChange={(e) => handleGainChange(parseInt(e.target.value, 10))}
            style={getSliderStyle(gain, 100, 1600)}
            className="w-full h-1.5 rounded cursor-pointer transition-all"
          />
          <div className="flex justify-between text-[9px] text-[var(--text-muted)] font-mono">
            <span>100 (Clean)</span>
            <span>400 (Standard)</span>
            <span>800 (Boost)</span>
            <span>1600 (Max)</span>
          </div>
        </div>

        {/* --- 2. FOCUS (LENS POSITION) --- */}
        <div className="flex flex-col gap-1.5 p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Focus className="w-3.5 h-3.5 text-blue-500" />
              <span className="text-xs font-bold text-[var(--text-primary)]">Focus Lens</span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleToggleAutoFocus}
                title={autoFocus ? "Auto-focus active. Click for manual." : "Manual focus active. Click for auto."}
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition-all cursor-pointer ${
                  autoFocus
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'bg-[var(--btn-secondary-bg)] text-[var(--text-secondary)] border border-[var(--border-color)]'
                }`}
              >
                {autoFocus ? 'AUTO (AF)' : 'MANUAL'}
              </button>
              <span className="font-mono text-xs font-bold text-blue-500 bg-blue-500/10 px-1.5 py-0.5 rounded">
                {focus} / 255
              </span>
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={255}
            step={1}
            value={focus}
            onChange={(e) => handleFocusChange(parseInt(e.target.value, 10))}
            style={getSliderStyle(focus, 0, 255)}
            className="w-full h-1.5 rounded cursor-pointer transition-all"
          />
          <div className="flex justify-between text-[9px] text-[var(--text-muted)] font-mono">
            <span>0 (Far / Infinity)</span>
            <span>120 (Standard)</span>
            <span>255 (Macro / Close)</span>
          </div>
        </div>

        {/* --- 3. EXPOSURE (SHUTTER TIME) --- */}
        <div className="flex flex-col gap-1.5 p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Sun className="w-3.5 h-3.5 text-amber-500" />
              <span className="text-xs font-bold text-[var(--text-primary)]">Exposure</span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleToggleAutoExposure}
                title={autoExposure ? "Auto-exposure active. Click for manual." : "Manual exposure active. Click for auto."}
                className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold transition-all cursor-pointer ${
                  autoExposure
                    ? 'bg-[var(--accent-pond)] text-white shadow-xs'
                    : 'bg-[var(--btn-secondary-bg)] text-[var(--text-secondary)] border border-[var(--border-color)]'
                }`}
              >
                {autoExposure ? 'AUTO (AE)' : 'MANUAL'}
              </button>
              <span className="font-mono text-xs font-bold text-[var(--accent-pond)] bg-[var(--accent-pond-subtle)] px-1.5 py-0.5 rounded">
                {exposure} ms
              </span>
            </div>
          </div>
          <input
            type="range"
            min={1}
            max={33}
            step={1}
            value={exposure}
            onChange={(e) => handleExposureChange(parseInt(e.target.value, 10))}
            style={getSliderStyle(exposure, 1, 33)}
            className="w-full h-1.5 rounded cursor-pointer transition-all"
          />
          <div className="flex justify-between text-[9px] text-[var(--text-muted)] font-mono">
            <span>1 ms (Fast / Low Blur)</span>
            <span>16 ms (Standard)</span>
            <span>33 ms (Max Light)</span>
          </div>
        </div>

        {/* --- 4. SECONDARY TONE CONTROLS (Brightness & Contrast) --- */}
        <div className="flex flex-col gap-2 pt-2 border-t border-[var(--border-color)]">
          {/* Brightness */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-[11px] font-medium text-[var(--text-secondary)]">
              <span>Brightness</span>
              <span className="font-mono text-xs font-bold text-[var(--text-primary)]">
                {brightness > 0 ? `+${brightness}` : brightness}
              </span>
            </div>
            <input
              type="range"
              min={-50}
              max={50}
              value={brightness}
              onChange={(e) => handleBrightnessChange(parseInt(e.target.value, 10))}
              style={getSliderStyle(brightness, -50, 50)}
              className="w-full h-1.5 rounded cursor-pointer transition-all"
            />
          </div>

          {/* Contrast */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-[11px] font-medium text-[var(--text-secondary)]">
              <span>Contrast</span>
              <span className="font-mono text-xs font-bold text-[var(--text-primary)]">
                {contrast}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={contrast}
              onChange={(e) => handleContrastChange(parseInt(e.target.value, 10))}
              style={getSliderStyle(contrast, 0, 100)}
              className="w-full h-1.5 rounded cursor-pointer transition-all"
            />
          </div>
        </div>

        {/* --- 5. Action Buttons --- */}
        <div className="pt-2 flex items-center gap-2 border-t border-[var(--border-color)]">
          <button
            type="button"
            onClick={handleAutoAdjust}
            className="flex-1 h-7 rounded-xl bg-[var(--accent-pond-subtle)] hover:bg-[var(--accent-pond)]/20 border border-[var(--border-color)] text-[var(--accent-pond)] text-[11px] font-bold flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Auto Calibrate
          </button>

          <button
            type="button"
            onClick={handleAutoAdjust}
            className="h-7 px-2.5 rounded-xl border border-[var(--border-color)] hover:bg-[var(--btn-secondary-hover)] text-[var(--text-secondary)] text-[11px] font-medium flex items-center justify-center transition-colors cursor-pointer"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
};
