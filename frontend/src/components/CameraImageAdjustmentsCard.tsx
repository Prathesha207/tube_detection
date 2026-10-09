import React, { useState, useEffect, useRef, useCallback } from 'react';
import { SlidersHorizontal, RotateCcw, Focus, Sun, Zap, CheckCircle2, RefreshCw } from 'lucide-react';
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
  // Auto (Stream) is a fixed preset; Manual enables the same sliders.
  const [mode, setMode] = useState<'auto' | 'manual'>(config?.controlMode === 'manual' ? 'manual' : 'auto');

  const [exposure, setExposure] = useState<number>(config?.exposure ?? 8);
  const [gain, setGain] = useState<number>(config?.gain ?? 400);
  const [focus, setFocus] = useState<number>(config?.focus ?? 0);
  const [brightness, setBrightness] = useState<number>(config?.brightness ?? 0);
  const [contrast, setContrast] = useState<number>(config?.contrast ?? 50);
  const [syncStatus, setSyncStatus] = useState<'synced' | 'adjusting'>('synced');
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const presetRef = useRef({ exposure: config?.exposure ?? 8, gain: config?.gain ?? 400, focus: config?.focus ?? 0, brightness: config?.brightness ?? 0, contrast: config?.contrast ?? 50 });

  // Network queuing & pacing refs for ultra-smooth responsiveness
  const inFlightRef = useRef(false);
  const pendingUpdateRef = useRef<any>(null);
  const throttleTimerRef = useRef<any>(null);
  const lastSentTimeRef = useRef<number>(0);

  // Sync internal state when external config loads or changes
  useEffect(() => {
    if (config) {
      if (config.controlMode !== 'manual') {
        presetRef.current = { exposure: config.exposure ?? 8, gain: config.gain ?? 400, focus: config.focus ?? 0, brightness: config.brightness ?? 0, contrast: config.contrast ?? 50 };
        if (typeof config.exposure === 'number') setExposure(config.exposure);
        if (typeof config.gain === 'number') setGain(config.gain);
        if (typeof config.focus === 'number') setFocus(config.focus);
        if (typeof config.brightness === 'number') setBrightness(config.brightness);
        if (typeof config.contrast === 'number') setContrast(config.contrast);
        setMode('auto');
      } else {
        if (typeof config.exposure === 'number') setExposure(config.exposure);
        if (typeof config.gain === 'number') setGain(config.gain);
        if (typeof config.focus === 'number') setFocus(config.focus);
        if (typeof config.brightness === 'number') setBrightness(config.brightness);
        if (typeof config.contrast === 'number') setContrast(config.contrast);
        setMode('manual');
      }
    }
  }, [config?.id, config?.exposure, config?.gain, config?.focus, config?.brightness, config?.contrast, config?.controlMode]);

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
    const THROTTLE_MS = 40;

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

  // Handler: Switch between Auto (Standard Stream) and Manual Modes
  const handleSwitchMode = async (newMode: 'auto' | 'manual') => {
    setMode(newMode);
    onUpdateConfig?.({ controlMode: newMode });

    if (newMode === 'auto') {
      onUpdateConfig?.({ controlMode: 'auto', exposure, gain, focus, brightness, contrast, autoExposure: false, autoFocus: false });
      scheduleDispatch({
        control_mode: 'auto', exposure, gain, focus: config?.focus ?? focus,
        brightness, contrast, auto_exposure: false, auto_focus: false,
      });
    } else {
      scheduleDispatch({
        control_mode: 'manual',
        exposure,
        gain,
        focus,
        brightness,
        contrast,
        auto_exposure: false,
        auto_focus: false,
      });
    }
  };

  // Handler: Change Exposure
  const handleExposureChange = (newExp: number) => {
    setExposure(newExp);
    onUpdateConfig?.({ exposure: newExp, controlMode: 'manual' });
    scheduleDispatch({
      control_mode: 'manual',
      exposure: newExp,
      gain,
      focus,
      brightness,
      contrast,
      auto_exposure: false,
      auto_focus: false,
    });
  };

  // Handler: Change Gain (Sensor ISO)
  const handleGainChange = (newGain: number) => {
    setGain(newGain);
    onUpdateConfig?.({ gain: newGain, controlMode: 'manual' });
    scheduleDispatch({
      control_mode: 'manual',
      exposure,
      gain: newGain,
      focus,
      brightness,
      contrast,
      auto_exposure: false,
      auto_focus: false,
    });
  };

  // Handler: Change Focus (Lens Position)
  const handleFocusChange = (newFocus: number) => {
    setFocus(newFocus);
    onUpdateConfig?.({ focus: newFocus, controlMode: 'manual' });
    scheduleDispatch({
      control_mode: 'manual',
      exposure,
      gain,
      focus: newFocus,
      brightness,
      contrast,
      auto_exposure: false,
      auto_focus: false,
    });
  };

  // Handler: Change Brightness
  const handleBrightnessChange = (newB: number) => {
    setBrightness(newB);
    onUpdateConfig?.({ brightness: newB, controlMode: 'manual' });
    scheduleDispatch({
      control_mode: 'manual',
      exposure,
      gain,
      focus,
      brightness: newB,
      contrast,
      auto_exposure: false,
      auto_focus: false,
    });
  };

  // Handler: Change Contrast
  const handleContrastChange = (newC: number) => {
    setContrast(newC);
    onUpdateConfig?.({ contrast: newC, controlMode: 'manual' });
    scheduleDispatch({
      control_mode: 'manual',
      exposure,
      gain,
      focus,
      brightness,
      contrast: newC,
      auto_exposure: false,
      auto_focus: false,
    });
  };

  // Reset to Calibrated Default Values
  const handleResetDefaults = () => {
    const { exposure: defaultExp, gain: defaultGain, focus: defaultFocus, brightness: defaultB, contrast: defaultC } = presetRef.current;

    setExposure(defaultExp);
    setGain(defaultGain);
    setFocus(defaultFocus);
    setBrightness(defaultB);
    setContrast(defaultC);

    onUpdateConfig?.({
      controlMode: 'manual',
      exposure: defaultExp,
      gain: defaultGain,
      focus: defaultFocus,
      brightness: defaultB,
      contrast: defaultC,
      autoExposure: false,
      autoFocus: false,
    });

    scheduleDispatch({
      control_mode: 'manual',
      exposure: defaultExp,
      gain: defaultGain,
      focus: defaultFocus,
      brightness: defaultB,
      contrast: defaultC,
      auto_exposure: false,
      auto_focus: false,
    });
  };

  const handleCalibrate = async () => {
    setCalibrating(true);
    setCalibrationError(null);
    setSyncStatus('adjusting');
    try {
      const settings = await cameraService.recalibrateControls();
      presetRef.current = { exposure: settings.exposure, gain: settings.gain, focus: settings.focus ?? undefined, brightness: settings.brightness ?? 0, contrast: settings.contrast ?? 50 };
      setExposure(settings.exposure);
      setGain(settings.gain);
      setFocus(settings.focus ?? 0);
      setBrightness(settings.brightness ?? 0);
      setContrast(settings.contrast ?? 50);
      setMode('auto');
      onUpdateConfig?.({
        controlMode: 'auto', exposure: settings.exposure, gain: settings.gain,
        iso: settings.gain, focus: settings.focus ?? undefined,
        focusAvailable: Boolean(settings.focus_available), brightness: settings.brightness ?? 0,
        contrast: settings.contrast ?? 50, autoExposure: false, autoFocus: false,
      });
      setSyncStatus('synced');
    } catch (err: any) {
      const message = err?.response?.data?.detail || err?.message || 'Calibration failed';
      setCalibrationError(message);
      setSyncStatus('synced');
    } finally {
      setCalibrating(false);
    }
  };

  const getSliderStyle = (val: number, min: number, max: number) => {
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    return {
      background: `linear-gradient(to right, var(--accent-pond) 0%, var(--accent-pond) ${pct}%, var(--border-color) ${pct}%, var(--border-color) 100%)`,
      accentColor: 'var(--accent-pond)',
    };
  };

  return (
    <div className="p-3 rounded-2xl border border-[var(--border-color)] bg-[var(--bg-card)] shadow-md flex flex-col w-full shrink-0 overflow-hidden transition-all duration-200">
      {/* 1. Header & Live Sync Status */}
      <div className="flex items-center justify-between pb-2.5 border-b border-[var(--border-color)]">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-[var(--accent-pond)]" />
          <span className="font-semibold text-xs tracking-wider uppercase text-[var(--text-primary)]">
            Camera Settings
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

          {mode === 'manual' && (
            <button
              type="button"
              onClick={handleResetDefaults}
              title="Reset sliders to default calibration"
              className="p-1 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)] transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* 2. Mode Selector: Auto (Standard Stream) vs Manual */}
      <div className="grid grid-cols-2 p-1 bg-[var(--bg-card-subtle)] rounded-xl border border-[var(--border-color)] mt-2.5 gap-1 items-center">
        <button
          type="button"
          onClick={() => handleSwitchMode('auto')}
          title="Default steady stream: autofocus and autoexposure hunting are disabled for stable inspection"
          className={`w-full py-1.5 px-2.5 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer select-none ${
            mode === 'auto'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
          }`}
        >
          <Zap className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">Auto (Stream)</span>
        </button>

        <button
          type="button"
          onClick={() => handleSwitchMode('manual')}
          title="Open manual sliders to adjust focus, shutter speed, and sensor gain"
          className={`w-full py-1.5 px-2.5 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer select-none ${
            mode === 'manual'
              ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
          }`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">Manual</span>
        </button>
      </div>

      {/* 3. Conditional Content: Auto (Default Stream) vs Manual Sliders */}
      {mode === 'auto' && (
        /* Auto (Stream) preset status */
        <div className="mt-3 p-3 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)] flex items-start gap-2.5 transition-all">
          <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
          <div className="flex flex-col gap-1 text-left">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-bold text-[var(--text-primary)]">Steady Stream Active</span>
              <span className="text-[9px] font-mono font-bold px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                LOCKED
              </span>
            </div>
            <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
              The displayed preset values are locked. Calibrate briefly lets the camera choose exposure and focus, then saves and locks those readings.
            </p>
            <div className="flex items-center gap-2 mt-1">
              <button type="button" onClick={handleCalibrate} disabled={calibrating || !isLive} className="inline-flex items-center gap-1.5 self-start px-2.5 py-1.5 rounded-lg bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] text-[11px] font-semibold disabled:opacity-50">
                <RefreshCw className={`w-3 h-3 ${calibrating ? 'animate-spin' : ''}`} />
                {calibrating ? 'Calibrating...' : 'Calibrate'}
              </button>
              <span className="text-[10px] text-[var(--text-muted)] italic">Focus may be off? Press Calibrate</span>
            </div>
            {calibrationError && <p className="text-[11px] text-red-500">{calibrationError}</p>}
          </div>
        </div>
      )}
      <fieldset disabled={mode === 'auto'} className="min-w-0 disabled:opacity-60">
        <div className="pt-3 flex flex-col gap-3 transition-all animate-fadeIn">
          {/* --- 1. SENSOR GAIN (ISO) --- */}
          <div className="flex flex-col gap-1.5 p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-bold text-[var(--text-primary)]">Sensor Gain</span>
              </div>
              <span className="font-mono text-xs font-bold text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded">
                ISO {gain}
              </span>
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
              <span className="font-mono text-xs font-bold text-blue-500 bg-blue-500/10 px-1.5 py-0.5 rounded">
                {config?.focusAvailable === false ? 'Unavailable' : `${focus} / 255`}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={255}
              step={1}
              value={focus}
              disabled={config?.focusAvailable === false}
              onChange={(e) => handleFocusChange(parseInt(e.target.value, 10))}
              style={getSliderStyle(focus, 0, 255)}
              className="w-full h-1.5 rounded cursor-pointer transition-all"
            />
            <div className="flex justify-between text-[9px] text-[var(--text-muted)] font-mono">
              <span>0 (Far)</span>
              <span>Preset value</span>
              <span>255 (Macro)</span>
            </div>
          </div>

          {/* --- 3. EXPOSURE (SHUTTER TIME) --- */}
          <div className="flex flex-col gap-1.5 p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Sun className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-bold text-[var(--text-primary)]">Exposure Shutter</span>
              </div>
              <span className="font-mono text-xs font-bold text-[var(--accent-pond)] bg-[var(--accent-pond-subtle)] px-1.5 py-0.5 rounded">
                {exposure} ms
              </span>
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
              <span>Preset value</span>
              <span>33 ms (Max Light)</span>
            </div>
          </div>

          {/* --- 4. TONE CONTROLS (Brightness & Contrast) --- */}
          <div className="flex flex-col gap-2 pt-2 border-t border-[var(--border-color)]">
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

          {/* --- 5. Reset Action --- */}
          <div className="pt-2 flex items-center gap-2 border-t border-[var(--border-color)]">
            <button
              type="button"
              onClick={handleResetDefaults}
              className="flex-1 h-7 rounded-xl border border-[var(--border-color)] hover:bg-[var(--btn-secondary-hover)] text-[var(--text-secondary)] text-[11px] font-medium flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Default Calibration
            </button>
          </div>
        </div>
      </fieldset>
    </div>
  );
};
