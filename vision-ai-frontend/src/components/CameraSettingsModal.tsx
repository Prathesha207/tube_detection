import React, { useState, useEffect, useCallback, useRef } from 'react';
import { CameraConfig } from '../types';
import {
  Camera,
  Check,
  Cpu,
  Wifi,
  Loader2,
  Usb,
  RefreshCw,
  AlertCircle,
  Radio,
  Trash2,
} from 'lucide-react';
import { Modal, Button } from './ui';
import { cameraService, type CameraData } from './service/cameraService';

interface CameraSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: CameraConfig;
  onSaveConfig: (newConfig: CameraConfig) => void;
  onReconnect: (configToConnect?: CameraConfig) => Promise<void> | void;
}

interface SavedCameraRow {
  id: number;
  name: string;
  ip_address: string;
  resolution?: string;
  fps?: number;
  is_enabled?: boolean;
}

interface HwDevice {
  mxid: string;
  name: string;
  ip_or_id: string;
  is_usb: boolean;
}

const isUsbAddress = (ip?: string) => !ip || ip.toLowerCase().includes('usb') || /^[0-9a-f-]{8,}$/i.test(ip || '');

export const CameraSettingsModal: React.FC<CameraSettingsModalProps> = ({
  isOpen,
  onClose,
  config,
  onSaveConfig,
  onReconnect,
}) => {
  const [localConfig, setLocalConfig] = useState<CameraConfig>({ ...config });
  const [activeTab, setActiveTab] = useState<'stream' | 'image' | 'oak'>('stream');

  const [savedCameras, setSavedCameras] = useState<SavedCameraRow[]>([]);
  const [, setHwDevices] = useState<HwDevice[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [isLoadingList, setIsLoadingList] = useState(false);

  const [connectingId, setConnectingId] = useState<number | 'new' | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [activeCameraId, setActiveCameraId] = useState<number | null>(null);

  // Error tracking for user visibility
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [errorCameraId, setErrorCameraId] = useState<number | null>(null);

  // New Camera Form state
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState<'usb' | 'ip'>('usb');
  const [newIp, setNewIp] = useState('');
  const [detectedUsb, setDetectedUsb] = useState<HwDevice | null>(null);

  const onReconnectRef = useRef(onReconnect);
  onReconnectRef.current = onReconnect;

  // Load saved cameras + hardware scan
  const refreshAll = useCallback(async () => {
    setIsLoadingList(true);
    try {
      const cameras = await cameraService.checkCamera().catch(() => []);
      if (Array.isArray(cameras)) setSavedCameras(cameras);
    } finally {
      setIsLoadingList(false);
    }
  }, []);

  const scanUsbDevices = useCallback(async () => {
    setIsScanning(true);
    try {
      const devices = await cameraService.getAvailableDevices().catch(() => []);
      const list: HwDevice[] = Array.isArray(devices) ? devices : [];
      setHwDevices(list);
      const usb = list.find((d) => d.is_usb) || list[0] || null;
      setDetectedUsb(usb);
      return usb;
    } finally {
      setIsScanning(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setNewName('');
      setNewIp('');
      setDetectedUsb(null);
      setErrorMessage(null);
      setErrorCameraId(null);
      return;
    }
    setLocalConfig({ ...config, targetFps: config.targetFps || 30 });
    if (config.connected && config.id) {
      setActiveCameraId(typeof config.id === 'number' ? config.id : parseInt(String(config.id), 10) || null);
    } else {
      setActiveCameraId(null);
    }
    refreshAll();
    scanUsbDevices();
  }, [isOpen, config.connected, config.id, refreshAll, scanUsbDevices, config]);

  // Connect to a SAVED camera row
  const handleConnectSaved = async (row: SavedCameraRow) => {
    setConnectingId(row.id);
    setErrorMessage(null);
    setErrorCameraId(null);
    const updated: CameraConfig = {
      ...localConfig,
      id: row.id,
      sourceName: row.name,
      ipAddress: row.ip_address,
      resolution: (row.resolution as any) || localConfig.resolution || '1920x1080',
      targetFps: row.fps || localConfig.targetFps || 30,
      connected: true,
    };
    setLocalConfig(updated);
    try {
      await onReconnectRef.current(updated);
      setActiveCameraId(row.id);
      setErrorMessage(null);
      setErrorCameraId(null);
    } catch (err: any) {
      setActiveCameraId(null);
      setErrorCameraId(row.id);
      setErrorMessage(err?.message || 'Failed to connect to camera device');
    } finally {
      setConnectingId(null);
    }
  };

  // Direct one-click delete without confirm popup
  const handleDelete = async (row: SavedCameraRow) => {
    setDeletingId(row.id);
    try {
      await cameraService.deleteCamera(row.id);
      setSavedCameras((prev) => prev.filter((c) => c.id !== row.id));
      if (activeCameraId === row.id) setActiveCameraId(null);
      if (errorCameraId === row.id) {
        setErrorCameraId(null);
        setErrorMessage(null);
      }
    } catch (e) {
      console.error('Failed to delete camera:', e);
    } finally {
      setDeletingId(null);
    }
  };

  // Detect USB device
  const handleDetectUsb = async () => {
    setErrorMessage(null);
    const usb = await scanUsbDevices();
    if (usb && !newName.trim()) {
      setNewName(usb.name || 'Luxonis OAK-D Pro (USB 3.0)');
    }
  };

  // Create + connect new camera
  const handleAddAndConnect = async () => {
    const ipAddress = newType === 'usb' ? (detectedUsb?.ip_or_id || 'usb') : newIp.trim();
    const name = newName.trim() || (newType === 'usb' ? (detectedUsb?.name || 'USB OAK Camera') : `IP Camera (${newIp.trim()})`);

    if (newType === 'ip' && !newIp.trim()) {
      setErrorMessage('Please enter a valid IP address or hostname');
      return;
    }

    setConnectingId('new');
    setErrorMessage(null);
    setErrorCameraId(null);
    try {
      const payload: CameraData = {
        name,
        ip_address: ipAddress,
        resolution: localConfig.resolution || '1920x1080',
        fps: localConfig.targetFps || 30,
        control_mode: 'auto',
      };
      const saved = await cameraService.createCamera(payload);
      const updated: CameraConfig = {
        ...localConfig,
        id: saved?.id,
        sourceName: name,
        ipAddress,
        connected: true,
      };
      setLocalConfig(updated);
      await onReconnectRef.current(updated);
      setActiveCameraId(saved?.id ?? null);
      await refreshAll();
      setNewName('');
      setNewIp('');
      setErrorMessage(null);
    } catch (err: any) {
      setActiveCameraId(null);
      setErrorMessage(err?.message || 'Failed to connect to camera device');
    } finally {
      setConnectingId(null);
    }
  };

  const handleSave = () => {
    onSaveConfig(localConfig);
    onClose();
  };

  const getSliderStyle = (val: number, min: number, max: number) => {
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    return {
      background: `linear-gradient(to right, var(--accent-pond) 0%, var(--accent-pond) ${pct}%, var(--border-color) ${pct}%, var(--border-color) 100%)`,
      accentColor: 'var(--accent-pond)',
    };
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="lg"
      icon={<Camera className="w-4 h-4 text-[var(--accent-pond)]" />}
      title="Camera & Stream Settings"
      description="Manage connected cameras and video pipeline"
      footer={
        <>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-7 text-xs">
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleSave}
            icon={<Check className="w-3.5 h-3.5" />}
            className="h-7 text-xs"
          >
            Apply Changes
          </Button>
        </>
      }
    >
      {/* Tab Switcher: Compact */}
      <div className="flex border-b border-[var(--border-color)] pb-1.5 mb-3 -mt-1 shrink-0">
        {(['stream', 'image', 'oak'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => {
              setActiveTab(tab);
            }}
            className={`pb-1 px-2.5 text-xs font-bold border-b-2 transition-all cursor-pointer ${activeTab === tab
              ? 'border-[var(--accent-pond)] text-[var(--accent-pond)]'
              : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
              }`}
          >
            {tab === 'stream' ? 'Cameras & Connection' : tab === 'image' ? 'Image Adjustments' : 'OAK DepthAI VPU'}
          </button>
        ))}
      </div>

      {/* Main Container: Compact, single smooth scroll handled by parent modal */}
      <div className="space-y-2.5">
        {activeTab === 'stream' && (
          <>
            {/* ── 1. STREAM OUTPUT PROFILE (Resolution & Frame Count) ── */}
            <div className="p-2.5 rounded-xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {/* Resolution */}
                <div className="space-y-1">
                  <div className="text-[10px] font-bold text-[var(--text-secondary)]">Resolution</div>
                  <div className="inline-flex items-center gap-1 p-0.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] w-full">
                    {(['1920x1080', '1280x720'] as const).map((res) => {
                      const isSelected = localConfig.resolution === res;
                      return (
                        <button
                          key={res}
                          type="button"
                          onClick={() => {
                            setLocalConfig({ ...localConfig, resolution: res });
                          }}
                          className={`flex-1 py-1 px-2 rounded-md text-[11px] font-bold transition-all cursor-pointer text-center whitespace-nowrap ${isSelected
                            ? 'bg-[var(--accent-pond)] text-white shadow-xs'
                            : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                            }`}
                        >
                          {res === '1920x1080' ? '1080p FHD' : '720p HD'}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Frame Count Slider (up to 60 FPS) */}
                <div className="space-y-1 flex flex-col justify-between">
                  <div className="flex items-center justify-between text-[10px] font-bold text-[var(--text-secondary)]">
                    <span>Frame Count</span>
                    <span className="font-mono font-bold text-[var(--accent-pond)] px-1.5 py-0.2 rounded bg-[var(--bg-card)] border border-[var(--border-color)]">
                      {localConfig.targetFps} FPS
                    </span>
                  </div>
                  <div className="h-[27px] flex items-center px-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)]">
                    <input
                      type="range"
                      min={1}
                      max={60}
                      step={1}
                      value={localConfig.targetFps}
                      onChange={(e) => setLocalConfig({ ...localConfig, targetFps: parseInt(e.target.value, 10) })}
                      style={getSliderStyle(localConfig.targetFps, 1, 60)}
                      className="w-full h-1.5 rounded cursor-pointer transition-all"
                    />
                  </div>
                </div>

                {/* Recording Video Format */}
                <div className="sm:col-span-2 space-y-1 pt-1.5 border-t border-[var(--border-color)]/60">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-[var(--text-secondary)]">Recording Video Format</span>
                    <span className="text-[10px] font-semibold text-[var(--accent-pond)]">
                      {(localConfig.recordingFormat || 'MP4') === 'MP4'
                        ? 'MP4 container • H.264 web/universal codec'
                        : (localConfig.recordingFormat || 'MP4') === 'AVI'
                          ? 'AVI container • MJPEG analytical codec'
                          : 'MKV container • FFV1 lossless codec'}
                    </span>
                  </div>
                  <div className="inline-flex items-center gap-1 p-0.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] w-full">
                    {([
                      { id: 'MP4', label: 'MP4 (H.264)' },
                      { id: 'AVI', label: 'AVI (MJPEG)' },
                      { id: 'FFV1', label: 'MKV (FFV1)' },
                    ] as const).map((fmt) => {
                      const isSelected = (localConfig.recordingFormat || 'MP4') === fmt.id;
                      return (
                        <button
                          key={fmt.id}
                          type="button"
                          onClick={() => {
                            setLocalConfig({ ...localConfig, recordingFormat: fmt.id });
                          }}
                          className={`flex-1 py-1 px-2 rounded-md text-[11px] font-bold transition-all cursor-pointer text-center whitespace-nowrap ${isSelected
                            ? 'bg-[var(--accent-pond)] text-white shadow-xs'
                            : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                            }`}
                        >
                          {fmt.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* ── 2. CAMERA CONNECTION SETUP (Compact Card & Small Button) ── */}
            <div className="p-3 rounded-xl bg-[var(--bg-card)] border border-[var(--border-color)] space-y-2.5">
              <div className="flex items-center justify-between pb-1.5 border-b border-[var(--border-color)]">
                <div className="flex items-center gap-1.5">
                  <div className="w-5 h-5 rounded-md bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] flex items-center justify-center border border-[var(--border-color)]">
                    <Radio className="w-3 h-3" />
                  </div>
                  <span className="text-xs font-bold text-[var(--text-primary)]">
                    Camera Connection
                  </span>
                </div>
                {/* Interface Switcher Pills */}
                <div className="inline-flex items-center gap-1 p-0.5 rounded-lg bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
                  <button
                    type="button"
                    onClick={() => {
                      setNewType('usb');
                      setErrorMessage(null);
                    }}
                    className={`px-2.5 py-0.5 rounded-md text-[11px] font-bold flex items-center gap-1 transition-all cursor-pointer ${newType === 'usb'
                      ? 'bg-[var(--accent-pond)] text-white shadow-xs'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                      }`}
                  >
                    <Usb className="w-3 h-3" /> USB
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setNewType('ip');
                      setErrorMessage(null);
                    }}
                    className={`px-2.5 py-0.5 rounded-md text-[11px] font-bold flex items-center gap-1 transition-all cursor-pointer ${newType === 'ip'
                      ? 'bg-[var(--accent-pond)] text-white shadow-xs'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                      }`}
                  >
                    <Wifi className="w-3 h-3" /> IP / PoE
                  </button>
                </div>
              </div>

              {/* Camera Name Input */}
              <div>
                <label className="block text-[10px] font-bold text-[var(--text-secondary)] mb-0.5">
                  Camera Label
                </label>
                <input
                  type="text"
                  placeholder={newType === 'usb' ? (detectedUsb?.name || 'e.g., Dock Inspection OAK-D') : 'e.g., Cam 01'}
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-card-subtle)] border border-[var(--border-color)] text-xs font-medium text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-hidden focus:border-[var(--accent-pond)] focus:bg-[var(--bg-card)] transition-all"
                />
              </div>

              {/* Interface Details: Compact */}
              {newType === 'usb' ? (
                <div className="p-2 rounded-lg bg-[var(--bg-card-subtle)] border border-[var(--border-color)] flex items-center justify-between gap-2.5">
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="w-6 h-6 rounded-md bg-[var(--bg-card)] border border-[var(--border-color)] flex items-center justify-center text-[var(--accent-pond)] shrink-0">
                      <Usb className="w-3.5 h-3.5" />
                    </div>
                    <div className="min-w-0">
                      {detectedUsb ? (
                        <>
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs font-bold text-[var(--text-primary)] truncate">
                              {detectedUsb.name}
                            </span>
                            <span className="px-1.5 py-0.2 rounded text-[8px] font-mono font-bold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30">
                              READY
                            </span>
                          </div>
                          <span className="text-[9px] text-[var(--text-muted)] font-mono block truncate">
                            ID: {detectedUsb.mxid}
                          </span>
                        </>
                      ) : (
                        <div>
                          <span className="text-xs font-bold text-[var(--text-primary)] block">
                            No USB Device Detected
                          </span>
                          <span className="text-[9px] text-[var(--text-muted)]">
                            Plug in camera and scan
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={isScanning}
                    onClick={handleDetectUsb}
                    className="h-6 px-2 text-[11px] font-bold shrink-0"
                  >
                    {isScanning ? (
                      <Loader2 className="w-3 h-3 animate-spin text-[var(--accent-pond)]" />
                    ) : (
                      'Scan'
                    )}
                  </Button>
                </div>
              ) : (
                <div>
                  <label className="block text-[10px] font-bold text-[var(--text-secondary)] mb-0.5">
                    IP Address / Hostname
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., 192.168.1.100"
                    value={newIp}
                    onChange={(e) => setNewIp(e.target.value)}
                    className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-card-subtle)] border border-[var(--border-color)] text-xs font-mono font-medium text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-hidden focus:border-[var(--accent-pond)] focus:bg-[var(--bg-card)] transition-all"
                  />
                </div>
              )}

              {/* Form Error Banner */}
              {errorMessage && !errorCameraId && (
                <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-[11px] flex items-start gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-rose-500" />
                  <div className="min-w-0">
                    <span className="font-bold">Error: </span>
                    <span className="leading-relaxed break-words">{errorMessage}</span>
                  </div>
                </div>
              )}

              {/* Compact Small Action Button (Right-aligned / Compact) */}
              <div className="flex justify-end pt-0.5">
                <Button
                  variant="primary"
                  size="sm"
                  disabled={
                    connectingId === 'new' ||
                    (newType === 'usb' && !detectedUsb) ||
                    (newType === 'ip' && !newIp.trim())
                  }
                  onClick={handleAddAndConnect}
                  className="h-7 px-3.5 text-xs font-bold flex items-center gap-1.5 cursor-pointer shadow-xs"
                >
                  {connectingId === 'new' ? (
                    <>
                      <Loader2 className="w-3 h-3 animate-spin" />
                      Connecting...
                    </>
                  ) : (
                    'Add & Connect'
                  )}
                </Button>
              </div>
            </div>

            {/* ── 3. SAVED / REGISTERED CAMERAS (Compact Cards with Left Active Dot) ── */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between px-0.5">
                <span className="text-[10px] font-black uppercase tracking-wider text-[var(--text-muted)]">
                  Configured Cameras {savedCameras.length > 0 && `(${savedCameras.length})`}
                </span>
                <button
                  type="button"
                  onClick={() => refreshAll()}
                  disabled={isLoadingList}
                  title="Refresh device list"
                  className="p-1 rounded border border-[var(--border-color)] bg-[var(--bg-card)] text-[var(--text-secondary)] hover:text-[var(--accent-pond)] hover:border-[var(--accent-pond)] transition-all cursor-pointer flex items-center gap-1 text-[10px] px-1.5 font-medium"
                >
                  <RefreshCw className={`w-2.5 h-2.5 ${isLoadingList ? 'animate-spin text-[var(--accent-pond)]' : ''}`} />
                  Refresh
                </button>
              </div>

              {savedCameras.length === 0 && !isLoadingList && (
                <div className="p-2.5 rounded-xl border border-dashed border-[var(--border-color)] text-center text-[11px] text-[var(--text-muted)] bg-[var(--bg-card-subtle)]">
                  No cameras registered yet. Use Add &amp; Connect above to save cameras.
                </div>
              )}

              <div className="space-y-1.5">
                {savedCameras.map((row) => {
                  const usb = isUsbAddress(row.ip_address);
                  const isActive = activeCameraId === row.id && Boolean(config.connected);
                  const isConnecting = connectingId === row.id;
                  const isRowFailed = errorCameraId === row.id;

                  return (
                    <div
                      key={row.id}
                      className={`p-2 rounded-xl border flex flex-col gap-1 transition-all ${isRowFailed
                        ? 'border-rose-500/50 bg-rose-500/5'
                        : isActive
                          ? 'border-emerald-500/40 bg-emerald-500/5'
                          : 'border-[var(--border-color)] bg-[var(--bg-card)]'
                        }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          {/* Icon with subtle Active indicator dot on the left */}
                          <div
                            className={`relative w-7 h-7 rounded-lg border flex items-center justify-center shrink-0 ${isRowFailed
                              ? 'bg-rose-500/15 border-rose-500/30 text-rose-500'
                              : isActive
                                ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
                                : 'bg-[var(--accent-pond-subtle)] border-[var(--border-color)] text-[var(--accent-pond)]'
                              }`}
                          >
                            {usb ? <Usb className="w-3.5 h-3.5" /> : <Wifi className="w-3.5 h-3.5" />}
                            {/* Subtle active pulse dot at left on the camera badge */}
                            {isActive && (
                              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-emerald-500 ring-1 ring-[var(--bg-card)] animate-pulse" />
                            )}
                          </div>

                          {/* Name and specs without bulky middle badge */}
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="text-xs font-bold text-[var(--text-primary)] truncate max-w-[170px] sm:max-w-[260px]">
                                {row.name}
                              </span>
                              {isActive ? (
                                <span className="text-[9px] font-mono font-bold text-emerald-600 dark:text-emerald-400">
                                  ● LIVE
                                </span>
                              ) : isRowFailed ? (
                                <span className="text-[9px] font-mono font-bold text-rose-500">
                                  ● OFFLINE
                                </span>
                              ) : null}
                            </div>
                            <span className="text-[10px] text-[var(--text-muted)] font-mono block truncate">
                              {usb ? 'USB' : row.ip_address} • {row.resolution || '1920x1080'} • {row.fps || 30}fps
                            </span>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-1 shrink-0">
                          <Button
                            variant={isActive ? 'secondary' : 'primary'}
                            size="sm"
                            disabled={isConnecting}
                            onClick={() => handleConnectSaved(row)}
                            className="h-6 text-[11px] px-2.5 font-bold"
                          >
                            {isConnecting ? (
                              <Loader2 className="w-2.5 h-2.5 animate-spin" />
                            ) : isActive ? (
                              'Reconnect'
                            ) : (
                              'Connect'
                            )}
                          </Button>
                          <button
                            type="button"
                            onClick={() => handleDelete(row)}
                            disabled={deletingId === row.id}
                            title="Remove camera"
                            className="p-1 rounded-md border border-[var(--border-color)] bg-[var(--bg-card)] text-[var(--text-muted)] hover:text-red-500 hover:border-red-400 transition-all cursor-pointer"
                          >
                            {deletingId === row.id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <Trash2 className="w-3 h-3" />
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Error Banner if connection failed */}
                      {isRowFailed && errorMessage && (
                        <div className="p-1.5 rounded-md bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-[10px] flex items-start gap-1">
                          <AlertCircle className="w-3 h-3 shrink-0 mt-0.5 text-rose-500" />
                          <span className="leading-tight break-words">{errorMessage}</span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {activeTab === 'image' && (
          <div className="space-y-3 p-0.5">
            <div>
              <div className="flex items-center justify-between text-xs font-bold text-[var(--text-primary)] mb-1">
                <span>Brightness</span>
                <span className="font-mono text-[var(--accent-pond)] font-bold">{localConfig.brightness}</span>
              </div>
              <input
                type="range"
                min={-50}
                max={50}
                value={localConfig.brightness}
                onChange={(e) => setLocalConfig({ ...localConfig, brightness: parseInt(e.target.value, 10) })}
                style={getSliderStyle(localConfig.brightness, -50, 50)}
                className="w-full h-1.5 rounded cursor-pointer transition-all"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs font-bold text-[var(--text-primary)] mb-1">
                <span>Contrast</span>
                <span className="font-mono text-[var(--accent-pond)] font-bold">{localConfig.contrast}</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={localConfig.contrast}
                onChange={(e) => setLocalConfig({ ...localConfig, contrast: parseInt(e.target.value, 10) })}
                style={getSliderStyle(localConfig.contrast, 0, 100)}
                className="w-full h-1.5 rounded cursor-pointer transition-all"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs font-bold text-[var(--text-primary)] mb-1">
                <span>Exposure Time</span>
                <span className="font-mono text-[var(--accent-pond)] font-bold">{localConfig.exposure} ms</span>
              </div>
              <input
                type="range"
                min={10}
                max={100}
                value={localConfig.exposure}
                onChange={(e) => setLocalConfig({ ...localConfig, exposure: parseInt(e.target.value, 10) })}
                style={getSliderStyle(localConfig.exposure, 10, 100)}
                className="w-full h-1.5 rounded cursor-pointer transition-all"
              />
            </div>

            <div className="flex items-center justify-between pt-1.5 border-t border-[var(--border-color)]">
              <div>
                <span className="text-xs font-bold text-[var(--text-primary)] block">Continuous Auto-Focus</span>
                <span className="text-[10px] text-[var(--text-muted)]">Automatic lens focus adjustment</span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setLocalConfig({ ...localConfig, autoFocus: !localConfig.autoFocus });
                }}
                className={`w-10 h-5 flex items-center rounded-full p-0.5 transition-colors cursor-pointer ${localConfig.autoFocus ? 'bg-[var(--accent-pond)]' : 'bg-[var(--btn-secondary-border)]'
                  }`}
              >
                <span
                  className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${localConfig.autoFocus ? 'translate-x-5' : 'translate-x-0'
                    }`}
                />
              </button>
            </div>
          </div>
        )}

        {activeTab === 'oak' && (
          <div className="space-y-2.5 p-0.5">
            <div className="p-2.5 rounded-xl bg-[var(--accent-pond-subtle)] border border-[var(--border-color)] text-xs leading-relaxed text-[var(--text-primary)]">
              <div className="font-bold flex items-center gap-1.5 text-[var(--accent-pond)] mb-0.5">
                <Cpu className="w-3.5 h-3.5" />
                On-Device VPU Neural Engine
              </div>
              The YOLOv8-TubeTracker model runs entirely on the Myriad X / Keem Bay VPU inside the OAK camera, outputting bounding box coordinates directly at zero CPU overhead.
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between py-1 border-b border-[var(--border-color)]">
                <span className="text-[var(--text-secondary)]">Pipeline Mode</span>
                <span className="font-mono font-bold text-[var(--accent-pond)]">DepthAI Spatial Detection (YOLOv8)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[var(--border-color)]">
                <span className="text-[var(--text-secondary)]">Hardware Status</span>
                <span className={`font-mono font-bold ${config.connected ? 'text-emerald-500' : 'text-slate-400'}`}>
                  {config.connected ? 'Online • DepthAI Device Connected' : 'Standby • No Device Connected'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-[var(--border-color)]">
                <span className="text-[var(--text-secondary)]">Target VPU Latency</span>
                <span className="font-mono font-bold text-[var(--text-primary)]">~6.4 ms / frame</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[var(--text-secondary)]">Active Profile</span>
                <span className="font-mono font-bold text-[var(--text-primary)]">
                  {localConfig.resolution} @ {localConfig.targetFps} FPS
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
};
