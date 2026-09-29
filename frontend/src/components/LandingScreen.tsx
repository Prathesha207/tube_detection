import React, { useState, useEffect, useRef } from 'react';
import {
  Bot,
  ArrowRight,
  Zap,
  Sparkles,
  ShieldCheck,
  Cpu,
  AlertTriangle,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { getApiBaseUrl } from '../lib/api';

interface LandingScreenProps {
  onInitialize: () => void;
  cameraConnected?: boolean | null;
}

const MAX_WAIT_MS = 30_000;
const POLL_INTERVAL_MS = 1000;

export const LandingScreen: React.FC<LandingScreenProps> = ({
  onInitialize,
  cameraConnected,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [healthStatus, setHealthStatus] = useState<'connecting' | 'starting' | 'ready' | 'error'>('connecting');
  const [healthMessage, setHealthMessage] = useState('Connecting to AI engine...');
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const startTimeRef = useRef(Date.now());
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    startTimeRef.current = Date.now();
    setHealthStatus('connecting');
    setHealthMessage('Connecting to AI engine...');

    const pollHealth = async () => {
      if (!mountedRef.current) return;

      try {
        const res = await fetch(`${getApiBaseUrl()}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'ready') {
            if (mountedRef.current) {
              setHealthStatus('ready');
              setHealthMessage('AI Engine ready');
            }
            return;
          } else if (data.status === 'starting') {
            if (mountedRef.current) {
              setHealthStatus('starting');
              setHealthMessage(
                data.stage === 'loading_models' ? 'Loading AI models...' : 'Initializing inference engine...'
              );
            }
          }
        }
      } catch {
        // Backend offline / booting up
      }

      const elapsed = Date.now() - startTimeRef.current;
      if (elapsed > 15000) {
        if (mountedRef.current && healthStatus !== 'starting') {
          setHealthMessage('AI engine is taking longer than expected...');
        }
      } else if (elapsed > 5000) {
        if (mountedRef.current && healthStatus === 'connecting') {
          setHealthMessage('Starting local backend...');
        }
      }

      if (elapsed > MAX_WAIT_MS) {
        if (mountedRef.current) {
          setHealthStatus('error');
        }
        return;
      }

      if (mountedRef.current) {
        setTimeout(pollHealth, POLL_INTERVAL_MS);
      }
    };

    pollHealth();

    return () => {
      mountedRef.current = false;
    };
  }, [retryCount]);

  const handleRetry = () => {
    setDetailsVisible(false);
    setRetryCount((prev) => prev + 1);
  };

  const handleStart = () => {
    if (healthStatus !== 'ready') return;
    onInitialize();
  };

  return (
    <div className="relative w-screen h-screen max-h-screen overflow-hidden bg-[var(--bg-page)] text-[var(--text-primary)] flex items-center justify-center p-4 sm:p-6 select-none transition-colors duration-300">
      <div className="absolute inset-0 pointer-events-none opacity-25 dark:opacity-20 bg-[radial-gradient(var(--accent-pond)_0.8px,transparent_0.8px)] [background-size:24px_24px]" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full bg-[var(--accent-pond)]/10 blur-3xl pointer-events-none" />

      <div className="relative z-10 w-full max-w-lg mx-auto flex flex-col items-center text-center animate-in fade-in zoom-in-95 duration-500 ease-out">
        {/* Logo / Badge */}
        <div className={`relative flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 rounded-3xl ${
          healthStatus === 'error'
            ? 'bg-[var(--status-warn-bg)] text-[var(--status-warn-text)]'
            : 'bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)]'
        } border border-[var(--border-color)] shadow-md mb-6 transition-transform duration-300 hover:scale-105`}>
          {healthStatus === 'error' ? (
            <AlertTriangle className="w-8 h-8 sm:w-10 sm:h-10" />
          ) : (
            <Bot className="w-8 h-8 sm:w-10 sm:h-10" />
          )}
          <span className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-[var(--bg-page)] shadow-xs ${
            healthStatus === 'ready'
              ? 'bg-[var(--status-normal-text)]'
              : healthStatus === 'error'
              ? 'bg-[var(--status-warn-text)]'
              : 'bg-[var(--accent-pond)] animate-pulse'
          }`} />
        </div>

        {/* Product Tag */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--bg-card)] border border-[var(--border-color)] text-[var(--accent-pond)] text-xs font-bold tracking-wide shadow-2xs mb-4">
          <Sparkles className="w-3.5 h-3.5 text-[var(--accent-tube)]" />
          <span>VisionMonitor &bull; v4.2 PRO</span>
        </div>

        {/* Title */}
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[var(--text-primary)] leading-tight mb-3">
          Tube Detection &amp; <br />
          <span className="text-[var(--accent-pond)]">Anomaly Monitoring</span>
        </h1>

        {/* Description */}
        <p className="text-sm sm:text-base text-[var(--text-secondary)] max-w-md mx-auto leading-relaxed mb-6 font-normal">
          Real-time YOLOv8 computer vision tracking, automated tube population counting, and instant anomaly alerts.
        </p>

        {/* Feature Pills */}
        <div className="flex items-center justify-center gap-4 sm:gap-6 mb-6 text-xs text-[var(--text-secondary)] font-medium">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-4 h-4 text-[var(--accent-pond)]" />
            <span>YOLOv8 Real-Time</span>
          </div>
          <span className="text-[var(--border-color)]">&bull;</span>
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-[var(--status-normal-text)]" />
            <span>Anomaly Safeguards</span>
          </div>
        </div>

        {/* Backend Status Indicator */}
        <div className="flex items-center gap-2 mb-6 px-3.5 py-1.5 rounded-full bg-[var(--bg-card)] border border-[var(--border-color)] text-xs shadow-2xs">
          {healthStatus === 'ready' ? (
            <>
              <span className="w-2 h-2 rounded-full bg-[var(--status-normal-text)] animate-pulse" />
              <span className="font-semibold text-[var(--status-normal-text)]">Backend Connected</span>
            </>
          ) : healthStatus === 'error' ? (
            <>
              <span className="w-2 h-2 rounded-full bg-[var(--status-anomaly-text)]" />
              <span className="font-semibold text-[var(--status-anomaly-text)]">Backend Offline</span>
            </>
          ) : (
            <>
              <Loader2 className="w-3 h-3 text-[var(--accent-pond)] animate-spin" />
              <span className="font-medium text-[var(--text-secondary)]">{healthMessage}</span>
            </>
          )}
        </div>

        {/* Main Action / Retry Button */}
        {healthStatus === 'error' ? (
          <div className="flex flex-col items-center w-full max-w-xs gap-3">
            <button
              onClick={handleRetry}
              className="w-full px-6 py-3 rounded-2xl bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] font-bold text-sm tracking-wide shadow-md transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer active:scale-98"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Retry Backend Connection</span>
            </button>
            <button
              onClick={() => setDetailsVisible(!detailsVisible)}
              className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] underline transition-colors cursor-pointer"
            >
              {detailsVisible ? 'Hide diagnostic details' : 'View diagnostic details'}
            </button>
            {detailsVisible && (
              <div className="mt-2 p-3 bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl text-left text-xs text-[var(--text-secondary)] w-full space-y-1">
                <p className="font-semibold text-[var(--text-primary)]">Backend did not respond on local port.</p>
                <p>Ensure backend is started: <code className="bg-[var(--bg-page)] px-1 rounded font-mono">uvicorn app.main:app --reload --port 8000</code></p>
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={handleStart}
            disabled={healthStatus !== 'ready'}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`w-full sm:w-auto min-w-[240px] px-8 py-3.5 rounded-2xl font-bold text-base tracking-wide transition-all duration-200 flex items-center justify-center gap-3 ${
              healthStatus === 'ready'
                ? 'bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] shadow-lg hover:shadow-xl hover:shadow-[var(--accent-pond)]/20 cursor-pointer active:scale-98 group'
                : 'bg-[var(--bg-card)] text-[var(--text-muted)] border border-[var(--border-color)] cursor-not-allowed opacity-80'
            }`}
          >
            {healthStatus === 'ready' ? (
              <>
                <Zap className="w-5 h-5 fill-current" />
                <span>Start System</span>
                <ArrowRight className={`w-5 h-5 transition-transform duration-200 ${isHovered ? 'translate-x-1' : ''}`} />
              </>
            ) : (
              <>
                <Loader2 className="w-5 h-5 animate-spin text-[var(--accent-pond)]" />
                <span>Starting AI Engine...</span>
              </>
            )}
          </button>
        )}

        <p className="text-[11px] text-[var(--text-muted)] mt-4">
          {healthStatus === 'ready'
            ? 'Click Start System to enter the monitoring workspace'
            : 'Connecting to local inference engine before launching'}
        </p>
      </div>
    </div>
  );
};
