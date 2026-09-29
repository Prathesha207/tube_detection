import React, { useState, useEffect, useCallback } from 'react';
import { ThemeMode, CameraConfig } from '../types';
import { 
  Sun, 
  Moon, 
  Leaf, 
  Settings, 
  HelpCircle, 
  Video, 
  Bot,
} from 'lucide-react';
import { IconButton } from './ui';
import { getApiBaseUrl } from '../lib/api';

interface HeaderProps {
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
  cameraConfig: CameraConfig;
  onOpenSettings: () => void;
  onOpenHelp: () => void;
  fps?: number;
  anomalyDetected?: boolean;
  onExitToLanding?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  theme,
  onThemeChange,
  cameraConfig,
  onOpenSettings,
  onOpenHelp,
  onExitToLanding,
}) => {
  const [backendConnected, setBackendConnected] = useState<boolean>(false);

  // Live Backend Health Ping
  const checkBackendHealth = useCallback(async () => {
    try {
      const baseUrl = getApiBaseUrl();
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${baseUrl}/health`, {
        method: 'GET',
        cache: 'no-store',
        signal: controller.signal,
      });
      clearTimeout(timer);
      setBackendConnected(res.ok || res.status === 503);
    } catch {
      setBackendConnected(false);
    }
  }, []);

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 4000);
    return () => clearInterval(interval);
  }, [checkBackendHealth]);

  const cycleTheme = () => {
    if (theme === 'pond-light') onThemeChange('pond-dark');
    else if (theme === 'pond-dark') onThemeChange('nature');
    else onThemeChange('pond-light');
  };

  const getThemeIcon = () => {
    if (theme === 'pond-light') return <Sun className="w-3.5 h-3.5 text-[var(--accent-tube)]" />;
    if (theme === 'pond-dark') return <Moon className="w-3.5 h-3.5 text-[var(--accent-pond)]" />;
    return <Leaf className="w-3.5 h-3.5 text-[var(--accent-pond)]" />;
  };

  return (
    <header className="sticky top-0 z-40 w-full bg-[var(--bg-card)] border-b border-[var(--border-color)] text-[var(--text-primary)] shadow-xs select-none">
      <div className="w-full max-w-[1720px] 2xl:max-w-[1920px] mx-auto px-3 sm:px-5 lg:px-6 py-2 sm:py-2.5 flex items-center justify-between gap-2 sm:gap-3">
        
        {/* 1. Left: Brand Title with PRO badge */}
        <div 
          onClick={() => {
            if (onExitToLanding) {
              onExitToLanding();
            }
          }}
          className={`flex items-center gap-2 sm:gap-2.5 shrink-0 min-w-0 ${onExitToLanding ? 'cursor-pointer hover:opacity-85 transition-opacity' : ''}`}
          title={onExitToLanding ? 'Return to Landing Screen' : undefined}
        >
          <div className="relative flex items-center justify-center w-7 sm:w-8 h-7 sm:h-8 rounded-xl bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] border border-[var(--border-color)] shrink-0">
            <Bot className="w-4 h-4 sm:w-4.5 sm:h-4.5" />
            <span className="absolute -bottom-0.5 -right-0.5 w-2 h-2 bg-[var(--accent-pond)] rounded-full border-2 border-[var(--bg-card)]" />
          </div>

          <div className="flex flex-col justify-center min-w-0">
            <div className="flex items-center gap-1.5">
              <h1 className="font-bold text-xs sm:text-sm md:text-base text-[var(--text-primary)] tracking-tight whitespace-nowrap leading-none">
                Vision Monitor
              </h1>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[9px] sm:text-[9.5px] font-bold tracking-wider bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] border border-[var(--border-color)] uppercase">
                V4.2 PRO
              </span>
            </div>
            <p className="text-[10px] sm:text-[11px] text-[var(--text-secondary)] font-medium whitespace-nowrap leading-tight mt-0.5">
              Live detection &amp; alerts
            </p>
          </div>
        </div>

        {/* 2. Right: Status Badges + Utilities */}
        <div className="flex items-center gap-1 sm:gap-2 shrink-0">
          
          {/* Status Badges on the right */}
          <div className="hidden md:flex items-center gap-2 mr-1">
            {/* Backend Indicator */}
            <button
              onClick={() => {
                checkBackendHealth();
              }}
              title="FastAPI Backend Status (Click to test connection)"
              className="inline-flex flex-row items-center gap-1.5 h-8 px-3 rounded-xl bg-[var(--btn-secondary-bg)] border border-[var(--btn-secondary-border)] text-xs font-semibold text-[var(--btn-secondary-text)] hover:bg-[var(--btn-secondary-hover)] cursor-pointer whitespace-nowrap"
            >
              <span className={`w-2 h-2 rounded-full shrink-0 ${backendConnected ? 'bg-[var(--status-normal-text)] animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-[var(--status-anomaly-text)]'}`} />
              <span className="text-[var(--text-secondary)] font-medium">Backend:</span>
              <span className={`font-bold ${backendConnected ? 'text-[var(--status-normal-text)]' : 'text-[var(--status-anomaly-text)]'}`}>
                {backendConnected ? 'Connected' : 'Offline'}
              </span>
            </button>

            {/* Camera Status */}
            <button
              onClick={() => {
                onOpenSettings();
              }}
              title="Camera Hardware Status (Click for settings)"
              className="inline-flex flex-row items-center gap-1.5 h-8 px-3 rounded-xl bg-[var(--btn-secondary-bg)] border border-[var(--btn-secondary-border)] text-xs font-semibold text-[var(--btn-secondary-text)] hover:bg-[var(--btn-secondary-hover)] cursor-pointer whitespace-nowrap"
            >
              <span className={`w-2 h-2 rounded-full shrink-0 ${cameraConfig.connected ? 'bg-[var(--status-normal-text)] animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-[var(--status-anomaly-text)]'}`} />
              <Video className="w-3.5 h-3.5 text-[var(--text-primary)] shrink-0" />
              <span className="text-[var(--text-secondary)] font-medium">Cam:</span>
              <span className={`font-bold ${cameraConfig.connected ? 'text-[var(--status-normal-text)]' : 'text-[var(--status-anomaly-text)]'}`}>
                {cameraConfig.connected ? 'Ready' : 'Offline'}
              </span>
            </button>
          </div>

          {/* Theme Switcher Button */}
          <IconButton
            onClick={cycleTheme}
            aria-label={
              theme === 'pond-light'
                ? 'Current: Light Theme (Click for Dark)'
                : theme === 'pond-dark'
                ? 'Current: Dark Theme (Click for Nature)'
                : 'Current: Nature Theme (Click for Light)'
            }
            icon={getThemeIcon()}
            size="md"
          />

          {/* Camera Settings Button */}
          <IconButton
            onClick={() => {
              onOpenSettings();
            }}
            aria-label="System Settings"
            icon={<Settings className="w-3.5 h-3.5 text-[var(--accent-pond)]" />}
            size="md"
          />

          {/* Help Guide Button */}
          <IconButton
            onClick={() => {
              onOpenHelp();
            }}
            aria-label="User Guide"
            icon={<HelpCircle className="w-3.5 h-3.5 text-[var(--accent-pond)]" />}
            size="md"
          />
        </div>

      </div>
    </header>
  );
};
