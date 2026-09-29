import { useState, useEffect } from 'react';
import { getApiBaseUrl } from '../lib/api';
import { HEALTH_POLL_INTERVAL_MS } from '../utils/constants';

export const INITIALIZED_STORAGE_KEY = 'vision_monitor_initialized';
// Session key: survives Ctrl+R refresh but clears when the window/tab is actually closed
const SESSION_INITIALIZED_KEY = 'vision_monitor_session_initialized';

export function useBackendHealth() {
  // Read from sessionStorage on first mount so page refreshes don't drop back to LandingScreen.
  // sessionStorage is cleared automatically when the browser tab / Electron window closes,
  // so a genuine new launch always shows LandingScreen.
  const [systemInitialized, setSystemInitializedState] = useState<boolean>(() => {
    try {
      return sessionStorage.getItem(SESSION_INITIALIZED_KEY) === 'true';
    } catch {
      return false;
    }
  });
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);

  // Keep sessionStorage in sync whenever the state changes
  const setSystemInitialized = (value: boolean | ((prev: boolean) => boolean)) => {
    setSystemInitializedState(prev => {
      const next = typeof value === 'function' ? value(prev) : value;
      try {
        if (next) {
          sessionStorage.setItem(SESSION_INITIALIZED_KEY, 'true');
        } else {
          sessionStorage.removeItem(SESSION_INITIALIZED_KEY);
        }
      } catch { }
      return next;
    });
  };

  // Background health polling to keep status up to date
  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const res = await fetch(`${getApiBaseUrl()}/health`, { method: 'GET', cache: 'no-store' });
        if (isMounted) {
          setIsBackendConnected(res.ok);
        }
      } catch {
        if (isMounted) setIsBackendConnected(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, HEALTH_POLL_INTERVAL_MS);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleInitializeSystem = () => {
    setSystemInitialized(true);
  };

  return {
    systemInitialized,
    setSystemInitialized,
    isBackendConnected,
    setIsBackendConnected,
    handleInitializeSystem,
  };
}
