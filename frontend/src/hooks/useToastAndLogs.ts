import { useState, useCallback } from 'react';
import type { LogEntry } from '../types';
import { MAX_LOG_ENTRIES, TOAST_AUTO_DISMISS_MS } from '../utils/constants';

export type ToastMessage = { type: 'error' | 'success' | 'info'; message: string };

export function useToastAndLogs() {
  // Toast notification state
  const [toast, setToast] = useState<ToastMessage | null>(null);

  // Metrics & Logs State
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const showToast = useCallback((type: 'error' | 'success' | 'info', message: string) => {
    setToast({ type, message });
    setTimeout(() => {
      setToast((prev) => (prev?.message === message ? null : prev));
    }, TOAST_AUTO_DISMISS_MS);
  }, []);

  // Helper to add log entry
  const addLog = useCallback((message: string, level: LogEntry['level'] = 'info') => {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const newEntry: LogEntry = {
      id: `log-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`,
      timestamp: timeStr,
      message,
      level,
    };
    setLogs((prev) => [...prev.slice(-MAX_LOG_ENTRIES), newEntry]);
  }, []);

  return {
    toast,
    setToast,
    showToast,
    logs,
    setLogs,
    addLog,
  };
}
