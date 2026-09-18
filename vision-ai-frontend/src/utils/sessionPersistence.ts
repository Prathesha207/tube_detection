/**
 * Persists and restores the active inference session across page refreshes.
 *
 * Only sessionStorage is used — it survives Ctrl+R but is cleared when the
 * browser tab / Electron window is actually closed, so a genuine new launch
 * always starts fresh.
 */

const KEY = 'vision_monitor_session_state';

export interface PersistedSessionState {
  sourceType: string;
  videoSessionId: string | null;
  customVideoUrl: string | undefined;
  customVideoName: string | undefined;
  expectedDucks?: number;
  selectedDuckId?: string | null;
  videoDimensions?: { width: number; height: number } | null;
}

export function saveSessionState(state: Partial<PersistedSessionState>) {
  try {
    const existing = loadSessionState() ?? ({} as PersistedSessionState);
    const merged = { ...existing, ...state };
    sessionStorage.setItem(KEY, JSON.stringify(merged));
  } catch {
    // sessionStorage quota exceeded or other error
  }
}

export function loadSessionState(): PersistedSessionState | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedSessionState;
  } catch {
    return null;
  }
}

export function clearSessionState() {
  try {
    sessionStorage.removeItem(KEY);
  } catch { }
}
