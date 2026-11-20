import axios from "axios";

let activeApiBaseUrl: string | null = null;

export function setActiveApiBaseUrl(url: string) {
  const clean = url.replace(/\/+$/, '');
  activeApiBaseUrl = clean;
  if (typeof window !== 'undefined') {
    try {
      sessionStorage.setItem('vision_active_backend_url', clean);
      localStorage.setItem('vision_backend_url', clean);
    } catch { }
  }
}

/**
 * Dynamically resolves the Backend API URL:
 * 1. Active discovered URL (in-memory verified).
 * 2. Query param override (?backendUrl=... or ?port=...) from URL / Electron loadFile.
 * 3. Session storage active URL ('vision_active_backend_url').
 * 4. User saved preference in localStorage ('vision_backend_url').
 * 5. Environment Variable `VITE_API_BASE_URL`.
 * 6. Dynamic match of current browser hostname on port 8000.
 * 7. Fallback default: http://127.0.0.1:8000
 */
export function getApiBaseUrl(): string {
  // 1. In-memory verified active URL
  if (activeApiBaseUrl) {
    return activeApiBaseUrl;
  }

  if (typeof window !== 'undefined') {
    // 2. Query param override (e.g. Electron passes ?backendUrl=http://127.0.0.1:8002 or ?port=8002)
    try {
      const search = window.location.search || (window.location.hash.includes('?') ? window.location.hash.split('?')[1] : '');
      const params = new URLSearchParams(search);
      const qUrl = params.get('backendUrl');
      if (qUrl && qUrl.trim() !== '') {
        const clean = qUrl.trim().replace(/\/+$/, '');
        activeApiBaseUrl = clean;
        try { sessionStorage.setItem('vision_active_backend_url', clean); } catch { }
        return clean;
      }
      const qPort = params.get('port');
      if (qPort && /^\d+$/.test(qPort.trim())) {
        const clean = `http://127.0.0.1:${qPort.trim()}`;
        activeApiBaseUrl = clean;
        try { sessionStorage.setItem('vision_active_backend_url', clean); } catch { }
        return clean;
      }
    } catch { }

    // 3. Session storage active URL
    try {
      const sessionUrl = sessionStorage.getItem('vision_active_backend_url');
      if (sessionUrl && sessionUrl.trim() !== '') {
        activeApiBaseUrl = sessionUrl.replace(/\/+$/, '');
        return activeApiBaseUrl;
      }
    } catch { }

    // 4. User saved preference in localStorage (if set)
    try {
      const saved = localStorage.getItem('vision_backend_url');
      if (saved && saved.trim() !== '') {
        return saved.replace(/\/+$/, '');
      }
    } catch { }
  }

  // 5. Explicit env override (e.g. from .env file or Cloudflare build)
  const envUrl = (import.meta as any).env?.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    const trimmed = envUrl.replace(/\/+$/, '');
    if (typeof window !== 'undefined' && window.location?.hostname && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      try {
        const parsed = new URL(trimmed);
        if (parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1') {
          return `${window.location.protocol}//${window.location.hostname}:${parsed.port || '8000'}`;
        }
      } catch { }
    }
    return trimmed;
  }

  // 6. Dynamic match of current browser host (e.g. http://localhost:5173 or LAN IP)
  if (typeof window !== 'undefined' && window.location?.hostname && window.location.protocol.startsWith('http')) {
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const host = window.location.hostname;
    return `${protocol}//${host}:8000`;
  }

  // 7. Fallback default
  return "http://127.0.0.1:8000";
}

/**
 * Probes candidate ports (8000, 8002, 8001, 8003, 8004, 8005, 8080) IN PARALLEL
 * against /health to find where the Vision Monitor FastAPI backend is actually listening.
 * Verifies that the response payload contains Vision Monitor status ('ready' | 'starting').
 */
export async function probeBackendHealth(candidateUrls?: string[]): Promise<string | null> {
  const host = (typeof window !== 'undefined' && window.location?.hostname && window.location.hostname.trim() !== '')
    ? window.location.hostname
    : '127.0.0.1';
  const protocol = (typeof window !== 'undefined' && window.location?.protocol === 'https:') ? 'https:' : 'http:';

  // 1. Direct query to Electron main process if available
  let electronUrl: string | null = null;
  if (typeof window !== 'undefined' && (window as any).electronAPI?.getBackendUrl) {
    try {
      electronUrl = await (window as any).electronAPI.getBackendUrl();
    } catch { }
  }

  const candidatePorts = [8000, 8002, 8001, 8003, 8004, 8005, 8080];
  const defaultCandidates: string[] = [];

  if (electronUrl && typeof electronUrl === 'string' && electronUrl.trim() !== '') {
    defaultCandidates.push(electronUrl.trim());
  }

  const current = getApiBaseUrl();
  if (current) defaultCandidates.push(current);

  for (const p of candidatePorts) {
    defaultCandidates.push(`${protocol}//${host}:${p}`);
    defaultCandidates.push(`http://127.0.0.1:${p}`);
    defaultCandidates.push(`http://localhost:${p}`);
  }

  const candidates = candidateUrls || defaultCandidates;
  const uniqueCandidates = Array.from(new Set(candidates.map((u) => u.replace(/\/+$/, ''))));

  // Parallel probe helper: probes a single candidate URL with short timeout
  const probeOne = async (url: string): Promise<{ url: string; ready: boolean } | null> => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);
      const res = await fetch(`${url}/health`, {
        method: 'GET',
        cache: 'no-store',
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (res.ok || res.status === 503) {
        const data = await res.json().catch(() => null);
        // Validate that this is the Vision Monitor backend, not vLLM or generic server
        if (data && (data.status === 'ready' || data.status === 'starting')) {
          return { url, ready: data.status === 'ready' };
        }
      }
    } catch {
      // Offline / error
    }
    return null;
  };

  // Run all candidate probes concurrently for instant detection
  const probeResults = await Promise.all(uniqueCandidates.map(probeOne));
  const validHits = probeResults.filter((r): r is { url: string; ready: boolean } => r !== null);

  if (validHits.length > 0) {
    // Prefer any candidate that is already 'ready' over one that is still 'starting'
    const best = validHits.find((h) => h.ready) || validHits[0];
    setActiveApiBaseUrl(best.url);
    return best.url;
  }

  return null;
}

// Auto-listen to Electron backend URL update event if available
if (typeof window !== 'undefined' && (window as any).electronAPI?.onBackendUrlUpdated) {
  try {
    (window as any).electronAPI.onBackendUrlUpdated((url: string) => {
      if (url && typeof url === 'string') {
        setActiveApiBaseUrl(url);
      }
    });
  } catch { }
}

export const API_BASE_URL = getApiBaseUrl();

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 200000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Dynamic interceptor: ensures API calls always resolve to current live host/port
api.interceptors.request.use((config) => {
  config.baseURL = getApiBaseUrl();
  return config;
});

export default api;
