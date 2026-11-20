import axios from "axios";

let activeApiBaseUrl: string | null = null;

export function setActiveApiBaseUrl(url: string) {
  const clean = url.replace(/\/+$/, '');
  activeApiBaseUrl = clean;
  if (typeof window !== 'undefined') {
    try {
      sessionStorage.setItem('vision_active_backend_url', clean);
    } catch { }
  }
}

/**
 * Dynamically resolves the Backend API URL:
 * 1. Active discovered URL (verified responsive during session or startup).
 * 2. User saved preference in localStorage ('vision_backend_url').
 * 3. Environment Variable `VITE_API_BASE_URL` (if set in .env or Cloudflare build).
 * 4. Dynamic match of current browser hostname on port 8000 / 8002.
 * 5. Fallback default: http://127.0.0.1:8000
 */
export function getApiBaseUrl(): string {
  // 1. In-memory verified active URL
  if (activeApiBaseUrl) {
    return activeApiBaseUrl;
  }

  if (typeof window !== 'undefined') {
    // 2. Session storage active URL
    try {
      const sessionUrl = sessionStorage.getItem('vision_active_backend_url');
      if (sessionUrl && sessionUrl.trim() !== '') {
        activeApiBaseUrl = sessionUrl.replace(/\/+$/, '');
        return activeApiBaseUrl;
      }
    } catch { }

    // 3. User saved preference in localStorage (if set)
    try {
      const saved = localStorage.getItem('vision_backend_url');
      if (saved && saved.trim() !== '') return saved.replace(/\/+$/, '');
    } catch { }
  }

  // 4. Explicit env override (e.g. from .env file or Cloudflare build)
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

  // 5. Dynamic match of current browser host (e.g. http://localhost:5173 or LAN IP)
  if (typeof window !== 'undefined' && window.location?.hostname && window.location.protocol.startsWith('http')) {
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const host = window.location.hostname;
    return `${protocol}//${host}:8000`;
  }

  // 6. Fallback default
  return "http://127.0.0.1:8000";
}

/**
 * Checks candidate ports (8000, 8002, 8001, 8003, 8080) against /health to find
 * where the Vision Monitor FastAPI backend is actually listening.
 * Verifies that the response payload contains Vision Monitor status ('ready' | 'starting').
 */
export async function probeBackendHealth(candidateUrls?: string[]): Promise<string | null> {
  const host = (typeof window !== 'undefined' && window.location?.hostname) ? window.location.hostname : '127.0.0.1';
  const protocol = (typeof window !== 'undefined' && window.location?.protocol === 'https:') ? 'https:' : 'http:';

  const defaultCandidates: string[] = [
    getApiBaseUrl(),
    `${protocol}//${host}:8002`,
    `${protocol}//${host}:8000`,
    `${protocol}//${host}:8001`,
    `${protocol}//${host}:8003`,
    'http://127.0.0.1:8002',
    'http://127.0.0.1:8000',
  ];

  const candidates = candidateUrls || defaultCandidates;
  const uniqueCandidates = Array.from(new Set(candidates.map((u) => u.replace(/\/+$/, ''))));

  for (const url of uniqueCandidates) {
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
          setActiveApiBaseUrl(url);
          return url;
        }
      }
    } catch {
      // Continue to next port
    }
  }

  return null;
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
