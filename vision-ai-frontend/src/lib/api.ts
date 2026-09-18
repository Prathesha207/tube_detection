import axios from "axios";

/**
 * Dynamically resolves the Backend API URL:
 * 1. Environment Variable `VITE_API_BASE_URL` (if set in .env or Cloudflare build).
 * 2. If in browser: dynamically matches current hostname (localhost, LAN IP, or Cloudflare domain) on port 8003.
 * 3. Fallback default: http://127.0.0.1:8003
 */
export function getApiBaseUrl(): string {
  // 1. User saved preference in localStorage (if set)
  if (typeof window !== 'undefined') {
    try {
      const saved = localStorage.getItem('vision_backend_url');
      if (saved && saved.trim() !== '') return saved.replace(/\/+$/, '');
    } catch { }
  }

  // 2. Explicit env override (e.g. from .env file or Cloudflare build)
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

  // 3. Dynamic match of current browser host (e.g. http://localhost:5173 or LAN IP)
  if (typeof window !== 'undefined' && window.location?.hostname && window.location.protocol.startsWith('http')) {
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    const host = window.location.hostname;
    return `${protocol}//${host}:8000`;
  }

  // 4. Fallback default
  return "http://127.0.0.1:8000";
}

export const API_BASE_URL = getApiBaseUrl();

// BUGFIX: video_router.py's /video/upload runs an ffmpeg transcode subprocess
// with a 180s budget (see `timeout=180` in run_transcode()). A 15s axios
// timeout here means any upload that needs more than 15s to transcode gets
// aborted client-side while the backend keeps working in the background and
// eventually finishes into a session the UI has already given up on -- which
// looks identical to "stuck" or "offline" to the user. Raised to comfortably
// exceed the backend's own budget. If a specific call genuinely needs to fail
// fast (e.g. a lightweight status check), override per-call with
// `api.get(url, { timeout: 5000 })` rather than lowering this default.
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 200000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Dynamic interceptor: ensures API calls always resolve to current host/tunnel
api.interceptors.request.use((config) => {
  config.baseURL = getApiBaseUrl();
  return config;
});

export default api;