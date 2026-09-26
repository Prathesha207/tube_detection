import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

export interface CanvasZoomState {
  zoom: number;
  pan: { x: number; y: number };
  isPanning: boolean;
  zoomIn: () => void;
  zoomOut: () => void;
  setZoomLevel: (level: number) => void;
  resetZoom: () => void;
  viewportRef: React.RefObject<HTMLDivElement | null>;
  transformStyle: React.CSSProperties;
  handleMouseDown: (e: React.MouseEvent) => void;
  handleDoubleClick: (e: React.MouseEvent) => void;
  cursorStyle: string;
  hasMoved: boolean;
}

const MIN_ZOOM = 1.0;
const MAX_ZOOM = 5.0;
const ZOOM_STEP = 0.25;

export const useCanvasZoom = (enabled: boolean = true) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const dragStartRef = useRef<{ x: number; y: number; panX: number; panY: number }>({
    x: 0,
    y: 0,
    panX: 0,
    panY: 0,
  });
  const hasMovedRef = useRef<boolean>(false);

  // Helper to clamp pan based on container dimensions and zoom
  const clampPan = useCallback((x: number, y: number, currentZoom: number) => {
    if (currentZoom <= 1.0 || !viewportRef.current) {
      return { x: 0, y: 0 };
    }
    const rect = viewportRef.current.getBoundingClientRect();
    const maxPanX = Math.max(0, (rect.width * (currentZoom - 1)) / 2 + 40);
    const maxPanY = Math.max(0, (rect.height * (currentZoom - 1)) / 2 + 40);
    return {
      x: Math.max(-maxPanX, Math.min(maxPanX, x)),
      y: Math.max(-maxPanY, Math.min(maxPanY, y)),
    };
  }, []);

  const setZoomLevel = useCallback(
    (newZoom: number, focalPoint?: { x: number; y: number }) => {
      setZoom((prevZoom) => {
        const targetZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Math.round(newZoom * 100) / 100));
        if (targetZoom <= 1.0) {
          setPan({ x: 0, y: 0 });
          return 1.0;
        }

        setPan((prevPan) => {
          if (focalPoint && viewportRef.current) {
            const rect = viewportRef.current.getBoundingClientRect();
            // Cursor position relative to center of viewport
            const cx = focalPoint.x - (rect.left + rect.width / 2);
            const cy = focalPoint.y - (rect.top + rect.height / 2);

            const scaleRatio = targetZoom / prevZoom;
            const newPanX = cx - (cx - prevPan.x) * scaleRatio;
            const newPanY = cy - (cy - prevPan.y) * scaleRatio;
            return clampPan(newPanX, newPanY, targetZoom);
          }
          return clampPan(prevPan.x, prevPan.y, targetZoom);
        });

        return targetZoom;
      });
    },
    [clampPan]
  );

  const zoomIn = useCallback(() => {
    setZoomLevel(zoom + (zoom >= 2.0 ? 0.5 : ZOOM_STEP));
  }, [zoom, setZoomLevel]);

  const zoomOut = useCallback(() => {
    setZoomLevel(zoom - (zoom > 2.0 ? 0.5 : ZOOM_STEP));
  }, [zoom, setZoomLevel]);

  const resetZoom = useCallback(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, []);

  // Handle native wheel zoom (with non-passive listener to prevent page scrolling)
  useEffect(() => {
    const el = viewportRef.current;
    if (!el || !enabled) return;

    const handleWheel = (e: WheelEvent) => {
      // Zoom with wheel
      e.preventDefault();
      e.stopPropagation();

      const delta = -e.deltaY;
      const factor = delta > 0 ? 1.15 : 0.85;
      const targetZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * factor));

      setZoomLevel(targetZoom, { x: e.clientX, y: e.clientY });
    };

    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', handleWheel);
    };
  }, [zoom, enabled, setZoomLevel]);

  // Mouse pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (zoom <= 1.0 || e.button !== 0) return;
      // Do not initiate pan drag if clicking an interactive button or badge
      if ((e.target as HTMLElement).closest('button, [role="button"]')) return;

      setIsPanning(true);
      hasMovedRef.current = false;
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        panX: pan.x,
        panY: pan.y,
      };
    },
    [zoom, pan]
  );

  useEffect(() => {
    if (!isPanning) return;

    const handleMouseMove = (e: MouseEvent) => {
      const dx = e.clientX - dragStartRef.current.x;
      const dy = e.clientY - dragStartRef.current.y;

      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        hasMovedRef.current = true;
      }

      const newPanX = dragStartRef.current.panX + dx;
      const newPanY = dragStartRef.current.panY + dy;

      setPan(clampPan(newPanX, newPanY, zoom));
    };

    const handleMouseUp = () => {
      setIsPanning(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isPanning, zoom, clampPan]);

  // Double click toggles between 1x and 2x zoom
  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      // If clicking button, skip
      if ((e.target as HTMLElement).closest('button, [role="button"]')) return;

      if (zoom > 1.05) {
        resetZoom();
      } else {
        setZoomLevel(2.0, { x: e.clientX, y: e.clientY });
      }
    },
    [zoom, resetZoom, setZoomLevel]
  );

  // Keyboard shortcuts (+ / - / 0)
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid firing when user is typing in an input
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;

      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        zoomIn();
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        zoomOut();
      } else if (e.key === '0') {
        e.preventDefault();
        resetZoom();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [enabled, zoomIn, zoomOut, resetZoom]);

  const transformStyle = useMemo<React.CSSProperties>(() => {
    return {
      transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
      transformOrigin: 'center center',
      transition: isPanning ? 'none' : 'transform 0.15s cubic-bezier(0.2, 0, 0, 1)',
      willChange: 'transform',
    };
  }, [pan.x, pan.y, zoom, isPanning]);

  const cursorStyle = useMemo(() => {
    if (zoom <= 1.0) return 'default';
    return isPanning ? 'grabbing' : 'grab';
  }, [zoom, isPanning]);

  return {
    zoom,
    pan,
    isPanning,
    zoomIn,
    zoomOut,
    setZoomLevel,
    resetZoom,
    viewportRef,
    transformStyle,
    handleMouseDown,
    handleDoubleClick,
    cursorStyle,
    hasMoved: hasMovedRef.current,
  };
};
