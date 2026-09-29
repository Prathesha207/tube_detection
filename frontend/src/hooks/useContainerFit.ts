import { useState, useEffect, useMemo } from 'react';

export const useContainerFit = (
  containerRef: React.RefObject<HTMLDivElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  videoAspect: number | null,
  videoDimensions: { width: number; height: number } | null | undefined,
  isCameraSource: boolean
) => {
  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  useEffect(() => {
    let rafId: number | null = null;
    let observer: ResizeObserver | null = null;
    
    if (containerRef.current) {
      observer = new ResizeObserver((entries) => {
        if (rafId !== null) cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => {
          for (const entry of entries) {
            if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
              setContainerSize({
                width: entry.contentRect.width,
                height: entry.contentRect.height,
              });

              if (canvasRef.current) {
                const dpr = Math.min(window.devicePixelRatio || 1, 2);
                const targetW = Math.round(entry.contentRect.width * dpr);
                const targetH = Math.round(entry.contentRect.height * dpr);
                if (canvasRef.current.width !== targetW || canvasRef.current.height !== targetH) {
                  canvasRef.current.width = targetW;
                  canvasRef.current.height = targetH;
                }
              }
            }
          }
        });
      });
      observer.observe(containerRef.current);
    }

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (observer) observer.disconnect();
    };
  }, [containerRef, canvasRef]);

  const fittedRect = useMemo(() => {
    if (!containerSize.width || !containerSize.height) {
      return { width: '100%', height: '100%' };
    }
    const targetAspect = 
      (videoDimensions?.width && videoDimensions?.height 
        ? videoDimensions.width / videoDimensions.height 
        : null) || 
      videoAspect || 
      (16 / 9);
    const containerAspect = containerSize.width / containerSize.height;
    
    let w: number;
    let h: number;
    if (containerAspect > targetAspect) {
      // Container is wider -> fit to container height
      h = containerSize.height;
      w = Math.floor(containerSize.height * targetAspect);
    } else {
      // Container is taller -> fit to container width
      w = containerSize.width;
      h = Math.floor(containerSize.width / targetAspect);
    }

    return {
      width: `${w}px`,
      height: `${h}px`,
    };
  }, [containerSize, videoAspect, videoDimensions, isCameraSource]);

  // The canvas lives inside the fitted video viewport, which can change size when
  // metadata arrives without causing the outer container to resize. Keep its
  // bitmap dimensions in sync so coordinates match the visible frame.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
  }, [canvasRef, fittedRect]);

  return { containerSize, fittedRect };
};
