import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { screenToFramePoint, validateRoi, Point } from '../../utils/roiUtils';
import type { RoiToolMode } from './RoiLeftToolbar';
import { AlertCircle } from 'lucide-react';

interface RoiDrawingOverlayProps {
  points: Point[];
  onChange: (points: Point[]) => void;
  activeTool: RoiToolMode;
  onSelectTool: (tool: RoiToolMode) => void;
  frameWidth: number;
  frameHeight: number;
}

export const RoiDrawingOverlay: React.FC<RoiDrawingOverlayProps> = ({
  points,
  onChange,
  activeTool,
  onSelectTool,
  frameWidth,
  frameHeight,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Safe frame dimensions
  const fw = frameWidth > 0 ? frameWidth : 1920;
  const fh = frameHeight > 0 ? frameHeight : 1080;

  // Interaction States
  const [dragState, setDragState] = useState<{
    type: 'none' | 'rect_drag' | 'vertex_drag' | 'polygon_pan';
    startPoint?: Point;
    vertexIndex?: number;
    initialPoints?: Point[];
  }>({ type: 'none' });

  const [rectCurrentPoint, setRectCurrentPoint] = useState<Point | null>(null);
  const [polygonDraft, setPolygonDraft] = useState<Point[]>([]);
  const [cursorPos, setCursorPos] = useState<Point | null>(null);
  const [hoveredVertexIndex, setHoveredVertexIndex] = useState<number | null>(null);

  // Handle radius in frame coordinates
  const handleRadius = Math.max(7, Math.round(fw / 140));

  // Check validation state of current points
  const validation = useMemo(() => validateRoi(points), [points]);

  // Clean draft if tool changes
  useEffect(() => {
    if (activeTool !== 'polygon') {
      setPolygonDraft([]);
    }
    if (activeTool !== 'rectangle') {
      setRectCurrentPoint(null);
    }
    setDragState({ type: 'none' });
  }, [activeTool]);

  // Keyboard shortcuts (Escape to cancel draft, Delete to clear)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (polygonDraft.length > 0) {
          setPolygonDraft([]);
        } else if (activeTool !== 'pointer') {
          onSelectTool('pointer');
        }
      }
      if (e.key === 'r' || e.key === 'R') {
        onSelectTool('rectangle');
      }
      if (e.key === 'p' || e.key === 'P') {
        onSelectTool('polygon');
      }
      if (e.key === 'v' || e.key === 'V') {
        onSelectTool('pointer');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [polygonDraft.length, activeTool, onSelectTool]);

  // Helper to get frame point from client event with clamping
  const getFramePt = useCallback(
    (e: React.PointerEvent<HTMLDivElement>): Point => {
      if (!containerRef.current) return [0, 0];
      const rect = containerRef.current.getBoundingClientRect();
      return screenToFramePoint(e.clientX, e.clientY, rect, fw, fh);
    },
    [fw, fh]
  );

  // Distance between two points
  const dist = (p1: Point, p2: Point) => {
    const dx = p1[0] - p2[0];
    const dy = p1[1] - p2[1];
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Find nearest vertex within threshold
  const findVertexIndex = useCallback(
    (pt: Point, thresholdPx: number = handleRadius * 1.5): number | null => {
      let closestIdx: number | null = null;
      let minDist = thresholdPx;
      points.forEach((p, idx) => {
        const d = dist(pt, p);
        if (d < minDist) {
          minDist = d;
          closestIdx = idx;
        }
      });
      return closestIdx;
    },
    [points, handleRadius]
  );

  // ---------------- Pointer Events ----------------
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const pt = getFramePt(e);

    if (activeTool === 'rectangle') {
      setDragState({ type: 'rect_drag', startPoint: pt });
      setRectCurrentPoint(pt);
      return;
    }

    if (activeTool === 'polygon') {
      // If clicking near the first point and we have at least 3 points, close polygon
      if (polygonDraft.length >= 3 && dist(pt, polygonDraft[0]) <= handleRadius * 2) {
        onChange(polygonDraft);
        setPolygonDraft([]);
        onSelectTool('pointer');
        return;
      }
      setPolygonDraft((prev) => [...prev, pt]);
      return;
    }

    if (activeTool === 'pointer') {
      // Check if clicking on an existing vertex
      const vIdx = findVertexIndex(pt);
      if (vIdx !== null) {
        setDragState({
          type: 'vertex_drag',
          vertexIndex: vIdx,
          initialPoints: points,
        });
        return;
      }

      // Check if clicking inside polygon to pan it
      if (points.length >= 3) {
        setDragState({
          type: 'polygon_pan',
          startPoint: pt,
          initialPoints: [...points],
        });
      }
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const pt = getFramePt(e);
    setCursorPos(pt);

    if (activeTool === 'pointer' && dragState.type === 'none') {
      const vIdx = findVertexIndex(pt);
      setHoveredVertexIndex(vIdx);
    }

    if (dragState.type === 'rect_drag' && dragState.startPoint) {
      setRectCurrentPoint(pt);
    } else if (dragState.type === 'vertex_drag' && dragState.vertexIndex !== undefined && dragState.initialPoints) {
      const newPts = [...points];
      newPts[dragState.vertexIndex] = pt;
      onChange(newPts);
    } else if (dragState.type === 'polygon_pan' && dragState.startPoint && dragState.initialPoints) {
      const dx = pt[0] - dragState.startPoint[0];
      const dy = pt[1] - dragState.startPoint[1];
      const shifted = dragState.initialPoints.map(([x, y]) => [
        Math.max(0, Math.min(fw, x + dx)),
        Math.max(0, Math.min(fh, y + dy)),
      ] as Point);
      onChange(shifted);
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture already released
    }

    if (dragState.type === 'rect_drag' && dragState.startPoint && rectCurrentPoint) {
      const minX = Math.min(dragState.startPoint[0], rectCurrentPoint[0]);
      const maxX = Math.max(dragState.startPoint[0], rectCurrentPoint[0]);
      const minY = Math.min(dragState.startPoint[1], rectCurrentPoint[1]);
      const maxY = Math.max(dragState.startPoint[1], rectCurrentPoint[1]);

      // Minimum 10x10 span
      if (maxX - minX >= 10 && maxY - minY >= 10) {
        const rectPts: Point[] = [
          [minX, minY],
          [maxX, minY],
          [maxX, maxY],
          [minX, maxY],
        ];
        onChange(rectPts);
        onSelectTool('pointer');
      }
      setRectCurrentPoint(null);
    }

    setDragState({ type: 'none' });
  };

  const handleDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation();
    if (activeTool === 'polygon' && polygonDraft.length >= 3) {
      onChange(polygonDraft);
      setPolygonDraft([]);
      onSelectTool('pointer');
    }
  };

  // Convert points array to SVG polygon points string
  const pointsToString = (pts: Point[]) => pts.map(([x, y]) => `${x},${y}`).join(' ');

  // Compute live rectangle preview coordinates
  const rectPreview = useMemo(() => {
    if (dragState.type !== 'rect_drag' || !dragState.startPoint || !rectCurrentPoint) {
      return null;
    }
    const x = Math.min(dragState.startPoint[0], rectCurrentPoint[0]);
    const y = Math.min(dragState.startPoint[1], rectCurrentPoint[1]);
    const w = Math.max(1, Math.abs(rectCurrentPoint[0] - dragState.startPoint[0]));
    const h = Math.max(1, Math.abs(rectCurrentPoint[1] - dragState.startPoint[1]));
    return { x, y, w, h };
  }, [dragState, rectCurrentPoint]);

  // Determine cursor style
  const cursorStyle = useMemo(() => {
    if (activeTool === 'rectangle') return 'crosshair';
    if (activeTool === 'polygon') return 'crosshair';
    if (hoveredVertexIndex !== null || dragState.type === 'vertex_drag') return 'grab';
    if (dragState.type === 'polygon_pan') return 'grabbing';
    return 'default';
  }, [activeTool, hoveredVertexIndex, dragState.type]);

  const isInvalid = !validation.valid;
  const strokeColor = isInvalid ? '#f43f5e' : '#06b6d4';

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onDoubleClick={handleDoubleClick}
      style={{ cursor: cursorStyle }}
      className="absolute inset-0 z-20 overflow-hidden pointer-events-auto select-none"
    >
      <svg
        className="w-full h-full block"
        viewBox={`0 0 ${fw} ${fh}`}
        preserveAspectRatio="none"
      >
        {/* Main Saved / Active ROI Polygon (Clean outline only, no color tint or dimming) */}
        {points.length >= 3 && (
          <polygon
            points={pointsToString(points)}
            fill="none"
            stroke={strokeColor}
            strokeWidth="2"
            strokeDasharray="6 4"
            strokeLinejoin="round"
            className="transition-colors duration-150"
          />
        )}

        {/* 3. Drag Handles on Vertices */}
        {activeTool === 'pointer' &&
          points.map(([x, y], idx) => {
            const isHovered = hoveredVertexIndex === idx || (dragState.type === 'vertex_drag' && dragState.vertexIndex === idx);
            return (
              <g key={`handle-${idx}`}>
                <circle
                  cx={x}
                  cy={y}
                  r={isHovered ? handleRadius * 1.3 : handleRadius}
                  fill={isHovered ? '#ffffff' : strokeColor}
                  stroke="#0f172a"
                  strokeWidth="2.5"
                  className="cursor-grab active:cursor-grabbing transition-transform"
                />
                <circle
                  cx={x}
                  cy={y}
                  r={handleRadius * 0.4}
                  fill={isHovered ? strokeColor : '#ffffff'}
                />
              </g>
            );
          })}

        {/* 4. Live Rectangle Drag Preview */}
        {rectPreview && (
          <rect
            x={rectPreview.x}
            y={rectPreview.y}
            width={rectPreview.w}
            height={rectPreview.h}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="2"
            strokeDasharray="6 4"
            className="pointer-events-none"
          />
        )}

        {/* 5. Live Polygon Drafting Preview */}
        {activeTool === 'polygon' && polygonDraft.length > 0 && (
          <g className="pointer-events-none">
            {/* Drafted line segments */}
            {polygonDraft.map((pt, idx) => {
              if (idx === 0) return null;
              const prev = polygonDraft[idx - 1];
              return (
                <line
                  key={`draft-line-${idx}`}
                  x1={prev[0]}
                  y1={prev[1]}
                  x2={pt[0]}
                  y2={pt[1]}
                  stroke="#38bdf8"
                  strokeWidth={Math.max(2, Math.round(fw / 600))}
                />
              );
            })}

            {/* Elastic guide line to current cursor */}
            {cursorPos && (
              <line
                x1={polygonDraft[polygonDraft.length - 1][0]}
                y1={polygonDraft[polygonDraft.length - 1][1]}
                x2={cursorPos[0]}
                y2={cursorPos[1]}
                stroke="#38bdf8"
                strokeWidth={Math.max(1.5, Math.round(fw / 700))}
                strokeDasharray="5 3"
              />
            )}

            {/* Draft vertices */}
            {polygonDraft.map(([x, y], idx) => {
              const isFirst = idx === 0;
              const isClosingNear = isFirst && cursorPos && dist([x, y], cursorPos) <= handleRadius * 2;
              return (
                <circle
                  key={`draft-pt-${idx}`}
                  cx={x}
                  cy={y}
                  r={isClosingNear ? handleRadius * 1.5 : handleRadius}
                  fill={isClosingNear ? '#10b981' : '#38bdf8'}
                  stroke="#ffffff"
                  strokeWidth="2"
                />
              );
            })}
          </g>
        )}
      </svg>

      {/* Validation / Guidance Banner */}
      {isInvalid && validation.error && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-rose-950/90 border border-rose-500/60 text-rose-200 text-xs font-semibold backdrop-blur-md shadow-lg pointer-events-none animate-in fade-in">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{validation.error}</span>
        </div>
      )}

      {/* Polygon Instruction Help Pill */}
      {activeTool === 'polygon' && polygonDraft.length > 0 && !isInvalid && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-30 px-3 py-1 rounded-full bg-slate-950/85 border border-cyan-500/40 text-cyan-200 text-xs font-mono backdrop-blur-md shadow-md pointer-events-none">
          Click first vertex or double-click to close polygon
        </div>
      )}
    </div>
  );
};
