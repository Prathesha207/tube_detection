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

type EdgeSide = 'top' | 'bottom' | 'left' | 'right';

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
    type: 'none' | 'rect_drag' | 'vertex_drag' | 'edge_drag' | 'polygon_pan';
    startPoint?: Point;
    vertexIndex?: number;
    edgeSide?: EdgeSide;
    initialPoints?: Point[];
  }>({ type: 'none' });

  // Rectangle Drawing State (supports both drag-to-size and click-to-start + click-to-finish)
  const [rectAnchor, setRectAnchor] = useState<Point | null>(null);
  const [rectCurrent, setRectCurrent] = useState<Point | null>(null);
  const [isShiftPressed, setIsShiftPressed] = useState<boolean>(false);

  // Polygon Draft State
  const [polygonDraft, setPolygonDraft] = useState<Point[]>([]);
  const [cursorPos, setCursorPos] = useState<Point | null>(null);
  const [hoveredVertexIndex, setHoveredVertexIndex] = useState<number | null>(null);
  const [hoveredEdgeSide, setHoveredEdgeSide] = useState<EdgeSide | null>(null);

  // Handle radius in frame coordinates
  const handleRadius = Math.max(8, Math.round(fw / 130));

  // Check validation state of current points
  const validation = useMemo(() => validateRoi(points), [points]);

  // Clean draft if tool changes
  useEffect(() => {
    if (activeTool !== 'polygon') {
      setPolygonDraft([]);
    }
    if (activeTool !== 'rectangle') {
      setRectAnchor(null);
      setRectCurrent(null);
    }
    setDragState({ type: 'none' });
  }, [activeTool]);

  // Keyboard shortcuts (Escape to cancel draft, Delete to clear, Shift for 1:1 square)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        setIsShiftPressed(true);
      }
      if (e.key === 'Escape') {
        if (polygonDraft.length > 0) {
          setPolygonDraft([]);
        } else if (rectAnchor !== null) {
          setRectAnchor(null);
          setRectCurrent(null);
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

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift') {
        setIsShiftPressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [polygonDraft.length, rectAnchor, activeTool, onSelectTool]);

  // Helper to get frame point from client event with clamping
  const getFramePt = useCallback(
    (clientX: number, clientY: number): Point => {
      if (!containerRef.current) return [0, 0];
      const rect = containerRef.current.getBoundingClientRect();
      return screenToFramePoint(clientX, clientY, rect, fw, fh);
    },
    [fw, fh]
  );

  // Distance between two points
  const dist = (p1: Point, p2: Point) => {
    const dx = p1[0] - p2[0];
    const dy = p1[1] - p2[1];
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Detect if points form an axis-aligned 4-point rectangle
  const isRectangle = useMemo(() => {
    if (points.length !== 4) return false;
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const w = maxX - minX;
    const h = maxY - minY;
    if (w < 10 || h < 10) return false;

    // Check each corner roughly matches the 4 bounding box points
    const corners: Point[] = [
      [minX, minY],
      [maxX, minY],
      [maxX, maxY],
      [minX, maxY],
    ];
    return points.every(
      (p, i) => Math.abs(p[0] - corners[i][0]) <= 8 && Math.abs(p[1] - corners[i][1]) <= 8
    );
  }, [points]);

  // Compute 4 edge midpoint coordinates for rectangles
  const edgeMidpoints = useMemo<{ [key in EdgeSide]: Point } | null>(() => {
    if (!isRectangle || points.length !== 4) return null;
    const [p0, p1, p2, p3] = points;
    return {
      top: [Math.round((p0[0] + p1[0]) / 2), p0[1]],
      right: [p1[0], Math.round((p1[1] + p2[1]) / 2)],
      bottom: [Math.round((p3[0] + p2[0]) / 2), p2[1]],
      left: [p0[0], Math.round((p0[1] + p3[1]) / 2)],
    };
  }, [isRectangle, points]);

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

  // Find nearest edge midpoint within threshold (for rectangles)
  const findEdgeSide = useCallback(
    (pt: Point, thresholdPx: number = handleRadius * 1.5): EdgeSide | null => {
      if (!edgeMidpoints) return null;
      let closestSide: EdgeSide | null = null;
      let minDist = thresholdPx;
      (Object.keys(edgeMidpoints) as EdgeSide[]).forEach((side) => {
        const mid = edgeMidpoints[side];
        const d = dist(pt, mid);
        if (d < minDist) {
          minDist = d;
          closestSide = side;
        }
      });
      return closestSide;
    },
    [edgeMidpoints, handleRadius]
  );

  // Helper to commit a rectangle from two points
  const commitRectangle = useCallback(
    (p1: Point, p2: Point) => {
      let minX = Math.min(p1[0], p2[0]);
      let maxX = Math.max(p1[0], p2[0]);
      let minY = Math.min(p1[1], p2[1]);
      let maxY = Math.max(p1[1], p2[1]);

      let w = maxX - minX;
      let h = maxY - minY;

      if (w < 10 || h < 10) return false;

      if (isShiftPressed) {
        const side = Math.max(w, h);
        if (p2[0] < p1[0]) minX = maxX - side;
        else maxX = minX + side;
        if (p2[1] < p1[1]) minY = maxY - side;
        else maxY = minY + side;
      }

      minX = Math.max(0, Math.min(fw - 1, minX));
      maxX = Math.max(0, Math.min(fw - 1, maxX));
      minY = Math.max(0, Math.min(fh - 1, minY));
      maxY = Math.max(0, Math.min(fh - 1, maxY));

      const rectPts: Point[] = [
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY],
      ];

      onChange(rectPts);
      onSelectTool('pointer');
      setRectAnchor(null);
      setRectCurrent(null);
      setDragState({ type: 'none' });
      return true;
    },
    [isShiftPressed, fw, fh, onChange, onSelectTool]
  );

  // ---------------- Pointer Events ----------------
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      // Ignored if capture unsupported
    }

    const pt = getFramePt(e.clientX, e.clientY);

    // 1. RECTANGLE TOOL: Supports both Click-Drag and Click-Move-Click
    if (activeTool === 'rectangle') {
      if (!rectAnchor) {
        // First click/press: set anchor and current
        setRectAnchor(pt);
        setRectCurrent(pt);
        setDragState({ type: 'rect_drag', startPoint: pt });
      } else {
        // Second click: commit rectangle!
        commitRectangle(rectAnchor, pt);
      }
      return;
    }

    // 2. POLYGON TOOL
    if (activeTool === 'polygon') {
      // If clicking near first point and polygon has >= 3 points, close it
      if (polygonDraft.length >= 3 && dist(pt, polygonDraft[0]) <= handleRadius * 2) {
        onChange(polygonDraft);
        setPolygonDraft([]);
        onSelectTool('pointer');
        return;
      }
      setPolygonDraft((prev) => [...prev, pt]);
      return;
    }

    // 3. POINTER / ADJUST TOOL
    if (activeTool === 'pointer') {
      // Check if clicking on an existing corner vertex
      const vIdx = findVertexIndex(pt);
      if (vIdx !== null) {
        setDragState({
          type: 'vertex_drag',
          vertexIndex: vIdx,
          initialPoints: [...points],
        });
        return;
      }

      // Check if clicking on an edge midpoint (for rectangles)
      const edgeSide = findEdgeSide(pt);
      if (edgeSide !== null) {
        setDragState({
          type: 'edge_drag',
          edgeSide,
          initialPoints: [...points],
        });
        return;
      }

      // Check if clicking inside polygon to pan/translate it
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
    const pt = getFramePt(e.clientX, e.clientY);
    setCursorPos(pt);

    // Hover detection for pointer tool
    if (activeTool === 'pointer' && dragState.type === 'none') {
      const vIdx = findVertexIndex(pt);
      setHoveredVertexIndex(vIdx);
      if (vIdx === null && isRectangle) {
        setHoveredEdgeSide(findEdgeSide(pt));
      } else {
        setHoveredEdgeSide(null);
      }
    }

    // 1. Rectangle creation preview
    if (activeTool === 'rectangle' && rectAnchor) {
      setRectCurrent(pt);
      return;
    }

    // 2. Vertex Drag (Smart Rectangle resizing or polygon vertex move)
    if (dragState.type === 'vertex_drag' && dragState.vertexIndex !== undefined && dragState.initialPoints) {
      const vIdx = dragState.vertexIndex;

      if (isRectangle && dragState.initialPoints.length === 4) {
        const p = dragState.initialPoints;
        let newPts: Point[];

        if (vIdx === 0) {
          // Top-Left: updates minX and minY; corner 2 (maxX, maxY) remains anchored
          const newMinX = Math.min(pt[0], p[1][0] - 10);
          const newMinY = Math.min(pt[1], p[3][1] - 10);
          newPts = [
            [newMinX, newMinY],
            [p[1][0], newMinY],
            [p[2][0], p[2][1]],
            [newMinX, p[3][1]],
          ];
        } else if (vIdx === 1) {
          // Top-Right: updates maxX and minY; corner 3 (minX, maxY) remains anchored
          const newMaxX = Math.max(pt[0], p[0][0] + 10);
          const newMinY = Math.min(pt[1], p[2][1] - 10);
          newPts = [
            [p[0][0], newMinY],
            [newMaxX, newMinY],
            [newMaxX, p[2][1]],
            [p[3][0], p[3][1]],
          ];
        } else if (vIdx === 2) {
          // Bottom-Right: updates maxX and maxY; corner 0 (minX, minY) remains anchored
          const newMaxX = Math.max(pt[0], p[3][0] + 10);
          const newMaxY = Math.max(pt[1], p[1][1] + 10);
          newPts = [
            [p[0][0], p[0][1]],
            [newMaxX, p[1][1]],
            [newMaxX, newMaxY],
            [p[3][0], newMaxY],
          ];
        } else {
          // Bottom-Left (vIdx === 3): updates minX and maxY; corner 1 (maxX, minY) remains anchored
          const newMinX = Math.min(pt[0], p[2][0] - 10);
          const newMaxY = Math.max(pt[1], p[0][1] + 10);
          newPts = [
            [newMinX, p[0][1]],
            [p[1][0], p[1][1]],
            [p[2][0], newMaxY],
            [newMinX, newMaxY],
          ];
        }
        onChange(newPts);
      } else {
        // Arbitrary polygon vertex drag
        const newPts = [...points];
        newPts[vIdx] = pt;
        onChange(newPts);
      }
      return;
    }

    // 3. Edge Drag (1D Rectangle stretching)
    if (dragState.type === 'edge_drag' && dragState.edgeSide && dragState.initialPoints?.length === 4) {
      const p = dragState.initialPoints;
      const side = dragState.edgeSide;
      let newPts: Point[];

      if (side === 'top') {
        const newMinY = Math.min(pt[1], p[3][1] - 10);
        newPts = [
          [p[0][0], newMinY],
          [p[1][0], newMinY],
          p[2],
          p[3],
        ];
      } else if (side === 'bottom') {
        const newMaxY = Math.max(pt[1], p[0][1] + 10);
        newPts = [
          p[0],
          p[1],
          [p[2][0], newMaxY],
          [p[3][0], newMaxY],
        ];
      } else if (side === 'left') {
        const newMinX = Math.min(pt[0], p[1][0] - 10);
        newPts = [
          [newMinX, p[0][1]],
          p[1],
          p[2],
          [newMinX, p[3][1]],
        ];
      } else {
        // right
        const newMaxX = Math.max(pt[0], p[0][0] + 10);
        newPts = [
          p[0],
          [newMaxX, p[1][1]],
          [newMaxX, p[2][1]],
          p[3],
        ];
      }
      onChange(newPts);
      return;
    }

    // 4. Polygon Pan / Translation
    if (dragState.type === 'polygon_pan' && dragState.startPoint && dragState.initialPoints) {
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
      // Ignored
    }

    // If dragging rectangle and moved >= 10px, commit immediately on mouse release
    if (dragState.type === 'rect_drag' && rectAnchor && rectCurrent) {
      const spanX = Math.abs(rectCurrent[0] - rectAnchor[0]);
      const spanY = Math.abs(rectCurrent[1] - rectAnchor[1]);
      if (spanX >= 10 && spanY >= 10) {
        commitRectangle(rectAnchor, rectCurrent);
        return;
      }
      // If user merely clicked once (< 10px movement), keep rectAnchor active for 2-click mode!
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
    if (activeTool !== 'rectangle' || !rectAnchor || !rectCurrent) return null;
    let minX = Math.min(rectAnchor[0], rectCurrent[0]);
    let maxX = Math.max(rectAnchor[0], rectCurrent[0]);
    let minY = Math.min(rectAnchor[1], rectCurrent[1]);
    let maxY = Math.max(rectAnchor[1], rectCurrent[1]);

    let w = maxX - minX;
    let h = maxY - minY;

    if (isShiftPressed) {
      const side = Math.max(w, h);
      if (rectCurrent[0] < rectAnchor[0]) minX = maxX - side;
      else maxX = minX + side;
      if (rectCurrent[1] < rectAnchor[1]) minY = maxY - side;
      else maxY = minY + side;
      w = side;
      h = side;
    }

    return { x: minX, y: minY, w: Math.max(1, w), h: Math.max(1, h) };
  }, [activeTool, rectAnchor, rectCurrent, isShiftPressed]);

  // Determine cursor style
  const cursorStyle = useMemo(() => {
    if (activeTool === 'rectangle') return 'crosshair';
    if (activeTool === 'polygon') return 'crosshair';

    if (dragState.type === 'vertex_drag' || hoveredVertexIndex !== null) {
      if (isRectangle) {
        const idx = dragState.vertexIndex !== undefined ? dragState.vertexIndex : hoveredVertexIndex;
        if (idx === 0 || idx === 2) return 'nwse-resize';
        if (idx === 1 || idx === 3) return 'nesw-resize';
      }
      return 'grab';
    }

    if (dragState.type === 'edge_drag' || hoveredEdgeSide !== null) {
      const side = dragState.edgeSide || hoveredEdgeSide;
      if (side === 'top' || side === 'bottom') return 'ns-resize';
      if (side === 'left' || side === 'right') return 'ew-resize';
    }

    if (dragState.type === 'polygon_pan') return 'grabbing';
    return 'default';
  }, [activeTool, hoveredVertexIndex, hoveredEdgeSide, dragState.type, dragState.vertexIndex, dragState.edgeSide, isRectangle]);

  const isInvalid = !validation.valid;
  const strokeColor = isInvalid ? '#f43f5e' : '#06b6d4';

  return (
    <div
      ref={containerRef}
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onDoubleClick={handleDoubleClick}
      style={{ cursor: cursorStyle, zIndex: 40 }}
      className="absolute inset-0 z-40 overflow-hidden pointer-events-auto select-none"
    >
      <svg
        className="w-full h-full block"
        viewBox={`0 0 ${fw} ${fh}`}
        preserveAspectRatio="none"
      >
        {/* Main Saved / Active ROI Polygon (Clean outline only, no color tint or dimming) */}
        {points.length >= 3 && activeTool !== 'rectangle' && (
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

        {/* Semi-transparent outline of previous polygon if drawing a new rectangle */}
        {points.length >= 3 && activeTool === 'rectangle' && (
          <polygon
            points={pointsToString(points)}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="1.5"
            strokeDasharray="4 4"
            opacity={0.35}
          />
        )}

        {/* 1. Corner Handles in POINTER Mode */}
        {activeTool === 'pointer' &&
          points.map(([x, y], idx) => {
            const isHovered =
              hoveredVertexIndex === idx ||
              (dragState.type === 'vertex_drag' && dragState.vertexIndex === idx);
            return (
              <g key={`handle-corner-${idx}`}>
                <circle
                  cx={x}
                  cy={y}
                  r={isHovered ? handleRadius * 1.3 : handleRadius}
                  fill={isHovered ? '#ffffff' : strokeColor}
                  stroke="#0f172a"
                  strokeWidth="2.5"
                  className="transition-transform"
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

        {/* 2. Edge Midpoint Handles for 4-point Rectangles */}
        {activeTool === 'pointer' &&
          isRectangle &&
          edgeMidpoints &&
          (Object.keys(edgeMidpoints) as EdgeSide[]).map((side) => {
            const [mx, my] = edgeMidpoints[side];
            const isHovered =
              hoveredEdgeSide === side ||
              (dragState.type === 'edge_drag' && dragState.edgeSide === side);
            const isVertical = side === 'top' || side === 'bottom';
            const w = isVertical ? handleRadius * 1.8 : handleRadius * 0.9;
            const h = isVertical ? handleRadius * 0.9 : handleRadius * 1.8;

            return (
              <rect
                key={`edge-handle-${side}`}
                x={mx - w / 2}
                y={my - h / 2}
                width={w}
                height={h}
                rx={2}
                fill={isHovered ? '#38bdf8' : '#0f172a'}
                stroke={isHovered ? '#ffffff' : '#38bdf8'}
                strokeWidth={2}
                className="transition-colors"
              />
            );
          })}

        {/* 3. Live Rectangle Preview (Active while dragging or two-click sizing) */}
        {rectPreview && (
          <g className="pointer-events-none">
            <rect
              x={rectPreview.x}
              y={rectPreview.y}
              width={rectPreview.w}
              height={rectPreview.h}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeDasharray="6 4"
            />
            {/* Dimensions Badge */}
            <g transform={`translate(${rectPreview.x + 8}, ${Math.max(16, rectPreview.y - 12)})`}>
              <rect
                x={0}
                y={-14}
                width={110}
                height={20}
                rx={4}
                fill="rgba(15, 23, 42, 0.9)"
                stroke="rgba(6, 182, 212, 0.6)"
                strokeWidth={1}
              />
              <text
                x={6}
                y={0}
                fill="#22d3ee"
                fontSize={11}
                fontFamily="monospace"
                fontWeight="bold"
              >
                {`${Math.round(rectPreview.w)} × ${Math.round(rectPreview.h)} px`}
              </text>
            </g>
          </g>
        )}

        {/* 4. Live Polygon Drafting Preview */}
        {activeTool === 'polygon' && polygonDraft.length > 0 && (
          <g className="pointer-events-none">
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

            {polygonDraft.map(([x, y], idx) => {
              const isFirst = idx === 0;
              const isClosingNear =
                isFirst && cursorPos && dist([x, y], cursorPos) <= handleRadius * 2;
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

      {/* Validation Banner */}
      {isInvalid && validation.error && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-40 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-rose-950/90 border border-rose-500/60 text-rose-200 text-xs font-semibold backdrop-blur-md shadow-lg pointer-events-none animate-in fade-in">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{validation.error}</span>
        </div>
      )}

      {/* Rectangle Instruction Help Pill */}
      {activeTool === 'rectangle' && !isInvalid && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-40 px-3.5 py-1.5 rounded-full bg-slate-950/90 border border-cyan-500/50 text-cyan-200 text-xs font-mono backdrop-blur-md shadow-lg pointer-events-none flex items-center gap-2 animate-in fade-in">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping shrink-0" />
          <span>Click & drag or click two opposite corners • Hold Shift for square</span>
        </div>
      )}

      {/* Polygon Instruction Help Pill */}
      {activeTool === 'polygon' && polygonDraft.length > 0 && !isInvalid && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-40 px-3.5 py-1.5 rounded-full bg-slate-950/90 border border-cyan-500/50 text-cyan-200 text-xs font-mono backdrop-blur-md shadow-lg pointer-events-none flex items-center gap-2 animate-in fade-in">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping shrink-0" />
          <span>Click first vertex or double-click to close polygon</span>
        </div>
      )}
    </div>
  );
};
