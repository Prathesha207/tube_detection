/**
 * roiUtils.ts
 * ===========
 * Coordinate mapping, bounding calculations, and geometric validation for
 * Region of Interest (ROI) polygons.
 *
 * ML coordinates operate in native video frame space (0..frameWidth, 0..frameHeight).
 * Rendering is performed on top of the fitted container element.
 */

export type Point = [number, number];

/**
 * Screen client coordinate -> Native video frame pixel [fx, fy].
 * Clamps coordinates to the container bounds so dragging doesn't freeze at the edges.
 */
export function screenToFramePoint(
  clientX: number,
  clientY: number,
  containerRect: { left: number; top: number; width: number; height: number },
  frameW: number,
  frameH: number
): Point {
  if (!containerRect || containerRect.width <= 0 || containerRect.height <= 0 || frameW <= 0 || frameH <= 0) {
    return [0, 0];
  }

  const relX = clientX - containerRect.left;
  const relY = clientY - containerRect.top;

  // Clamped to container viewport
  const clampedX = Math.max(0, Math.min(containerRect.width, relX));
  const clampedY = Math.max(0, Math.min(containerRect.height, relY));

  const fx = Math.round((clampedX / containerRect.width) * frameW);
  const fy = Math.round((clampedY / containerRect.height) * frameH);

  return [
    Math.max(0, Math.min(frameW, fx)),
    Math.max(0, Math.min(frameH, fy)),
  ];
}

/**
 * Native video frame pixel -> Container screen pixel [sx, sy].
 */
export function frameToScreenPoint(
  point: Point,
  containerRect: { width: number; height: number },
  frameW: number,
  frameH: number
): Point {
  if (frameW <= 0 || frameH <= 0 || !containerRect || containerRect.width <= 0 || containerRect.height <= 0) {
    return [0, 0];
  }
  return [
    (point[0] / frameW) * containerRect.width,
    (point[1] / frameH) * containerRect.height,
  ];
}

/**
 * Line segment intersection using cross product (CCW).
 */
export function segmentsIntersect(p1: Point, p2: Point, p3: Point, p4: Point): boolean {
  const ccw = (a: Point, b: Point, c: Point) => {
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
  };
  return (
    ccw(p1, p3, p4) !== ccw(p2, p3, p4) &&
    ccw(p1, p2, p3) !== ccw(p1, p2, p4)
  );
}

/**
 * Check if polygon edges cross each other (self-intersection).
 */
export function isSelfIntersecting(pts: Point[]): boolean {
  const n = pts.length;
  if (n < 4) return false;

  for (let i = 0; i < n; i++) {
    const p1 = pts[i];
    const p2 = pts[(i + 1) % n];

    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue; // adjacent closing edge
      const p3 = pts[j];
      const p4 = pts[(j + 1) % n];
      if (segmentsIntersect(p1, p2, p3, p4)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Shoelace polygon area calculation.
 */
export function calculatePolygonArea(pts: Point[]): number {
  const n = pts.length;
  if (n < 3) return 0;
  let area = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += pts[i][0] * pts[j][1];
    area -= pts[j][0] * pts[i][1];
  }
  return Math.abs(area) / 2.0;
}

/**
 * Client-side validation matching backend TubeAnalyzer.roi_from_points.
 */
export function validateRoi(points: Point[] | null | undefined): { valid: boolean; error?: string } {
  if (!points || points.length === 0) {
    return { valid: true }; // Cleared ROI is valid (full frame)
  }

  // Remove duplicate closing point if user provided one
  let pts = [...points];
  if (pts.length > 3 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]) {
    pts = pts.slice(0, pts.length - 1);
  }

  if (pts.length < 3) {
    return { valid: false, error: 'ROI requires at least 3 points.' };
  }
  if (pts.length > 100) {
    return { valid: false, error: 'ROI cannot exceed 100 points.' };
  }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const wSpan = maxX - minX;
  const hSpan = maxY - minY;
  if (wSpan < 10 || hSpan < 10) {
    return { valid: false, error: 'ROI must be at least 10x10 pixels.' };
  }

  const area = calculatePolygonArea(pts);
  if (area < 100) {
    return { valid: false, error: 'ROI area is too small (minimum 100 square pixels).' };
  }

  if (isSelfIntersecting(pts)) {
    return { valid: false, error: 'ROI polygon cannot be self-intersecting.' };
  }

  return { valid: true };
}
