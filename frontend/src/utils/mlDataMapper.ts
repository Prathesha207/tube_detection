import { TubeEntity } from '../types';

export const resetBBoxCache = () => {};

/**
 * Maps raw backend Tube ML detection dictionary into frontend-friendly TubeEntity objects.
 * Everything here is a direct pass-through of TubeDetection properties
 * (role, bbox, width_px, width_mm, size, pair_id, opaque, colour, status).
 */
export const mapDetectionsToTubes = (
  data: any,
  vw: number,
  vh: number
): TubeEntity[] => {
  if (!data || !Array.isArray(data.detections)) {
    return [];
  }

  const rawVw = Number(data?.video_width);
  const rawVh = Number(data?.video_height);
  const effectiveVw = rawVw > 0 ? rawVw : (vw > 0 ? vw : 1920);
  const effectiveVh = rawVh > 0 ? rawVh : (vh > 0 ? vh : 1080);

  const incomingTubes: TubeEntity[] = [];

  data.detections.forEach((d: any, idx: number) => {
    let px = 0, py = 0, pw = 0, ph = 0;
    const bArr = d.bbox || d.box;
    if (Array.isArray(bArr) && bArr.length === 4) {
      if (d.role !== undefined || d.width_px !== undefined || d.pair_id !== undefined) {
        // Tube ML bbox: [x, y, width, height] in pixels
        px = (bArr[0] / effectiveVw) * 100;
        py = (bArr[1] / effectiveVh) * 100;
        pw = (bArr[2] / effectiveVw) * 100;
        ph = (bArr[3] / effectiveVh) * 100;
      } else {
        // Fallback: [x1, y1, x2, y2]
        px = (bArr[0] / effectiveVw) * 100;
        py = (bArr[1] / effectiveVh) * 100;
        pw = ((bArr[2] - bArr[0]) / effectiveVw) * 100;
        ph = ((bArr[3] - bArr[1]) / effectiveVh) * 100;
      }
    }

    const role = d.role || 'UNKNOWN';
    const status = d.status || 'OK';
    const isAnomaly = status !== 'OK';
    const rawId = d.id !== null && d.id !== undefined ? d.id : idx + 1;
    const uniqueId = `${role}-${rawId}`;
    const displayId = String(rawId);
    const isCoasting = Boolean(d.is_coasting || (d.missed_frames && d.missed_frames > 0) || status === 'COASTED');

    incomingTubes.push({
      id: uniqueId,
      displayId: displayId,
      role: role,
      label: role === 'HEAD' ? 'Head End' : role === 'TAIL' ? 'Tail End' : (d.class || 'Tube End'),
      species: role === 'HEAD' ? 'Head' : role === 'TAIL' ? 'Tail' : (d.class || 'Tube'),
      class: d.class || role,
      status: status,
      isAnomaly: isAnomaly,
      is_coasting: isCoasting,
      confidence: typeof d.confidence === 'number' ? d.confidence : 1.0,
      x: px,
      y: py,
      width: pw,
      height: ph,
      width_px: d.width_px ?? null,
      width_mm: d.width_mm ?? null,
      size: d.size ?? null,
      pair_id: d.pair_id ?? null,
      opaque: d.opaque ?? null,
      colour: d.colour ?? null,
    });
  });

  return incomingTubes;
};
