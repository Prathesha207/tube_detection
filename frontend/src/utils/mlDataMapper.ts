import { TubeEntity } from '../types';

export const resetBBoxCache = () => {};

/** Convert the wheel's native pixel boxes to canvas percentages for rendering. */
export const mapDetectionsToTubes = (data: any): TubeEntity[] => {
  const frameWidth = Number(data?.video_width);
  const frameHeight = Number(data?.video_height);
  if (!Array.isArray(data?.detections) || frameWidth <= 0 || frameHeight <= 0) return [];

  return data.detections.flatMap((d: any): TubeEntity[] => {
    const bbox = d?.bbox;
    if (!Array.isArray(bbox) || bbox.length !== 4 || d.id == null || !d.role) return [];
    const [x, y, width, height] = bbox.map(Number);
    if (![x, y, width, height].every(Number.isFinite)) return [];

    const role = String(d.role);
    const status = typeof d.status === 'string' ? d.status : undefined;
    const zMm = typeof d.z_mm === 'number'
      ? d.z_mm
      : (typeof d.z_m === 'number' ? d.z_m * 1000 : null);

    return [{
      id: `${role}-${d.id}`,
      displayId: String(d.id),
      role,
      label: role === 'HEAD' ? 'Head End' : role === 'TAIL' ? 'Tail End' : role,
      species: role === 'HEAD' ? 'Head' : role === 'TAIL' ? 'Tail' : role,
      class: typeof d.class === 'string' ? d.class : role,
      status,
      is_coasting: d.is_coasting,
      confidence: typeof d.conf === 'number' ? d.conf : undefined,
      x: (x / frameWidth) * 100,
      y: (y / frameHeight) * 100,
      width: (width / frameWidth) * 100,
      height: (height / frameHeight) * 100,
      width_px: typeof d.width_px === 'number' ? d.width_px : null,
      width_mm: typeof d.width_mm === 'number' ? d.width_mm : null,
      width_mm_depth: typeof d.width_mm_depth === 'number' ? d.width_mm_depth : null,
      z_m: typeof d.z_m === 'number' ? d.z_m : null,
      z_mm: zMm,
      xyz_m: Array.isArray(d.xyz_m) ? d.xyz_m : null,
      depth_source: d.depth_source ?? null,
      width_now_px: typeof d.width_now_px === 'number' ? d.width_now_px : null,
      width_spread_px: typeof d.width_spread_px === 'number' ? d.width_spread_px : null,
      width_source: d.width_source ?? null,
      size: d.size ?? null,
      pair_id: d.pair_id ?? null,
      opaque: d.opaque ?? null,
      colour: d.colour ?? null,
    }];
  });
};
