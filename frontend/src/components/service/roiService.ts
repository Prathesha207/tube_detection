/**
 * roiService.ts
 * =============
 * API client service for fetching and updating the inspection Region of Interest (ROI).
 */

import api from '../../lib/api';
import { Point } from '../../utils/roiUtils';

export interface RoiResponse {
  points: Point[];
  frame_width?: number | null;
  frame_height?: number | null;
}

export const roiService = {
  /**
   * Fetch current ROI points and frame dimensions from backend.
   */
  async getRoi(sessionId?: string, streamW?: number, streamH?: number): Promise<RoiResponse> {
    try {
      const params: Record<string, any> = {};
      if (sessionId) params.session_id = sessionId;
      if (streamW) params.stream_width = streamW;
      if (streamH) params.stream_height = streamH;
      const res = await api.get<RoiResponse>('/roi', { params });
      return res.data || { points: [] };
    } catch (err) {
      console.warn('[roiService] Failed to fetch ROI:', err);
      return { points: [] };
    }
  },

  /**
   * Persist ROI points to backend and trigger atomic runtime hot-swap.
   */
  async saveRoi(
    points: Point[] | null,
    frameW?: number,
    frameH?: number,
    sessionId?: string,
    sourceType?: string
  ): Promise<{ status: string; message: string; points: Point[] }> {
    const payload = {
      points: points && points.length > 0 ? points : [],
      frame_width: frameW || null,
      frame_height: frameH || null,
      session_id: sessionId || null,
      source_type: sourceType || null,
    };
    const res = await api.post<{ status: string; message: string; points: Point[] }>('/roi', payload);
    return res.data;
  },
};
