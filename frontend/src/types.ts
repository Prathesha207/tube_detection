export type ThemeMode = 'pond-light' | 'pond-dark' | 'nature';

export type StreamSourceType = 'sample-pond' | 'oak-camera' | 'uploaded-video' | 'webcam';

export type TubeRole = 'HEAD' | 'TAIL' | string;
export type TubeSize = 'BIGGER' | 'SMALLER' | 'SAME' | null;

// Native detection record emitted by head_tail_analyzer.
export interface TubeDetection {
  id: number;
  displayId?: string | number;
  role: TubeRole;
  conf?: number;
  class?: string;
  status?: string;
  bbox?: [number, number, number, number]; // [x, y, width, height] px, top-left origin
  width_px?: number | null;
  width_mm?: number | null;
  width_mm_depth?: number | null;
  z_m?: number | null;
  z_mm?: number | null;
  xyz_m?: [number, number, number] | null;
  depth_source?: string | null;
  width_now_px?: number | null;
  width_spread_px?: number | null;
  width_source?: string | null;
  size?: TubeSize;
  pair_id?: number | null; // ends sharing a pair_id = the same tube
  opaque?: boolean | null;
  colour?: { saturation: number; hue: number | null; coloured: boolean } | null;
  is_coasting?: boolean;
  missed_frames?: number;
  ratio?: number | null;
}

export interface TubeFrameResult {
  frame: number;
  heads_count: number;
  tails_count: number;
  detections: TubeDetection[];
  bigger_tube: TubeDetection | null;
  smaller_tube: TubeDetection | null;
}

export interface MLFrameDetection {
  index: string | number;
  type: string;
  widthPx: number;
  sizeTag?: string | null;
  tubeTag?: string | null;
}

export interface MLFrameOutput {
  frameNumber: number;
  headsCount: number;
  tailsCount: number;
  timestamp: string;
  verdict: string;
  detections: MLFrameDetection[];
}



// Canvas geometry is normalized for display in the frontend mapper. Detection
// measurements and confidence come from the native wheel fields above.
export interface TubeEntity {
  id: string;
  displayId?: string | number;
  role: TubeRole; // 'HEAD' | 'TAIL'
  class?: string; // Classification category from model
  label?: string; // Human-friendly display label (e.g. 'Head End', 'Tail End')
  species?: string; // Optional alias for backward compatibility
  status?: string;
  is_coasting?: boolean;
  confidence?: number;
  x: number; // percentage 0 - 100
  y: number; // percentage 0 - 100
  width: number;
  height: number;
  width_px?: number | null;
  width_mm?: number | null;
  width_mm_depth?: number | null;
  z_m?: number | null;
  z_mm?: number | null;
  xyz_m?: [number, number, number] | null;
  depth_source?: string | null;
  width_now_px?: number | null;
  width_spread_px?: number | null;
  width_source?: string | null;
  size?: TubeSize;
  pair_id?: number | null;
  opaque?: boolean | null;
  colour?: { saturation: number; hue: number | null; coloured: boolean } | null;
}

// Anomaly types map 1:1 onto signals the tube ML actually emits — nothing here
// is inferred or invented on the frontend:
//   - NO_TUBING     <- a detection's status is "NO_TUBING_MASK" or "BAD_TUBING_MASK"
//                      (analyse() in measure_from_masks.py couldn't attach/measure it)
//   - LOW_CONFIDENCE <- a detection's status starts with "CHECK" (edge/mask width
//                      disagreement, or no clear walls — see analyse())
//   - UNMATCHED_END  <- a HEAD or TAIL detection with pair_id === null
//                      (pair_ends() in measure_from_masks.py found no mutual width match)
//   - NONE           <- every detection is status "OK" and paired
export type AnomalyType = 'NONE' | 'NO_TUBING' | 'LOW_CONFIDENCE' | 'UNMATCHED_END' | 'UNKNOWN';

export interface AnomalyStatus {
  isAnomaly: boolean;
  type: AnomalyType;
  message: string;
  subMessage: string;
  headsCount: number;
  tailsCount: number;
}

export interface CameraConfig {
  id?: number;
  sourceName: string;
  resolution: '1920x1080' | '1280x720';
  targetFps: number;
  recordingFormat?: 'AVI' | 'MP4' | 'FFV1';
  recordingVideoPath?: string | null;
  rotationAngle?: number;
  controlMode?: 'auto' | 'manual';
  exposure: number; // 0 - 100
  gain?: number;
  focus?: number;
  focusAvailable?: boolean;
  brightness: number; // -50 to +50
  contrast: number; // 0 to 100
  iso: number;
  autoFocus: boolean;
  autoExposure?: boolean;
  connected: boolean;
  connectionQuality?: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  ipAddress: string;
}

export interface ProcessStep {
  id: string;
  label: string;
  status: 'completed' | 'active' | 'pending' | 'error';
  timestamp?: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  level: 'info' | 'success' | 'warn' | 'error' | 'anomaly';
}

export interface DetectionMetrics {
  fps: number;
  inferenceTimeMs: number;
  latencyMs?: number;
  framesProcessed: number;
  uptimeSeconds: number;
  avgConfidence: number;
  roleCounts: Record<string, number>;
  speciesCounts?: Record<string, number>;
}

export type LabelMode = 'compact' | 'pins' | 'hidden';

