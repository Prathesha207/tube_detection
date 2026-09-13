import { create } from 'zustand';

// A single detection box, matching DuckAnalyzer._build_record()'s
// detections_json entries exactly (duck_analyzer.py) -- id is null until
// the anchor locks and this object gets a permanent ID.
export interface Detection {
  id: string | null;
  species: 'duck' | 'other_toy';
  confidence: number;
  isAnomaly: boolean;
  provisional: boolean;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] pixel corners from backend
  thumbnail: string | null; // base64 data URI, or null if the crop failed
}

// Two different things share this one field, because the backend does:
//   1. session["stats"]["status"] = "processing"   (once, before the frame loop)
//   2. session["stats"].update({"status": result["status"], ...})  (every frame,
//      overwriting #1 with whatever DuckAnalyzer returned)
// So "processing" is essentially never what you actually see live -- after
// frame 1 it's always one of the FrameStatus values below. The session-
// lifecycle values only show up before processing starts or after it ends.
export type SessionLifecycleStatus =
  | 'idle' | 'queued' | 'processing' | 'completed' | 'error' | 'stopped';
export type FrameStatus = 'WARMING' | 'NORMAL' | 'ANOMALY' | 'HAND';

export interface InferenceStats {
  [x: string]: any;
  session_id: string;
  status: SessionLifecycleStatus | FrameStatus;
  frames_processed: number;
  total_frames: number;
  progress: number;
  fps: number;
  detected_duck_count: number;
  expected_duck_count: number;
  detected_other_toy_count: number;
  anchor_locked: boolean;
  hand_detected: boolean;
  missing_ids: string[];
  added_ids: string[];
  excess_ids?: string[];
  excess_count?: number;
  other_ids: string[];
  reasons: string[];

  // Present on every poll response but previously untyped:
  detections: Detection[];
  video_width: number;
  video_height: number;
  original_filename: string | null;

  // Only present once the analyzer has actually started (i.e. not on the
  // very first "queued" response) -- optional so callers must check.
  output_dir?: string;
  results_json_path?: string;
  thumbnail_dir?: string;
  // Only present if config.yaml has save_raw_frames: true.
  frames_dir?: string;
  anomaly_frames_dir?: string;
}

interface InferenceStoreState {
  stats: InferenceStats;
  isVideoLoading: boolean;
  setVideoLoading: (loading: boolean) => void;
  setStats: (newStats: Partial<InferenceStats>) => void;
  replaceStats: (stats: InferenceStats) => void;
  resetStats: () => void;
  isRecording: boolean;
  setIsRecording: (recording: boolean) => void;
}

const initialStats: InferenceStats = {
  session_id: '',
  status: 'idle',
  frames_processed: 0,
  total_frames: 0,
  progress: 0,
  fps: 0,
  detected_duck_count: 0,
  expected_duck_count: 18, // matches App.tsx's own default -- keep these two in sync if you change one
  detected_other_toy_count: 0,
  anchor_locked: false,
  hand_detected: false,
  missing_ids: [],
  added_ids: [],
  excess_ids: [],
  excess_count: 0,
  other_ids: [],
  reasons: [],
  detections: [],
  thumbnails: [],
  video_width: 0,
  video_height: 0,
  original_filename: null,
};

export const useInferenceStore = create<InferenceStoreState>((set) => ({
  stats: initialStats,
  isVideoLoading: false,
  isRecording: false,
  setIsRecording: (recording) => set({ isRecording: recording }),
  setVideoLoading: (loading) => set({ isVideoLoading: loading }),
  setStats: (newStats) => set((state) => {
    let mergedThumbnails = state.stats.thumbnails || [];
    if (Array.isArray(newStats.thumbnails) && newStats.thumbnails.length > 0) {
      const existing = new Set(mergedThumbnails.map((t: any) => `${t.id}_${t.event}`));
      const fresh = newStats.thumbnails.filter((t: any) => !existing.has(`${t.id}_${t.event}`));
      if (fresh.length > 0) {
        mergedThumbnails = [...mergedThumbnails, ...fresh];
      }
    }
    return {
      stats: {
        ...state.stats,
        ...newStats,
        thumbnails: mergedThumbnails,
      },
    };
  }),
  replaceStats: (stats) => set({ stats }),
  resetStats: () => set({ stats: initialStats }),
}));
