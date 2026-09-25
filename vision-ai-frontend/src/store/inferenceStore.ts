import { create } from 'zustand';
import type { TubeDetection } from '../types';

export type { TubeDetection };
export type Detection = TubeDetection;

export type SessionLifecycleStatus =
  | 'idle' | 'queued' | 'processing' | 'completed' | 'error' | 'stopped' | 'paused';
export type FrameStatus = 'WARMING' | 'NORMAL' | 'ANOMALY' | string;

export interface InferenceStats {
  session_id: string;
  status: SessionLifecycleStatus | FrameStatus;
  frame?: number;
  frames_processed: number;
  total_frames: number;
  progress: number;
  fps: number;
  latency_ms?: number;

  // Tube ML fields from TubeAnalyzer / run_video_frames:
  heads_count: number;
  tails_count: number;
  detections: TubeDetection[];
  bigger_tube: TubeDetection | null;
  smaller_tube: TubeDetection | null;

  video_width: number;
  video_height: number;
  original_filename: string | null;
  output_dir?: string;
  output_file?: string;
  reasons?: string[];
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
  frame: 0,
  frames_processed: 0,
  total_frames: 0,
  progress: 0,
  fps: 0,
  latency_ms: 0,
  heads_count: 0,
  tails_count: 0,
  detections: [],
  bigger_tube: null,
  smaller_tube: null,
  video_width: 0,
  video_height: 0,
  original_filename: null,
  reasons: [],
};

export const useInferenceStore = create<InferenceStoreState>((set) => ({
  stats: initialStats,
  isVideoLoading: false,
  isRecording: false,
  setIsRecording: (recording) => set({ isRecording: recording }),
  setVideoLoading: (loading) => set({ isVideoLoading: loading }),
  setStats: (newStats) => set((state) => ({
    stats: {
      ...state.stats,
      ...newStats,
    },
  })),
  replaceStats: (stats) => set({ stats }),
  resetStats: () => set({ stats: initialStats }),
}));
