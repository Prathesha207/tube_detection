import { api } from "../../lib/api"

export interface StartRecordingResponse {
  status: string;
  session_id: string;
  recording_path: string;
}

export interface StopRecordingResponse {
  status: string;
  session_id?: string;
  recording_session_id?: string;
  recording_path?: string;
  filename?: string;
  duration?: number;
  frames?: number;
  stream_url?: string;
  message?: string;
}

export const recordingService = {

  async startRecording(sessionId: string, recordingFormat?: string) {
    const res = await api.post("/recording/start", {
      session_id: sessionId,
      recording_format: recordingFormat,
    });
    return res.data as StartRecordingResponse;
  },

  async stopRecording(sessionId: string) {
    const res = await api.post("/recording/stop", { session_id: sessionId });
    return res.data as StopRecordingResponse;
  },

  async getStatus() {
    const res = await api.get("/recording/status");
    return res.data as { is_recording: boolean; active_count: number };
  },

}
