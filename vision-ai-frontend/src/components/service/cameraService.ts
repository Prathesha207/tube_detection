import { api } from "../../lib/api";

export interface CameraData {
  id?: number;
  name?: string;
  ip_address?: string;
  resolution?: string;
  fps?: number;
  rotation_angle?: number;
  control_mode?: 'auto' | 'manual';
  exposure?: number;
  gain?: number;
  focus?: number;
  brightness?: number;
  contrast?: number;
  auto_focus?: boolean;
  auto_exposure?: boolean;
  is_enabled?: boolean;
}

export const cameraService = {
  async checkCamera() {
    const res = await api.get("/camera/");
    return res.data;
  },

  async createCamera(data: CameraData) {
    const res = await api.post("/camera/create", data);
    return res.data;
  },

  async enableCamera(id: string | number) {
    const res = await api.post(`/camera/enable/${id}`);
    return res.data;
  },

  async disableCamera(id: string | number) {
    const res = await api.post(`/camera/disable/${id}`);
    return res.data;
  },

  async updateCamera(id: string | number, data: CameraData) {
    const res = await api.put(`/camera/update/${id}`, data);
    return res.data;
  },

  async deleteCamera(id: string | number) {
    const res = await api.delete(`/camera/${id}`);
    return res.data;
  },

  async saveCamera(data: CameraData) {
    if (data.id) {
      return await this.updateCamera(data.id, data);
    }
    return await this.createCamera(data);
  },

  async start(payload?: { camera_id?: number; ip_address?: string }) {
    const res = await api.post("/oak/start", payload || {});
    return res.data;
  },

  async stop() {
    const res = await api.post("/oak/stop");
    return res.data;
  },

  async startStream() {
    const res = await api.post("/oak/stream/start");
    return res.data;
  },

  async stopStream() {
    const res = await api.post("/oak/stream/stop");
    return res.data;
  },

  async startLiveInference(sessionId: string = "live") {
    const res = await api.post(`/oak/inference/start/${sessionId}`);
    return res.data;
  },

  async stopLiveInference() {
    const res = await api.post("/oak/inference/stop");
    return res.data;
  },

  async health() {
    const res = await api.get("/oak/health");
    return res.data;
  },

  async updateLiveControls(id: string | number | undefined, data: any) {
    try {
      const res = await api.post("/oak/controls", data);
      return res.data;
    } catch {
      if (id) {
        const res = await api.patch(`/camera/live-controls/${id}`, data);
        return res.data;
      }
    }
  },

   async updateInferenceMode(mode: "testing" | "production") {
    const res = await api.patch("/camera/inference-mode", { mode });
    return res.data;
  },

  async getAvailableDevices() {
    try {
      const res = await api.get("/oak/devices");
      return res.data || [];
    } catch {
      return [];
    }
  },
};

export default cameraService;
