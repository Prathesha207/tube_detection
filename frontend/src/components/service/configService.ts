
import { api } from "../../lib/api"
import type { BasicConfig } from "../../store/useCameraSystemStore"

export const configService = {

  /* ================= FETCH ================= */
  async getConfig(): Promise<BasicConfig> {
    const res = await api.get("/camera/config")
    return res.data
  },

  /* ================= SAVE ================= */
  async saveConfig(data: BasicConfig): Promise<any> {
    const res = await api.put("/camera/basic-config-update", data)
    return res.data
  },

}