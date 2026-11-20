const { contextBridge, ipcRenderer, webUtils } = require("electron")

contextBridge.exposeInMainWorld("electronAPI", {
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectFile: () => ipcRenderer.invoke("select-file"),
  getPathForFile: (file) => {
    try {
      if (webUtils && typeof webUtils.getPathForFile === "function") {
        return webUtils.getPathForFile(file)
      }
    } catch (e) {
      console.warn("getPathForFile error:", e)
    }
    return file?.path || ""
  },
  writeLog: (message) => ipcRenderer.invoke("write-log", message),
  // Notify main process about inference state for the close guard dialog
  setInferenceRunning: (running) => ipcRenderer.send("set-inference-running", running),
})
