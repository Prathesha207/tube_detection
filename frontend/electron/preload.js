const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("electronAPI", {
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectFile: () => ipcRenderer.invoke("select-file"),
  writeLog: (message) => ipcRenderer.invoke("write-log", message),
  // Notify main process about inference state for the close guard dialog
  setInferenceRunning: (running) => ipcRenderer.send("set-inference-running", running),
})
