const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  protocol,
  net,
} = require("electron")

const path = require("path")
const axios = require("axios")
const { spawn, execSync } = require("child_process")
const fs = require("fs")
const netSocket = require("net")

/* =========================================================
   1. SINGLE INSTANCE LOCK (MUST BE BEFORE ANY PROCESS OPERATIONS)
========================================================= */
const gotTheLock = app.requestSingleInstanceLock()
if (!gotTheLock) {
  console.log("Another instance of Vision Monitor is already running. Focusing existing window and quitting duplicate instance.")
  app.quit()
  process.exit(0)
}

let mainWindow = null
let splashWindow = null
let backendProcess = null
let isQuitting = false
let isInferenceRunning = false  // tracks renderer inference state for close guard

app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  }
})

app.disableHardwareAcceleration()

process.on("uncaughtException", (err) => {
  console.error("UNCAUGHT EXCEPTION:", err)
})

process.on("unhandledRejection", (err) => {
  console.error("UNHANDLED REJECTION:", err)
})

const API_BASE = "http://127.0.0.1:8000"
const isDev = !app.isPackaged

/* =========================================================
   AUTHORITATIVE USER DATA DIRECTORY & PID MARKER
========================================================= */
function getUserDataDir() {
  if (process.platform === "win32") {
    const root = process.env.LOCALAPPDATA || process.env.APPDATA || path.join(process.env.USERPROFILE || "", "AppData", "Local")
    return path.join(root, "Vision-Monitor")
  }
  const xdg = process.env.XDG_STATE_HOME || path.join(process.env.HOME || "", ".local", "state")
  return path.join(xdg, "vision-monitor")
}

const DATA_DIR = getUserDataDir()
const PID_FILE = path.join(DATA_DIR, "backend.pid")

function saveBackendPid(pid) {
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true })
    fs.writeFileSync(PID_FILE, String(pid), "utf8")
  } catch (err) {
    console.error("Could not write backend.pid:", err.message)
  }
}

function clearBackendPid() {
  try {
    if (fs.existsSync(PID_FILE)) {
      fs.unlinkSync(PID_FILE)
    }
  } catch (err) {
    console.error("Could not remove backend.pid:", err.message)
  }
}

function killProcessTreeSync(pid) {
  if (!pid) return
  try {
    if (process.platform === "win32") {
      execSync(`taskkill /F /T /PID ${pid}`, { stdio: "ignore" })
    } else {
      process.kill(-pid, "SIGKILL")
    }
  } catch (e) {
    try {
      process.kill(pid, "SIGKILL")
    } catch (e2) {}
  }
}

async function cleanupStaleBackend() {
  if (!fs.existsSync(PID_FILE)) return

  try {
    const oldPidStr = fs.readFileSync(PID_FILE, "utf8").trim()
    const oldPid = parseInt(oldPidStr, 10)
    if (!oldPid || isNaN(oldPid)) {
      clearBackendPid()
      return
    }

    let isAlive = false
    try {
      process.kill(oldPid, 0)
      isAlive = true
    } catch (e) {
      isAlive = false
    }

    if (isAlive) {
      console.log(`Found previous backend process with PID ${oldPid}. Terminating process tree...`)
      killProcessTreeSync(oldPid)
      // Wait up to 4s for port 8000 to actually be released (PyInstaller + CUDA
      // torch take longer than 500ms to teardown all socket handles on Windows)
      const deadline = Date.now() + 4000
      while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 300))
        const free = await checkPortAvailable(8000)
        if (free) break
      }
    }
  } catch (err) {
    console.error("Error inspecting stale backend PID:", err.message)
  } finally {
    clearBackendPid()
  }
}

function checkPortAvailable(port) {
  return new Promise((resolve) => {
    const s = netSocket.createServer()
    s.once("error", () => resolve(false))
    s.once("listening", () => {
      s.close()
      resolve(true)
    })
    s.listen(port, "127.0.0.1")
  })
}

/* =========================================================
   APP PROTOCOL
========================================================= */
protocol.registerSchemesAsPrivileged([
  {
    scheme: "app",
    privileges: {
      secure: true,
      standard: true,
      corsEnabled: true,
      supportFetchAPI: true,
    },
  },
])

function getStaticPath() {
  return path.join(__dirname, "..", "dist")
}

/* =========================================================
   SPLASH WINDOW
========================================================= */
function createSplashWindow() {
  const iconPath = process.platform === "win32"
    ? path.join(__dirname, "..", "public", "icon.ico")
    : path.join(__dirname, "..", "public", "icon.png")

  splashWindow = new BrowserWindow({
    width: 420,
    height: 320,
    frame: false,
    transparent: false,
    alwaysOnTop: true,
    center: true,
    resizable: false,
    movable: false,
    fullscreenable: false,
    backgroundColor: "#0B1020",
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
  })

  splashWindow.loadURL(`
    data:text/html;charset=UTF-8,
    <html>
      <body style="
        margin:0;
        display:flex;
        justify-content:center;
        align-items:center;
        flex-direction:column;
        background:#0B1020;
        color:white;
        font-family:sans-serif;
        height:100vh;
      ">
        <h1 style="margin-bottom:10px;">
          Vision Monitor
        </h1>

        <p id="status-msg" style="opacity:0.7; transition: opacity 0.3s;">
          Starting backend services...
        </p>

        <div style="
          margin-top:20px;
          width:220px;
          height:6px;
          background:#1E293B;
          border-radius:999px;
          overflow:hidden;
        ">
          <div style="
            width:40%;
            height:100%;
            background:#06B6D4;
            animation: loading 1s infinite;
          "></div>
        </div>

        <style>
          @keyframes loading {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(350%); }
          }
        </style>

        <script>
          // Update splash text to reflect model-loading phase
          setTimeout(() => {
            const el = document.getElementById('status-msg');
            if (el) el.textContent = 'Loading AI model into GPU...';
          }, 2500);
        </script>
      </body>
    </html>
  `)
}

/* =========================================================
   MAIN WINDOW
========================================================= */
function createWindow() {
  const iconPath = process.platform === "win32"
    ? path.join(__dirname, "..", "public", "icon.ico")
    : path.join(__dirname, "..", "public", "icon.png")

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    show: false,
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
      allowRunningInsecureContent: true,
    },
  })

  mainWindow.webContents.on("did-fail-load", (event, code, desc) => {
    console.error("FAILED TO LOAD:", code, desc)
  })

  mainWindow.webContents.on("render-process-gone", (event, details) => {
    console.error("RENDER PROCESS GONE:", details)
  })

  // Desktop-app close guard: show a native dialog if inference is running
  mainWindow.on("close", async (event) => {
    if (isQuitting) return  // already confirmed, let it close
    if (isInferenceRunning) {
      event.preventDefault()
      const { response } = await dialog.showMessageBox(mainWindow, {
        type: "warning",
        buttons: ["Keep Running", "Stop & Close"],
        defaultId: 0,
        cancelId: 0,
        title: "Inference Is Running",
        message: "Inference is currently active.",
        detail: "Closing now will stop the inference session. Do you want to continue?",
      })
      if (response === 1) {
        // User confirmed: stop backend and quit
        isQuitting = true
        await stopBackend()
        app.quit()
      }
      // response === 0: user clicked "Keep Running" — do nothing, window stays open
    }
  })

  mainWindow.on("closed", () => {
    console.log("Main window closed")
  })

  if (!app.isPackaged) {
    mainWindow.loadURL("http://localhost:5173")
    mainWindow.webContents.openDevTools()
  } else {
    const indexPath = path.join(__dirname, "..", "dist", "index.html")
    console.log("Loading packaged index.html from:", indexPath)
    mainWindow.loadFile(indexPath)
  }

  mainWindow.once("ready-to-show", () => {
    if (splashWindow) {
      splashWindow.destroy()
      splashWindow = null
    }

    mainWindow.show()
  })

  return mainWindow
}

/* =========================================================
   KILL WHATEVER HOLDS PORT 8000 (Windows netstat method)
========================================================= */
function killPortHolderSync(port) {
  if (process.platform !== "win32") return
  try {
    // netstat -ano gives lines like: TCP  127.0.0.1:8000  ...  LISTENING  <pid>
    const out = execSync(`netstat -ano -p TCP 2>nul`, { encoding: "utf8", timeout: 3000 })
    const lines = out.split("\n")
    for (const line of lines) {
      if (line.includes(`:${port}`) && line.includes("LISTENING")) {
        const parts = line.trim().split(/\s+/)
        const pid = parseInt(parts[parts.length - 1], 10)
        if (pid && !isNaN(pid) && pid !== process.pid) {
          console.log(`Force-killing PID ${pid} which holds port ${port}`)
          try { execSync(`taskkill /F /T /PID ${pid}`, { stdio: "ignore" }) } catch (e) {}
        }
      }
    }
  } catch (e) {
    // netstat failed — non-fatal, continue anyway
  }
}

/* =========================================================
   START BACKEND
========================================================= */
async function startBackend() {
  if (!app.isPackaged) {
    console.log("Development mode - backend handled separately")
    return
  }

  // 1. Clean up stale backend recorded in PID marker (waits for port release)
  await cleanupStaleBackend()

  // 2. Pre-flight check on port 8000
  let portFree = await checkPortAvailable(8000)
  if (!portFree) {
    // Port still occupied — try to reuse if the existing backend is healthy
    try {
      const res = await axios.get(`${API_BASE}/health`, { timeout: 2000 })
      if (res.data && res.data.status === "ready") {
        console.log("Existing backend is already healthy and responsive. Reusing instance.")
        return
      }
    } catch (e) {}

    // Health check failed — the old process is dying but still holds the port.
    // Force-kill whatever is listening on 8000, then wait for it to free up.
    console.warn("Port 8000 occupied and unhealthy — force-clearing it...")
    killPortHolderSync(8000)
    // Wait up to 5 seconds for the port to become free
    const deadline = Date.now() + 5000
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 300))
      portFree = await checkPortAvailable(8000)
      if (portFree) break
    }
    if (!portFree) {
      console.error("Port 8000 still occupied after forced cleanup — backend may fail to start.")
    }
  }

  const backendExecutable = process.platform === "win32" ? "backend.exe" : "backend"
  const backendPath = path.join(process.resourcesPath, "backend", backendExecutable)

  console.log("Starting backend:", backendPath)

  if (!fs.existsSync(backendPath)) {
    console.error("Backend executable not found:", backendPath)
    return
  }

  if (process.platform !== "win32") {
    try { fs.chmodSync(backendPath, 0o755) } catch (e) {}
  }

  backendProcess = spawn(backendPath, [], {
    cwd: path.dirname(backendPath),
    shell: false,
    detached: process.platform !== "win32",
    windowsHide: true,
  })

  if (backendProcess && backendProcess.pid) {
    saveBackendPid(backendProcess.pid)
  }

  backendProcess.on("error", (error) => {
    console.error("Could not launch bundled backend:", error)
    clearBackendPid()
  })

  backendProcess.on("exit", (code, signal) => {
    console.log("Bundled backend exited:", { code, signal })
    backendProcess = null
    clearBackendPid()
  })
}

/* =========================================================
   WAIT FOR BACKEND
========================================================= */
async function waitForBackend() {
  let backendReady = false
  const startTime = Date.now()
  // Keep polling until /health returns {status: "ready"}.
  // The backend returns 503 {status: "starting"} while the ML model is loading.
  // Hard cap of 3 minutes only to detect a crashed backend — not used in normal operation.
  const HARD_CAP_MS = 180000

  while (!backendReady) {
    try {
      const res = await axios.get(`${API_BASE}/health`, {
        timeout: 2000,
        // Accept 503 as a valid response (model still loading) — don't throw
        validateStatus: (status) => status === 200 || status === 503,
      })

      if (res.data && res.data.status === "ready") {
        backendReady = true
        console.log("Backend ready confirmed — ML model fully loaded.")
        break
      } else {
        // status === "starting" — model is still loading, keep polling silently
        console.log(`Backend starting: ${res.data && res.data.message || 'loading...'}`)
      }
    } catch (err) {
      // Connection refused — process still booting
    }
    if (Date.now() - startTime > HARD_CAP_MS) {
      console.warn(`Backend hard cap (${HARD_CAP_MS / 1000}s) reached; proceeding anyway.`)
      break
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
}


/* =========================================================
   STOP BACKEND
========================================================= */
async function stopBackend() {
  const pidToStop = backendProcess ? backendProcess.pid : null

  // 1. Tell backend to stop active cameras/inference
  try {
    await axios.post(`${API_BASE}/oak/oak/stop`, {}, { timeout: 800 })
  } catch (err) {}

  // 2. Request graceful backend shutdown
  try {
    await axios.post(`${API_BASE}/api/system/shutdown`, {}, { timeout: 800 })
  } catch (err) {}

  // 3. Grace wait — 1500ms so FastAPI/uvicorn can flush sockets and release port 8000
  //    before we force-kill. 500ms was too short on Windows: the port stayed bound
  //    for another 1-2s after the process exited, causing "backend disconnected" on
  //    immediate relaunch.
  await new Promise((resolve) => setTimeout(resolve, 1500))

  // 4. Force process tree cleanup if still alive
  if (pidToStop) {
    try {
      process.kill(pidToStop, 0)
      console.log(`Backend PID ${pidToStop} still running; terminating process tree...`)
      killProcessTreeSync(pidToStop)
    } catch (e) {
      // Already exited cleanly
    }
  }

  backendProcess = null
  clearBackendPid()
}

/* =========================================================
   APP READY
========================================================= */
app.whenReady().then(async () => {
  if (!isDev) {
    const staticPath = getStaticPath()

    protocol.handle("app", (request) => {
      const url = new URL(request.url)
      let pathname = url.pathname

      if (pathname === "/" || pathname === "") {
        pathname = "/index.html"
      } else if (!path.extname(pathname)) {
        pathname = pathname.replace(/\/?$/, "/index.html")
      }

      const filePath = path.join(staticPath, pathname)
      return net.fetch(`file://${filePath}`).catch(() =>
        net.fetch(`file://${path.join(staticPath, "index.html")}`)
      )
    })
  }

  console.log("Electron app ready")
  console.log("NODE_ENV:", process.env.NODE_ENV)
  console.log("app.isPackaged:", app.isPackaged)

  createSplashWindow()
  console.log("Splash window created")

  await startBackend()
  console.log("Backend start triggered")

  await waitForBackend()
  console.log("Backend ready confirmed")

  createWindow()
  console.log("Main window created")
})

/* =========================================================
   IPC HANDLERS
========================================================= */
const logFilePath = path.join(app.getPath("userData"), "log.txt")

ipcMain.handle("write-log", (_event, message) => {
  const timestamp = new Date().toISOString()
  const line = `[${timestamp}] ${message}\n`
  fs.appendFileSync(logFilePath, line, "utf8")
})

// Renderer calls this to keep the main process aware of inference state
// so the close-guard dialog can be shown correctly
ipcMain.on("set-inference-running", (_event, running) => {
  isInferenceRunning = !!running
})

ipcMain.handle("select-folder", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  })

  if (result.canceled) return null
  return result.filePaths[0]
})

ipcMain.handle("select-file", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [
      {
        name: "Videos",
        extensions: ["mp4", "avi", "mov", "mkv", "wmv", "flv", "m4v", "webm"],
      },
    ],
  })

  if (result.canceled) return null
  return result.filePaths[0]
})

/* =========================================================
   SAFE EXIT & LIFECYCLE
========================================================= */
app.on("before-quit", async (event) => {
  if (isQuitting) return
  event.preventDefault()
  isQuitting = true

  console.log("Stopping backend before quit...")
  await stopBackend()
  app.quit()
})

app.on("window-all-closed", async () => {
  await stopBackend()
  if (process.platform !== "darwin") {
    app.quit()
  }
})

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})
