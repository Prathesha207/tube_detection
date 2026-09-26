import { create } from "zustand";
import { persist } from "zustand/middleware";
import { cameraService } from "../components/service/cameraService";
import { configService } from "../components/service/configService";
import { showToast } from "../lib/toast";

/* ================= TYPES ================= */

export type CameraControlKey =
    | "exposure"
    | "gain"
    | "focus"
    | "brightness"
    | "contrast";

export type Camera = {
    id: number;
    name: string;
    ip: string;
    resolution: string;
    rotation_angle: number;
    is_enabled: boolean;

    control_mode: "auto" | "manual";
    exposure: number;
    gain: number;
    focus: number;
    brightness: number;
    contrast: number;

    auto_exposure: boolean;
    auto_focus: boolean;
};

export type CameraConfig = {
    name?: string;
    ip_address?: string;
    resolution?: string;
    fps?: number;
    rotation_angle?: number;

    control_mode?: "auto" | "manual";

    exposure?: number;
    gain?: number;
    focus?: number;
    brightness?: number;
    contrast?: number;

    auto_exposure?: boolean;
    auto_focus?: boolean;
    recording_video_path?: string;
    recording_video_testing_path?: string;
    stream_url?: string;
    is_inference_enabled?: boolean;
    roi_enabled?: boolean;

    is_enabled?: boolean;
};

export type BasicConfig = {
    camera?: CameraConfig;
    training?: {
        feature_window?: number;
        anomaly_scorer_nn_count?: number;
    };
    inference?: {
        trained_model?: string;
        min_detection_area?: number;
        use_dynamic_threshold?: boolean;
        processed_video?: boolean;
        raw_video?: boolean;
        save_failed_frame?: boolean;
        mode?: "production" | "testing";
        inference_video_path?: string;
        inference_video_testing_path?: string;
        model1_frame_count?: number;
        model2_frame_count?: number;
        model3_frame_count?: number;
        model1_pass_frames?: number;
        model2_pass_frames?: number;
        model3_pass_frames?: number;
        model2_start_skip_frame?: number;
        model3_start_skip_frame?: number;
        recording_format?: "MJPEG" | "FFV1";
        inference_format?: "MJPEG" | "FFV1";
    };
};

/* ================= DEFAULTS ================= */

const DEFAULT_CONTROLS = {
    exposure: 10,
    gain: 100,
    focus: 120,
    brightness: 0,
    contrast: 50,
};

/* ================= STORE ================= */

type StoreState = {
    /* CAMERA */
    camera: Camera | null;
    cameraLoading: boolean;

    /* CONFIG */
    config: BasicConfig | null;
    configLoading: boolean;
    isFetched: boolean;
    error: string | null;

    /* MODE & TESTING PATHS */
    systemMode: "testing" | "production";
    inference_video_testing_path: string;
    recording_video_testing_path: string;

    /* CONTROLS */
    exposure: number;
    gain: number;
    focus: number;
    brightness: number;
    contrast: number;

    /* CONNECTION */
    isConnecting: boolean;
    retryCount: number;
    maxRetries: number;

    /* ACTIONS */

    // controls
    setControl: (key: CameraControlKey, value: number) => void;
    setBulkControls: (values: Partial<Record<CameraControlKey, number>>) => void;
    resetControls: () => void;

    // camera
    loadCamera: () => Promise<void>;
    toggleCameraPower: () => Promise<void>;
    ensureCameraRunning: () => Promise<void>;

    // config
    fetchConfig: (force?: boolean) => Promise<void>;
    saveConfig: (data: BasicConfig) => Promise<boolean>;

    // mode & testing paths
    setSystemMode: (mode: "testing" | "production") => Promise<void>;
    setTestingPaths: (paths: { inference?: string; recording?: string }) => void;
};

/* ================= STORE ================= */

export const useCameraSystemStore = create<StoreState>()(
    persist(
        (set, get) => ({
            /* ================= STATE ================= */

            camera: null,
            cameraLoading: false,

            config: null,
            configLoading: false,
            isFetched: false,
            error: null,

            systemMode: "production",
            inference_video_testing_path: "",
            recording_video_testing_path: "",

            ...DEFAULT_CONTROLS,

            isConnecting: false,
            retryCount: 0,
            maxRetries: 5,

            /* ================= CONTROLS ================= */

            setControl: (key, value) => set({ [key]: value }),

            setBulkControls: (values) => set({ ...values }),

            resetControls: () => set({ ...DEFAULT_CONTROLS }),

            /* ================= MODE & TESTING PATHS ================= */

            setSystemMode: async (mode) => {
                try {
                    await cameraService.updateInferenceMode(mode);
                    set({ systemMode: mode });
                    showToast(
                        "success",
                        mode === "testing"
                            ? "Switched to Testing Mode"
                            : "Switched to Production Mode"
                    );
                } catch (err) {
                    console.error("Failed to update system mode:", err);
                    showToast("error", "Failed to update system mode");
                }
            },

            setTestingPaths: (paths) => set({
                ...(paths.inference !== undefined && { inference_video_testing_path: paths.inference }),
                ...(paths.recording !== undefined && { recording_video_testing_path: paths.recording }),
            }),

            /* ================= CAMERA ================= */

            ensureCameraRunning: async () => {
                const { camera, isConnecting, retryCount, maxRetries } = get();

                if (!camera || !camera.is_enabled) return;
                if (isConnecting) return;

                try {
                    set({ isConnecting: true });

                    const health = await cameraService.health();

                    if (health?.camera_running === true) {
                        set({ isConnecting: false, retryCount: 0 });
                        return;
                    }

                    

                    await cameraService.start();

                    set({ isConnecting: false, retryCount: 0 });
                } catch (err: any) {
                    console.error("Camera start failed:", err);
                    const isDeviceInUse = err?.message?.toLowerCase().includes("already used") || err?.message?.toLowerCase().includes("in use");

                    if (isDeviceInUse) {
                        set({ isConnecting: false });
                        showToast(
                            "error",
                            err.message || "Camera device is already in use by another application/process."
                        );
                        return;
                    }

                    if (retryCount < maxRetries) {
                        const delay = Math.min(1000 * 2 ** retryCount, 10000);

                        set({
                            retryCount: retryCount + 1,
                            isConnecting: false,
                        });

                        setTimeout(() => {
                            const { camera } = get();

                            // 🛑 STOP retry if camera disabled
                            if (!camera?.is_enabled) return;

                            get().ensureCameraRunning();
                        }, delay);
                    } else {
                        console.error("Max retries reached");

                        set({ isConnecting: false });

                        showToast(
                            "error",
                            "Camera failed to start after multiple attempts"
                        );
                    }
                }
            },

            loadCamera: async () => {
                set({ cameraLoading: true });

                try {
                    const data = await cameraService.checkCamera();

                    if (!data?.length) {
                        set({ camera: null, cameraLoading: false });
                        return;
                    }

                    const cam: Camera = {
                        id: data[0].id,
                        name: data[0].name,
                        ip: data[0].ip_address,
                        resolution: data[0].resolution,
                        rotation_angle: data[0].rotation_angle,
                        is_enabled: data[0].is_enabled,

                        control_mode: data[0].control_mode,
                        exposure: data[0].exposure ?? 0,
                        gain: data[0].gain ?? 0,
                        focus: data[0].focus ?? 0,
                        brightness: data[0].brightness ?? 0,
                        contrast: data[0].contrast ?? 0,

                        auto_exposure: data[0].auto_exposure ?? true,
                        auto_focus: data[0].auto_focus ?? true,
                    };

                    set({
                        camera: cam,
                        cameraLoading: false,

                        // sync controls
                        exposure: cam.exposure,
                        gain: cam.gain,
                        focus: cam.focus,
                        brightness: cam.brightness,
                        contrast: cam.contrast,
                    });

                    // 🔥 auto ensure running
                    if (cam.is_enabled) {
                        get().ensureCameraRunning();
                    }
                } catch (err) {
                    console.error("Load camera failed:", err);
                    set({ cameraLoading: false });
                }
            },

            toggleCameraPower: async () => {
                const { camera } = get();
                if (!camera) return;

                try {
                    set({ cameraLoading: true });

                    if (camera.is_enabled) {
                        // disable
                        await cameraService.disableCamera(String(camera.id));
                        await cameraService.stop?.();

                        showToast("success", "Camera disconnected");
                    } else {
                        // enable
                        await cameraService.enableCamera(String(camera.id));

                        showToast("success", "Camera connected");
                    }

                    // reload fresh state
                    await get().loadCamera();

                    const updatedCamera = get().camera;

                    // 🔥 ensure running only if enabled
                    if (updatedCamera?.is_enabled) {
                        setTimeout(() => {
                            get().ensureCameraRunning();
                        }, 500);
                    }
                } catch (err) {
                    console.error("Camera toggle failed:", err);
                    showToast("error", "Camera operation failed");
                } finally {
                    set({ cameraLoading: false });
                }
            },

            /* ================= CONFIG ================= */

            fetchConfig: async (force = false) => {
                const { isFetched, configLoading } = get();

                if (configLoading) return;
                if (isFetched && !force) return;

                set({ configLoading: true, error: null });

                try {
                    const data = await configService.getConfig();

                    set({
                        config: data,
                        configLoading: false,
                        isFetched: true,
                        // Always sync systemMode from the API — prevents a stale
                        // localStorage value from overriding the backend's source of truth.
                        ...(data?.inference?.mode && { systemMode: data.inference.mode }),
                    });
                } catch (err: any) {
                    set({
                        configLoading: false,
                        error: err?.message || "Failed to fetch config",
                    });
                }
            },

            saveConfig: async (data) => {
                set({ configLoading: true, error: null });

                try {
                    const saved = await configService.saveConfig(data);

                    set({
                        config: saved,
                        configLoading: false,
                        isFetched: true,
                    });

                    return true;
                } catch (err: any) {
                    set({
                        configLoading: false,
                        error: err?.message || "Failed to save config",
                    });

                    return false;
                }
            },
        }),
        {
            name: "camera-system",

            /* persist controls + mode + testing paths */
            partialize: (state) => ({
                exposure: state.exposure,
                gain: state.gain,
                focus: state.focus,
                brightness: state.brightness,
                contrast: state.contrast,
                systemMode: state.systemMode,
                inference_video_testing_path: state.inference_video_testing_path,
                recording_video_testing_path: state.recording_video_testing_path,
            }),
        }
    )
);