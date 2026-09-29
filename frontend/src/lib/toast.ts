export type ToastType = "success" | "error" | "info" | "warning";

export function showToast(type: ToastType, message: string) {
  console.log(`[TOAST:${type.toUpperCase()}] ${message}`);
  
  // Custom DOM toast notification event for UI listeners
  if (typeof window !== "undefined") {
    window.dispatchEvent(
      new CustomEvent("app-toast", {
        detail: { type, message, timestamp: Date.now() },
      })
    );
  }
}
