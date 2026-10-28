import os
import sys
import time
import av
import cv2
import numpy as np
from ultralytics import YOLO

RGB_PATH = "d:/tube_detection/20261027_145823_862_video.mp4"
DEPTH_PATH = "d:/tube_detection/20261027_145823_862_depth.mkv"
MODEL_PATH = "d:/tube_detection/backend/app/ml/model/best.pt"

# Known OAK-D RGB camera focal length for 1080p (approximate: fx ~ 1500 px)
FOCAL_LENGTH_PX = 1500.0


def main():
    if not os.path.exists(RGB_PATH):
        print(f"Error: RGB video not found at {RGB_PATH}")
        return
    if not os.path.exists(DEPTH_PATH):
        print(f"Error: Depth video not found at {DEPTH_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print(f"Opening RGB video:   {RGB_PATH}")
    cap_rgb = cv2.VideoCapture(RGB_PATH)

    print(f"Opening Depth video: {DEPTH_PATH}")
    depth_container = av.open(DEPTH_PATH)
    depth_stream = depth_container.streams.video[0]
    depth_iter = depth_container.decode(depth_stream)

    win_name = "RGB + Depth Tube Analyzer (Q: Quit | Space: Pause/Resume)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1440, 405)

    print("\nControls:")
    print("  [Q] or [ESC] - Exit")
    print("  [SPACE]     - Pause / Resume")
    print("  [S]         - Step single frame while paused\n")

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ret_rgb, rgb_frame = cap_rgb.read()
            if not ret_rgb:
                print("\nReached end of RGB stream.")
                break

            try:
                depth_packet = next(depth_iter)
                depth_frame = depth_packet.to_ndarray()
            except StopIteration:
                print("\nReached end of Depth stream.")
                break

            frame_idx += 1

            # 1. Run YOLO inference on RGB frame
            t0 = time.time()
            results = model.predict(rgb_frame, imgsz=1280, conf=0.30, verbose=False)
            dt_ms = (time.time() - t0) * 1000
            r = results[0]

            # 2. Colorize Depth (clipped between 300mm and 1500mm)
            depth_clipped = np.clip(depth_frame, 300, 1500)
            depth_8u = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)

            # 3. Annotate detections on both RGB and Depth
            annotated_rgb = rgb_frame.copy()

            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Clamp coords to image boundaries
                h_img, w_img = depth_frame.shape[:2]
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_img, x2), min(h_img, y2)

                # Extract depth within the bounding box
                depth_roi = depth_frame[y1_c:y2_c, x1_c:x2_c]
                valid_depth = depth_roi[depth_roi > 0]

                if len(valid_depth) > 0:
                    z_mm = float(np.median(valid_depth))
                    # Physical width in millimeters (Formula: px * Z / fx)
                    w_px = x2 - x1
                    width_mm = (w_px * z_mm) / FOCAL_LENGTH_PX
                    dist_text = f"Z={z_mm:.0f}mm | {width_mm:.1f}mm"
                else:
                    dist_text = "Z=N/A"

                box_color = (0, 255, 0) if cls_name == "head" else (0, 165, 255)
                label = f"{cls_name.upper()} {conf:.2f} ({dist_text})"

                # Draw on RGB frame
                cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    annotated_rgb,
                    label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                )

                # Draw on Depth frame
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    depth_color,
                    dist_text,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            # 4. Add frame information banners
            cv2.putText(
                annotated_rgb,
                f"RGB Frame: {frame_idx} | YOLO: {dt_ms:.1f}ms",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                depth_color,
                "Depth Heatmap (Turbo)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            # 5. Side-by-side display (each resized to 960x540)
            small_rgb = cv2.resize(annotated_rgb, (960, 540))
            small_depth = cv2.resize(depth_color, (960, 540))
            combined = np.hstack([small_rgb, small_depth])

            cv2.imshow(win_name, combined)

            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx} | Detections: {len(r.boxes)} | Inference: {dt_ms:.1f}ms", end="\r")

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord("q") or key == 27:
            print("\nStopped by user.")
            break
        elif key == ord(" "):
            paused = not paused
            print(f"\n{'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("s") and paused:
            paused = False
            # process 1 frame then re-pause
            continue

    cap_rgb.release()
    depth_container.close()
    cv2.destroyAllWindows()
    print("\nCompleted.")


if __name__ == "__main__":
    main()
