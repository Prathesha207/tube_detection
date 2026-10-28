import av
import cv2
import numpy as np
# D:\tube_detection\depth_videos\20261027_145823_862_depth.mkv

video_path = "d:/tube_detection/depth_videos/20261027_145823_862_depth.mkv"
print(f"Opening {video_path}...")

container = av.open(video_path)
stream = container.streams.video[0]

win_name = "OAK Depth Stream (Press 'q' to quit)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 960, 540)

print("Starting playback. Look for the window titled 'OAK Depth Stream' on your taskbar/screen.")

for idx, frame in enumerate(container.decode(stream)):
    # 1. Raw 16-bit depth in millimeters (H, W uint16)
    depth_mm = frame.to_ndarray()

    # 2. Normalize 300mm–1500mm range into 0–255 for visualization
    depth_clipped = np.clip(depth_mm, 300, 1500)
    depth_8bit = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Apply color heatmap (blue = close, red = far)
    colorized_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)

    cv2.imshow(win_name, colorized_depth)

    # 30 ms delay for smooth ~30 fps playback and GUI rendering
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC
        print("\nPlayback stopped by user.")
        break

    if idx % 30 == 0:
        print(f"Playing frame {idx} (depth: min={depth_mm.min()}mm, max={depth_mm.max()}mm)...", end='\r')

cv2.destroyAllWindows()
print("\nDone.")

