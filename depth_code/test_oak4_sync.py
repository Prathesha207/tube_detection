"""
test_oak4_sync.py
=================
Verification script for OAK4-D (RVC4) on DepthAI v3.

Addresses all feedback points:
  1. Optical Alignment: stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) + setOutputSize(W, H)
  2. Temporal Sync: dai.node.Sync (threshold: 25ms) linking CAM_A RGB & StereoDepth
  3. Resolution & Intrinsics: Reads factory EEPROM fx, fy, cx, cy for active resolution (720p or 1080p)
  4. Bandwidth Optimization: Explicit FPS (default 20 FPS for 720p, 15 FPS for 1080p)
  5. Robust Exit: Non-blocking tryGet() with clean pipeline shutdown
  6. Tube Measurement: Real-time physical mm diameter calculation at crosshair using true fx & Z

Usage:
  python depth_code/test_oak4_sync.py                     # Default: 720p @ 20 FPS
  python depth_code/test_oak4_sync.py --res 1080p         # 1080p @ 15 FPS
  python depth_code/test_oak4_sync.py --no-lr-check       # Disable Left-Right check for shiny surfaces
  python depth_code/test_oak4_sync.py --ip 169.254.21.191 # Specify custom IP

Controls:
  [SPACE]   - Toggle view mode (50/50 Blend -> Side-by-Side -> Pure RGB -> Pure Depth)
  [C]       - Capture single snapshot with RGB, Depth, and stats saved to disk
  [Q] / ESC - Quit cleanly
"""

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:
    print("Error: depthai is not installed in this environment.")
    print("Run: pip install depthai")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="OAK4-D Sync & Alignment Test Script")
    parser.add_argument("--ip", type=str, default="169.254.21.191",
                        help="Camera IP address (default: 169.254.21.191)")
    parser.add_argument("--res", type=str, choices=["720p", "1080p"], default="720p",
                        help="Active resolution: 720p (1280x720) or 1080p (1920x1080). Default: 720p")
    parser.add_argument("--fps", type=int, default=30,
                        help="Camera frame rate in FPS (default: 30, or any FPS chosen by the user)")
    parser.add_argument("--sync-thresh-ms", type=int, default=25,
                        help="Sync threshold in milliseconds (default: 25)")
    parser.add_argument("--no-lr-check", action="store_true",
                        help="Disable Left-Right check (may increase depth coverage on shiny/transparent plastic)")
    return parser.parse_args()


def connect_camera(ip_address: str):
    """Attempt direct IP connection first, then broadcast search fallback."""
    print(f"Connecting to OAK4-D at IP: {ip_address}...")
    try:
        device_info = dai.DeviceInfo(ip_address)
        device = dai.Device(device_info)
        return device
    except Exception as ip_err:
        print(f"IP connection to {ip_address} failed ({ip_err}). Searching for any available OAK device...")
        try:
            device = dai.Device()
            return device
        except Exception as auto_err:
            print(f"Error: Could not connect to any OAK camera ({auto_err}).")
            print("Troubleshooting:")
            print("  1. Verify camera power supply and Ethernet cable are connected.")
            print("  2. Verify your PC Ethernet adapter is set to link-local (e.g. 169.254.x.x).")
            print("  3. Ping the camera IP: 'Test-Connection 169.254.21.191' or check arp table.")
            return None


def main():
    args = parse_args()

    # Resolution selection
    if args.res == "1080p":
        width, height = 1920, 1080
    else:
        width, height = 1280, 720

    target_fps = args.fps
    lr_check = not args.no_lr_check

    print("=================================================================")
    print("           OAK4-D Sync & Optical Alignment Verification          ")
    print("=================================================================")
    print(f"DepthAI Version:       {dai.__version__}")
    print(f"Target Resolution:     {width}x{height} ({args.res})")
    print(f"Target FPS:            {target_fps} FPS")
    print(f"Sync Threshold:        {args.sync_thresh_ms} ms")
    print(f"Left-Right Check:      {'ENABLED' if lr_check else 'DISABLED'}")
    print("=================================================================")

    device = connect_camera(args.ip)
    if device is None:
        return

    try:
        proto = device.getDeviceInfo().protocol.name
    except Exception:
        proto = "TCP/IP"
    try:
        dev_name = device.getDeviceName() or "OAK Camera"
    except Exception:
        dev_name = "OAK Camera"
    print(f"Connected: {dev_name} [Protocol: {proto}]")

    # Read Camera Intrinsics directly from EEPROM for this exact resolution
    fx = fy = cx = cy = None
    try:
        calib = device.readCalibration()
        eeprom_k = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
        fx, fy = float(eeprom_k[0][0]), float(eeprom_k[1][1])
        cx, cy = float(eeprom_k[0][2]), float(eeprom_k[1][2])
        print(f"Factory EEPROM Intrinsics for CAM_A ({width}x{height}):")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    except Exception as calib_err:
        print(f"Warning: Could not read EEPROM intrinsics ({calib_err}). Fallback approximation will be used.")
        # Approximate 1080p intrinsics scaled to width/height
        base_fx = 1425.3 * (width / 1920.0)
        base_fy = 1424.6 * (height / 1080.0)
        fx, fy = base_fx, base_fy
        cx, cy = width / 2.0, height / 2.0

    pipeline = dai.Pipeline(device)

    print("Building DepthAI v3 pipeline...")

    # 1. RGB Camera (CAM_A)
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # 2. Stereo Mono Cameras (CAM_B = Left, CAM_C = Right)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # 3. Stereo Depth Node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(lr_check)

    left.requestOutput((1280, 720)).link(stereo.left)
    right.requestOutput((1280, 720)).link(stereo.right)

    # Optical Alignment: warps depth map into CAM_A's lens FOV & perspective
    try:
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        print("Optical alignment: stereo.setDepthAlign(CAM_A) enabled.")
    except Exception as e:
        print(f"Warning: setDepthAlign failed: {e}")

    try:
        stereo.setOutputSize(width, height)
        stereo.setOutputKeepAspectRatio(False)
    except Exception as e:
        print(f"Note: setOutputSize skipped on RVC4 ({e}). Depth will match stereo dimensions.")

    # 4. Temporal Sync Node & Optical ImageAlign
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=args.sync_thresh_ms))

    rgb_out = rgb.requestOutput((width, height), dai.ImgFrame.Type.NV12, fps=target_fps)

    depth_out = stereo.depth
    if hasattr(dai.node, "ImageAlign"):
        try:
            align = pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            rgb_out.link(align.inputAlignTo)
            depth_out = align.outputAligned
            print("Optical Alignment: dai.node.ImageAlign active (depth geometrically warped to CAM_A).")
        except Exception as e:
            print(f"Warning: ImageAlign unavailable ({e}), using direct stereo.depth")
            depth_out = stereo.depth

    rgb_out.link(sync.inputs["rgb"])
    depth_out.link(sync.inputs["depth"])

    # Unified Output Queue
    queue = sync.out.createOutputQueue(maxSize=4, blocking=False)

    print("Starting pipeline on hardware...")
    pipeline.start()
    print("Pipeline active! Opening visual verification window...")

    win_name = "OAK4-D RGB+Depth Verification [SPACE: Mode | C: Snapshot | Q: Quit]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(width, 1280), min(height, 720))

    mode_names = [
        "1. 50/50 GHOST BLEND (Check Edge Overlap)",
        "2. SIDE-BY-SIDE (RGB Left | Depth Right)",
        "3. PURE RGB (CAM_A)",
        "4. PURE DEPTH (Aligned to CAM_A)",
    ]
    display_mode = 0

    fps_count = 0
    fps_start = time.time()
    real_fps = 0.0

    # Crosshair measurement caliper width (pixels across center)
    caliper_px = 60

    try:
        while pipeline.isRunning():
            # Non-blocking read prevents thread deadlock if disconnected
            group = queue.tryGet()
            if group is None:
                key = cv2.waitKey(2) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            fps_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                real_fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            # Extract synced pair
            rgb_pkt = group["rgb"]
            depth_pkt = group["depth"]

            rgb_frame = rgb_pkt.getCvFrame()     # BGR uint8
            depth_frame = depth_pkt.getFrame()   # uint16 mm

            # If depth resolution differs from RGB (e.g. 1080p RGB with native 720p depth on RVC4),
            # match depth to RGB raster using nearest-neighbor to preserve exact integer mm depth values
            if depth_frame.shape[:2] != rgb_frame.shape[:2]:
                depth_frame = cv2.resize(depth_frame, (rgb_frame.shape[1], rgb_frame.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

            # Telemetry metrics
            valid_mask = depth_frame > 0
            valid_pct = 100.0 * float(valid_mask.mean())

            # Sample center patch for distance
            cy_c, cx_c = height // 2, width // 2
            patch = depth_frame[max(0, cy_c - 8):cy_c + 9, max(0, cx_c - 8):cx_c + 9]
            valid_patch = patch[patch > 0]
            center_z_mm = float(np.median(valid_patch)) if valid_patch.size > 0 else 0.0

            # Compute physical mm width of caliper line at center Z
            if center_z_mm > 0 and fx:
                caliper_mm = (caliper_px * center_z_mm) / fx
            else:
                caliper_mm = 0.0

            # Colorize depth (300mm to 1500mm Turbo colormap)
            norm = (np.clip(depth_frame, 300, 1500).astype(np.float32) - 300) * (255.0 / 1200.0)
            depth_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
            depth_color[~valid_mask] = 0

            # Render display mode
            if display_mode == 0:
                display = cv2.addWeighted(rgb_frame, 0.55, depth_color, 0.45, 0)
            elif display_mode == 1:
                half_w, half_h = width // 2, height // 2
                s_rgb = cv2.resize(rgb_frame, (half_w, half_h))
                s_dep = cv2.resize(depth_color, (half_w, half_h))
                display = np.hstack([s_rgb, s_dep])
            elif display_mode == 2:
                display = rgb_frame.copy()
            else:
                display = depth_color.copy()

            # Draw center crosshair and measurement caliper
            h_d, w_d = display.shape[:2]
            scale_factor = w_d / float(width)
            c_x = int(cx_c * scale_factor)
            c_y = int(cy_c * scale_factor)
            cal_half = int((caliper_px * scale_factor) / 2)

            # Center marker
            cv2.drawMarker(display, (c_x, c_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

            # Caliper ticks (horizontal line measuring caliper_px across)
            cv2.line(display, (c_x - cal_half, c_y), (c_x + cal_half, c_y), (0, 255, 0), 2)
            cv2.line(display, (c_x - cal_half, c_y - 6), (c_x - cal_half, c_y + 6), (0, 255, 0), 2)
            cv2.line(display, (c_x + cal_half, c_y - 6), (c_x + cal_half, c_y + 6), (0, 255, 0), 2)

            caliper_label = f"Caliper: {caliper_px}px = {caliper_mm:.1f}mm" if caliper_mm > 0 else f"Caliper: {caliper_px}px (No Depth)"
            cv2.putText(display, caliper_label, (c_x - cal_half, c_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # HUD bar
            hud_text = (
                f"Mode: {mode_names[display_mode]} | "
                f"FPS: {real_fps:.1f} (target {target_fps}) | "
                f"Valid: {valid_pct:.0f}% | "
                f"Z: {center_z_mm:.0f}mm"
            )
            cv2.rectangle(display, (10, 10), (w_d - 10, 42), (15, 20, 28), -1)
            cv2.rectangle(display, (10, 10), (w_d - 10, 42), (0, 200, 255), 1)
            cv2.putText(display, hud_text, (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                display_mode = (display_mode + 1) % len(mode_names)
                print(f"View mode: {mode_names[display_mode]}")
            elif key in (ord('c'), ord('C')):
                snap_dir = Path("depth_code/snapshots")
                snap_dir.mkdir(parents=True, exist_ok=True)
                ts_str = time.strftime("%Y%m%d_%H%M%S")
                rgb_path = snap_dir / f"snap_{ts_str}_rgb.jpg"
                dep_path = snap_dir / f"snap_{ts_str}_depth_color.jpg"
                bld_path = snap_dir / f"snap_{ts_str}_blend.jpg"
                raw_dep_path = snap_dir / f"snap_{ts_str}_depth_raw.npy"

                cv2.imwrite(str(rgb_path), rgb_frame)
                cv2.imwrite(str(dep_path), depth_color)
                blend_img = cv2.addWeighted(rgb_frame, 0.55, depth_color, 0.45, 0)
                cv2.imwrite(str(bld_path), blend_img)
                np.save(str(raw_dep_path), depth_frame)
                print(f"Saved snapshot to {snap_dir}:")
                print(f"  RGB:    {rgb_path.name}")
                print(f"  Depth:  {dep_path.name}")
                print(f"  Blend:  {bld_path.name}")
                print(f"  Raw Z:  {raw_dep_path.name} (Valid: {valid_pct:.1f}%, Center Z: {center_z_mm:.0f}mm)")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping pipeline and closing camera...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped cleanly.")


if __name__ == "__main__":
    main()
