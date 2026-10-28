import os
import sys
import time
import av
import cv2
import numpy as np
from ultralytics import YOLO

# Video paths (checks depth_videos/ folder first, then root directory)
RGB_PATH = "d:/tube_detection/depth_videos/20261027_145823_862_video.mp4"
DEPTH_PATH = "d:/tube_detection/depth_videos/20261027_145823_862_depth.mkv"
MODEL_PATH = "d:/tube_detection/backend/app/ml/model/best.pt"

# Genuine factory-calibrated focal length read directly from camera EEPROM
FOCAL_LENGTH_PX = 1425.300049

# High-contrast color palette for tubes (BGR)
TUBE_COLORS = [
    (255, 220, 0),    # Cyan / Neon Blue
    (0, 165, 255),    # Amber / Neon Orange
    (0, 255, 128),    # Spring Green
    (230, 80, 255),   # Bright Magenta
]


def draw_hud(frame, detections, focal_length_px):
    """Draw a professional semi-transparent HUD card with tube telemetry."""
    if not detections:
        return

    h_f, w_f = frame.shape[:2]
    card_w = 490
    row_h = 56
    header_h = 42
    footer_h = 42 if len(detections) >= 2 else 10
    card_h = header_h + len(detections) * row_h + footer_h
    card_x = w_f - card_w - 20
    card_y = 20

    # Semi-transparent dark slate backdrop
    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    # Border
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 70, 90), 2)

    # Header
    cv2.putText(frame, "TUBE DIAMETER TELEMETRY", (card_x + 16, card_y + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"fx = {focal_length_px:.1f}px", (card_x + card_w - 135, card_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 185, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (card_x + 12, card_y + 34), (card_x + card_w - 12, card_y + 34), (45, 65, 85), 1)

    # Rows
    curr_y = card_y + header_h
    for det in detections:
        t_id = det["id"]
        col = det["color"]
        role = det["role"]
        conf = det["conf"]
        z_mm = det["z_mm"]
        th_px = det["thickness_px"]
        w_mm = det["width_mm"]

        # Number badge pill
        cv2.rectangle(frame, (card_x + 16, curr_y + 8), (card_x + 64, curr_y + 38), col, -1)
        cv2.putText(frame, f"#{t_id}", (card_x + 24, curr_y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

        # Meta string
        meta = f"{role} ({conf*100:.0f}%) | Z: {z_mm:.0f}mm | {th_px:.1f}px"
        cv2.putText(frame, meta, (card_x + 78, curr_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 225), 1, cv2.LINE_AA)

        # Diameter result
        if z_mm > 0:
            diam_txt = f"Diameter: {w_mm:.2f} mm"
            val_col = (0, 255, 128)
        else:
            diam_txt = "Diameter: N/A (no depth)"
            val_col = (130, 130, 130)

        cv2.putText(frame, diam_txt, (card_x + 78, curr_y + 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, val_col, 1, cv2.LINE_AA)

        curr_y += row_h

    # Comparison footer
    if len(detections) >= 2:
        cv2.line(frame, (card_x + 12, curr_y + 2), (card_x + card_w - 12, curr_y + 2), (45, 65, 85), 1)
        w1 = detections[0]["width_mm"]
        w2 = detections[1]["width_mm"]
        diff = abs(w1 - w2)
        if diff < 0.5:
            verdict = f"VERDICT: IDENTICAL SIZES (Δ = {diff:.2f} mm)"
            v_col = (0, 255, 128)
        elif w1 > w2:
            verdict = f"VERDICT: TUBE #1 IS BIGGER (+{diff:.2f} mm)"
            v_col = detections[0]["color"]
        else:
            verdict = f"VERDICT: TUBE #2 IS BIGGER (+{diff:.2f} mm)"
            v_col = detections[1]["color"]

        cv2.putText(frame, verdict, (card_x + 16, curr_y + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, v_col, 1, cv2.LINE_AA)


def draw_caliper_zoom_inset(frame, raw_rgb, primary_tubes, zoom=4.0, inset_size=(460, 320)):
    """Draw a clean, dedicated 4x zoomed caliper inspection window from raw RGB."""
    if not primary_tubes:
        return

    all_x = []
    all_y = []
    for det in primary_tubes:
        x1, y1, x2, y2 = det["bbox"]
        all_x.extend([x1, x2])
        all_y.extend([y1, y2])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    h_f, w_f = frame.shape[:2]
    pad_x = 55
    pad_y = 45
    cx1 = max(0, min_x - pad_x)
    cy1 = max(0, min_y - pad_y)
    cx2 = min(w_f, max_x + pad_x)
    cy2 = min(h_f, max_y + pad_y)

    crop = raw_rgb[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return

    iw, ih = inset_size
    magnified = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_LINEAR)
    scale_x = iw / (cx2 - cx1)
    scale_y = ih / (cy2 - cy1)

    # Sort tubes top-to-bottom so badges don't collide
    sorted_tubes = sorted(primary_tubes, key=lambda d: d["bbox"][1])

    for idx, det in enumerate(sorted_tubes):
        col = det["color"]
        t_id = det["id"]
        poly = det.get("poly")
        rect = det.get("rect")
        w_mm = det["width_mm"]
        th_px = det["thickness_px"]

        # Scaled polygon
        if poly is not None:
            m_poly = poly.copy().astype(np.float32)
            m_poly[:, 0] = (m_poly[:, 0] - cx1) * scale_x
            m_poly[:, 1] = (m_poly[:, 1] - cy1) * scale_y
            m_poly_int = m_poly.astype(np.int32)

            m_over = magnified.copy()
            cv2.fillPoly(m_over, [m_poly_int], col)
            cv2.addWeighted(m_over, 0.28, magnified, 0.72, 0, magnified)
            cv2.polylines(magnified, [m_poly_int], True, col, 2, cv2.LINE_AA)

        # Caliper cross-section line
        if rect is not None:
            (cx, cy), (rw, rh), angle = rect
            pts = cv2.boxPoints(rect)
            v0 = pts[1] - pts[0]
            v1 = pts[2] - pts[1]
            l0 = np.linalg.norm(v0)
            l1 = np.linalg.norm(v1)

            u_thick = (v0 / l0) if l0 < l1 else (v1 / l1)
            half_th = (min(l0, l1) / 2.0)

            p1 = np.array([cx, cy]) - u_thick * half_th
            p2 = np.array([cx, cy]) + u_thick * half_th

            mp1 = (int((p1[0] - cx1) * scale_x), int((p1[1] - cy1) * scale_y))
            mp2 = (int((p2[0] - cx1) * scale_x), int((p2[1] - cy1) * scale_y))
            mcx = int((cx - cx1) * scale_x)
            mcy = int((cy - cy1) * scale_y)

            # Caliper line with endpoint dots
            cv2.line(magnified, mp1, mp2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(magnified, mp1, 4, (0, 0, 255), -1)
            cv2.circle(magnified, mp2, 4, (0, 0, 255), -1)

            # Measurement badge with vertical separation (top tube above, bottom tube below)
            badge_txt = f"#{t_id}: {w_mm:.2f}mm ({th_px:.1f}px)"
            (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)

            badge_y = mcy - 18 if idx == 0 else mcy + bh + 24
            badge_x = max(10, min(iw - bw - 10, mcx - bw // 2))

            cv2.rectangle(magnified, (badge_x - 4, badge_y - bh - 4),
                          (badge_x + bw + 4, badge_y + 4), (12, 16, 24), -1)
            cv2.rectangle(magnified, (badge_x - 4, badge_y - bh - 4),
                          (badge_x + bw + 4, badge_y + 4), col, 1)
            cv2.putText(magnified, badge_txt, (badge_x, badge_y - 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 240, 255), 1, cv2.LINE_AA)

    # Inset window placement (bottom-right of RGB frame)
    pos_x = w_f - iw - 20
    pos_y = h_f - ih - 20

    # Inset header bar
    cv2.rectangle(magnified, (0, 0), (iw, 26), (15, 20, 28), -1)
    cv2.putText(magnified, f"ZOOM INSPECTION ({zoom:.1f}x) - CALIPER PROFILE", (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 1, cv2.LINE_AA)
    cv2.rectangle(magnified, (0, 0), (iw - 1, ih - 1), (0, 240, 255), 2)

    # Paste onto main frame
    frame[pos_y:pos_y + ih, pos_x:pos_x + iw] = magnified


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

    win_name = "RGB + Depth Tube Analyzer - Precision Telemetry (SPACE: Pause | Z: Zoom | H: HUD | V: View)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 480)

    print("\n" + "=" * 65)
    print("  RGB+D TUBE INSPECTION & PRECISION TELEMETRY PLAYER")
    print("=" * 65)
    print("  Controls:")
    print("    [SPACE]     - Pause / Resume")
    print("    [S]         - Step single frame while paused")
    print("    [Z]         - Toggle 4x Zoom Caliper Inset on/off")
    print("    [H]         - Toggle HUD Telemetry card on/off")
    print("    [V]         - Toggle View Mode (Split-Screen / Full RGB HD / Depth)")
    print("    [R]         - Toggle Inspection Pad ROI Filter on/off")
    print("    [Q] / [ESC] - Exit")
    print("=" * 65 + "\n")

    frame_idx = 0
    paused = False
    show_zoom = True
    show_hud = True
    view_mode = 0      # 0: Split Screen, 1: Full RGB with PiP Depth, 2: Full Depth
    roi_filter = True  # Focus on foam pad inspection area (excludes robot hand/screws)

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
            raw_rgb = rgb_frame.copy()

            # 1. Run YOLO inference on RGB frame
            t0 = time.time()
            results = model.predict(rgb_frame, imgsz=1280, conf=0.30, verbose=False)
            dt_ms = (time.time() - t0) * 1000
            r = results[0]

            # 2. Extract tube measurements using segmentation mask minAreaRect
            detections = []
            for i in range(len(r.boxes)):
                box = r.boxes[i]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_name = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = xyxy

                rect = None
                poly = None
                if r.masks is not None and len(r.masks.xy) > i:
                    p = r.masks.xy[i].astype(np.float32)
                    if len(p) >= 5:
                        poly = p.astype(np.int32)
                        rect = cv2.minAreaRect(p)
                        rw, rh = rect[1]
                        thickness_px = min(rw, rh)
                        # Median depth sampled strictly from inside the mask pixels
                        mask_img = np.zeros(depth_frame.shape, dtype=np.uint8)
                        cv2.fillPoly(mask_img, [poly], 255)
                        d_vals = depth_frame[mask_img > 0]
                        d_vals = d_vals[d_vals > 0]
                    else:
                        thickness_px = min(x2 - x1, y2 - y1)
                        d_roi = depth_frame[max(0, y1):min(depth_frame.shape[0], y2),
                                            max(0, x1):min(depth_frame.shape[1], x2)]
                        d_vals = d_roi[d_roi > 0]
                else:
                    thickness_px = min(x2 - x1, y2 - y1)
                    d_roi = depth_frame[max(0, y1):min(depth_frame.shape[0], y2),
                                        max(0, x1):min(depth_frame.shape[1], x2)]
                    d_vals = d_roi[d_roi > 0]

                z_mm = float(np.median(d_vals)) if len(d_vals) > 0 else 0.0
                # True physical width via camera EEPROM calibration
                width_mm = (thickness_px * z_mm) / FOCAL_LENGTH_PX if z_mm > 0 else 0.0

                detections.append({
                    "raw_id": i,
                    "role": cls_name.upper(),
                    "conf": conf,
                    "bbox": xyxy,
                    "poly": poly,
                    "rect": rect,
                    "thickness_px": thickness_px,
                    "z_mm": z_mm,
                    "width_mm": width_mm,
                })

            # Filter detections based on ROI if enabled
            if roi_filter:
                active_tubes = [
                    d for d in detections
                    if d["z_mm"] > 300 and 450 < d["bbox"][0] < 1200 and d["bbox"][1] < 650
                ]
            else:
                active_tubes = [d for d in detections if d["z_mm"] > 0]

            # Sort active tubes top-to-bottom and assign distinct badges & colors
            active_tubes.sort(key=lambda d: d["bbox"][1])
            for idx, d in enumerate(active_tubes):
                d["id"] = idx + 1
                d["color"] = TUBE_COLORS[idx % len(TUBE_COLORS)]

            # 3. Annotate RGB frame
            annotated_rgb = rgb_frame.copy()
            for idx, det in enumerate(active_tubes):
                x1, y1, x2, y2 = det["bbox"]
                col = det["color"]
                poly = det["poly"]
                t_id = det["id"]
                w_mm = det["width_mm"]

                # Subtle mask fill + border
                if poly is not None:
                    overlay = annotated_rgb.copy()
                    cv2.fillPoly(overlay, [poly], col)
                    cv2.addWeighted(overlay, 0.22, annotated_rgb, 0.78, 0, annotated_rgb)
                    cv2.polylines(annotated_rgb, [poly], True, col, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), col, 2)

                # Non-overlapping callout badge with leader line
                anchor_x = x1
                anchor_y = (y1 + y2) // 2

                if idx == 0:
                    callout_x = x1 - 140
                    callout_y = y1 - 30
                else:
                    callout_x = x1 - 140
                    callout_y = y2 + 45

                badge_txt = f"#{t_id} {w_mm:.1f}mm"
                (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)

                cv2.line(annotated_rgb, (anchor_x, anchor_y),
                         (callout_x + tw + 10, callout_y - th // 2), col, 1, cv2.LINE_AA)
                cv2.circle(annotated_rgb, (anchor_x, anchor_y), 3, col, -1)

                cv2.rectangle(annotated_rgb, (callout_x - 4, callout_y - th - 5),
                              (callout_x + tw + 8, callout_y + 4), (12, 16, 24), -1)
                cv2.rectangle(annotated_rgb, (callout_x - 4, callout_y - th - 5),
                              (callout_x + tw + 8, callout_y + 4), col, 1)
                cv2.putText(annotated_rgb, badge_txt, (callout_x, callout_y - 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, col, 1, cv2.LINE_AA)

            # 4. Colorize Depth Map (Turbo heatmap)
            depth_clipped = np.clip(depth_frame, 300, 1500)
            depth_8u = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)

            # Annotate Depth Map with anti-collision labels
            for idx, det in enumerate(active_tubes):
                x1, y1, x2, y2 = det["bbox"]
                col = det["color"]
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), col, 2)
                txt = f"#{det['id']} Z={det['z_mm']:.0f}mm"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

                tag_y = y1 - 8 if idx == 0 else y2 + th + 14
                cv2.rectangle(depth_color, (x1 - 2, tag_y - th - 4), (x1 + tw + 4, tag_y + 2), (10, 12, 18), -1)
                cv2.rectangle(depth_color, (x1 - 2, tag_y - th - 4), (x1 + tw + 4, tag_y + 2), col, 1)
                cv2.putText(depth_color, txt, (x1, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

            # Draw HUD card if enabled
            if show_hud:
                draw_hud(annotated_rgb, active_tubes, FOCAL_LENGTH_PX)

            # Draw 4x Magnified Caliper Inset if enabled
            if show_zoom and len(active_tubes) > 0:
                draw_caliper_zoom_inset(annotated_rgb, raw_rgb, active_tubes, zoom=4.0, inset_size=(460, 320))

            # Stream header overlays
            cv2.putText(annotated_rgb, f"RGB Frame {frame_idx} | YOLO: {dt_ms:.1f}ms | ROI: {'PAD' if roi_filter else 'ALL'}",
                        (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(depth_color, "Depth Map (300mm - 1500mm)",
                        (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Composite according to selected view mode
            if view_mode == 0:
                # Split Screen (Side-by-Side)
                small_rgb = cv2.resize(annotated_rgb, (960, 540))
                small_depth = cv2.resize(depth_color, (960, 540))
                display_img = np.hstack([small_rgb, small_depth])
            elif view_mode == 1:
                # Full RGB 1080p with Depth PiP thumbnail in bottom-left
                display_img = annotated_rgb.copy()
                pip_w, pip_h = 384, 216
                pip_depth = cv2.resize(depth_color, (pip_w, pip_h))
                cv2.rectangle(display_img, (20, 1080 - pip_h - 20), (20 + pip_w, 1080 - 20), (255, 255, 255), 2)
                display_img[1080 - pip_h - 20: 1080 - 20, 20: 20 + pip_w] = pip_depth
            else:
                # Full Depth View
                display_img = depth_color

            cv2.imshow(win_name, display_img)

            if frame_idx % 15 == 0:
                t_info = " | ".join([f"#{t['id']}: {t['width_mm']:.2f}mm" for t in active_tubes])
                print(f"Frame {frame_idx:4d} | Tubes: {len(active_tubes)} [{t_info}]", end="\r")

        # Key handling
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord("q") or key == 27:
            print("\nStopped by user.")
            break
        elif key == ord(" "):
            paused = not paused
            print(f"\n{'[PAUSED]' if paused else '[RESUMED]'}")
        elif key == ord("s") and paused:
            paused = False
            continue
        elif key == ord("z"):
            show_zoom = not show_zoom
            print(f"\nZoom Magnifier: {'ENABLED' if show_zoom else 'DISABLED'}")
        elif key == ord("h"):
            show_hud = not show_hud
            print(f"\nHUD Telemetry Card: {'ENABLED' if show_hud else 'DISABLED'}")
        elif key == ord("v"):
            view_mode = (view_mode + 1) % 3
            modes = ["Split-Screen (RGB + Depth)", "Full RGB HD with PiP Depth", "Full Depth Heatmap"]
            print(f"\nView Mode: {modes[view_mode]}")
        elif key == ord("r"):
            roi_filter = not roi_filter
            print(f"\nInspection ROI Filter: {'FOAM PAD ONLY' if roi_filter else 'FULL FRAME'}")

    cap_rgb.release()
    depth_container.close()
    cv2.destroyAllWindows()
    print("\nCompleted.")


if __name__ == "__main__":
    main()
