import cv2


def stream_camera(camera_url):

    cap = cv2.VideoCapture(camera_url)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()