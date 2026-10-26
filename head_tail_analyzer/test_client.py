"""
test_client.py - Example Client for Head/Tail ML Microservice
=============================================================
Demonstrates how to send frames (e.g. from OpenCV or image files)
to the running ML microservice and parse the results.
"""

from pathlib import Path
import time
import requests

# Port 8000 by default (or 8001 if port 8000 is occupied by another service)
API_URL = "http://127.0.0.1:8001/infer"
SAMPLE_IMAGE = Path(__file__).parent / "sample_frame.jpg"

def main():
    if not SAMPLE_IMAGE.exists():
        print(f"Error: {SAMPLE_IMAGE} does not exist.")
        return

    print(f"Sending frames from '{SAMPLE_IMAGE.name}' to {API_URL}...")

    # Read the image file bytes once
    with open(SAMPLE_IMAGE, "rb") as f:
        image_bytes = f.read()

    # The tracker requires min_frames (default: 3) consecutive frames to confirm detections
    for frame_idx in range(1, 5):
        t0 = time.perf_counter()
        
        # Send as multipart form-data (standard file upload)
        files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
        params = {"frame_index": frame_idx}
        
        resp = requests.post(API_URL, files=files, params=params)
        roundtrip_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            data = resp.json()
            model_latency = resp.headers.get("x-latency-ms", "N/A")
            detections = data.get("detections", [])
            print(f"\n[Frame {frame_idx}] 200 OK | Model Latency: {model_latency} ms | Roundtrip: {roundtrip_ms:.1f} ms")
            print(f"   Heads: {data.get('heads_count')} | Tails: {data.get('tails_count')} | Confirmed Detections: {len(detections)}")
            print(f"   Bigger Tube: {data.get('bigger_tube')}")
            for d in detections:
                print(f"      - {d.get('role')} #{d.get('id')}: conf={d.get('conf')}, width={d.get('width_px')}px, bbox={d.get('bbox')}")
        else:
            print(f"\n[Frame {frame_idx}] Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    main()
