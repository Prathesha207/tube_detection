import argparse
import sys

from app.core.database import Base, SessionLocal, engine
from app.models.camera_model import Camera


def discover_devices():
    """Return devices discovered by DepthAI on this computer."""
    try:
        import depthai as dai
    except Exception as error:
        print(f"DepthAI is not available: {error}")
        return []

    try:
        return dai.Device.getAllAvailableDevices()
    except Exception as error:
        print(f"Could not scan for OAK cameras: {error}")
        return []


def check_camera_connections(devices):
    """Open each discovered device and test features/pipeline."""
    import depthai as dai

    connected = []
    for index, device_info in enumerate(devices, start=1):
        device_id = str(getattr(device_info, "deviceId", "") or getattr(device_info, "mxid", "") or device_info)
        platform_name = str(getattr(device_info, "platform", "Unknown"))
        protocol_name = str(getattr(device_info, "protocol", "Unknown"))
        device_name = str(getattr(device_info, "name", "OAK"))

        print(f"\n[{index}] Device ID: {device_id} | Name: {device_name} | Platform: {platform_name} | Protocol: {protocol_name}")

        try:
            device = dai.Device(device_info)
            actual_info = device.getDeviceInfo()
            actual_id = str(getattr(actual_info, "deviceId", "") or getattr(actual_info, "mxid", "") or device_id)
            print(f"    ✓ Hardware Connection: SUCCESS (MXID: {actual_id})")

            # Check connected camera features
            features_ok = False
            try:
                features = device.getConnectedCameraFeatures()
                feature_desc = [f"{f.socket.name} ({f.sensorName})" for f in features]
                print(f"    ✓ Sensors available: {', '.join(feature_desc) if feature_desc else 'None found'}")
                features_ok = True
            except Exception as feat_err:
                err_text = str(feat_err)
                if "OAK4 has not been setup yet" in err_text or "setup.luxonis.com" in err_text:
                    print(f"    ⚠️  OAK4 Setup Required: The OAK4 device requires initial configuration!")
                    print(f"       Follow instructions at: https://setup.luxonis.com/")
                    print(f"       Or run in terminal: oakctl device setup apply -p <password> --dont-check-internet")
                else:
                    print(f"    ⚠️  Sensors query notice: {feat_err}")

            # Test pipeline construction
            pipeline_ok = False
            try:
                p = dai.Pipeline(device)
                cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                pipeline_ok = True
                print(f"    ✓ Pipeline build with dai.node.Camera: SUCCESS")
            except Exception as pipe_err:
                pipe_err_str = str(pipe_err)
                if "OAK4 has not been setup yet" in pipe_err_str:
                    print(f"    ⚠️  Pipeline: Requires OAK4 setup completion before stream can start")
                else:
                    print(f"    ↳ dai.node.Camera failed ({pipe_err_str}). Testing ColorCamera fallback...")
                    try:
                        p2 = dai.Pipeline(device)
                        cam2 = p2.create(dai.node.ColorCamera)
                        cam2.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                        pipeline_ok = True
                        print(f"    ✓ Pipeline build with dai.node.ColorCamera: SUCCESS")
                    except Exception as p2_err:
                        print(f"    ✗ ColorCamera fallback also failed: {p2_err}")

            connected.append((actual_id, device_info, pipeline_ok))
            device.close()

        except Exception as error:
            err_msg = str(error)
            if "already used" in err_msg.lower():
                print(f"    ✗ Connection failed: Device is in use by another process or application")
            else:
                print(f"    ✗ Connection failed: {error}")

    return connected


def seed_connected_cameras(db, connected_devices):
    """Update database to prefer actual connected hardware cameras."""
    # Deactivate mock 127.0.0.1 camera if present
    mock_cams = db.query(Camera).filter(Camera.ip_address.in_(["127.0.0.1", "localhost", "usb", "auto"])).all()
    for mc in mock_cams:
        mc.is_enabled = False

    existing = {camera.ip_address: camera for camera in db.query(Camera).all()}

    for index, (device_id, _, pipeline_passed) in enumerate(connected_devices, start=1):
        if device_id in existing:
            cam = existing[device_id]
            cam.is_enabled = bool(pipeline_passed)
            status_text = "ENABLED" if pipeline_passed else "DISABLED (requires setup)"
            print(f"  - Updated Camera ID {cam.id} ({cam.name} | {cam.ip_address}) -> {status_text}")
        else:
            camera = Camera(
                name=f"OAK Camera {index}",
                ip_address=device_id,
                resolution="1920x1080",
                fps=30,
                is_enabled=bool(pipeline_passed),
            )
            db.add(camera)
            status_text = "ENABLED" if pipeline_passed else "DISABLED (requires setup)"
            print(f"  - Created Camera: {camera.name} | {camera.ip_address} -> {status_text}")

    db.commit()


def show_database_cameras(db):
    """Print all cameras currently configured in the database."""
    cameras = db.query(Camera).all()
    print("\n---------------- Current Database Cameras ----------------")
    if not cameras:
        print("  (No cameras in database)")
    for c in cameras:
        status = "ACTIVE [ENABLED]" if c.is_enabled else "DISABLED"
        print(f"  ID {c.id}: {c.name} | Address: {c.ip_address} | {status} | Res: {c.resolution} | FPS: {c.fps}")
    print("----------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Check OAK cameras and configure active devices.")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Insert or enable successfully connected OAK devices in the database.",
    )
    args = parser.parse_args()

    print("==================================================")
    print("       OAK Camera Diagnostic & Configuration     ")
    print("==================================================")

    print("\nScanning Linux USB/network devices with DepthAI...")
    devices = discover_devices()
    print(f"Found {len(devices)} DepthAI device entries.")

    if not devices:
        print("No OAK device was discovered. Check USB, power, permissions, and DepthAI.")
        return

    connected_devices = check_camera_connections(devices)
    print(f"\n==================================================")
    print(f"Successfully opened: {len(connected_devices)} of {len(devices)} devices")
    print("==================================================")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if args.seed and connected_devices:
            seed_connected_cameras(db, connected_devices)
            print("\nDatabase updated with connected hardware cameras.")
        elif not args.seed:
            print("\nTip: Run 'python3 seed_cameras.py --seed' to activate connected cameras in the DB.")

        show_database_cameras(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
