import { CameraConfig } from '../types';

export const DEFAULT_CAMERA_CONFIG: CameraConfig = {
  sourceName: 'OAK-D Pro Stream',
  resolution: '1920x1080',
  targetFps: 30,
  recordingFormat: 'MP4',
  rotationAngle: 0,
  controlMode: 'auto',
  exposure: 50,
  gain: 100,
  focus: 50,
  brightness: 0,
  contrast: 50,
  iso: 800,
  autoFocus: true,
  autoExposure: true,
  connected: false,
  connectionQuality: 'Excellent',
  ipAddress: '192.168.1.100',
};
