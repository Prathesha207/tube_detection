import { useMemo, useRef, useEffect } from 'react';
import type { AnomalyStatus, TubeEntity, LogEntry } from '../types';

export function useAnomalyStatus({
  hasActiveStream,
  isRunning,
  isStarting,
  isCameraSource,
  framesProcessed,
  backendStats,
  addLog,
  tubes,
}: {
  hasActiveStream: boolean;
  isRunning: boolean;
  isStarting: boolean;
  isCameraSource: boolean;
  framesProcessed: number;
  backendStats: any;
  addLog: (message: string, level?: LogEntry['level']) => void;
  tubes: TubeEntity[];
}) {
  const activeTubes = useMemo(() => {
    if (!hasActiveStream && tubes.length === 0) return [];
    return tubes;
  }, [hasActiveStream, tubes]);

  const backendStatus = backendStats.status;

  const anomalyStatus: AnomalyStatus = useMemo(() => {
    if (!hasActiveStream && tubes.length === 0) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: isCameraSource ? 'NO CAMERA' : 'STANDBY',
        subMessage: isCameraSource
          ? 'No Luxonis OAK-D / USB camera connected'
          : 'Waiting for video stream...',
        headsCount: 0,
        tailsCount: 0,
      };
    }

    const hasInferenceResult = framesProcessed > 0 || tubes.length > 0;

    // When inference has NOT run yet: Standby / Ready / Live Stream
    if (!isRunning && !hasInferenceResult) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: isCameraSource
          ? (hasActiveStream ? 'LIVE STREAM' : 'READY')
          : (hasActiveStream ? 'VIDEO READY' : 'STANDBY'),
        subMessage: isCameraSource
          ? (hasActiveStream ? 'Live camera feed active • Click Start Inference' : 'Camera ready • Click Start Stream or Start Inference')
          : (hasActiveStream ? 'Video loaded • Click Start Inference to begin analysis' : 'Waiting for video stream...'),
        headsCount: 0,
        tailsCount: 0,
      };
    }

    if (isRunning && (isStarting || backendStatus === 'WARMING' || framesProcessed === 0)) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: 'WARMING',
        subMessage: 'Warming up AI engine and acquiring tube targets...',
        headsCount: 0,
        tailsCount: 0,
      };
    }

    // Dynamic counts straight from ML output
    const headsCount = Number(backendStats.heads_count ?? tubes.filter(d => d.role === 'HEAD').length);
    const tailsCount = Number(backendStats.tails_count ?? tubes.filter(d => d.role === 'TAIL').length);
    const totalEnds = headsCount + tailsCount;

    if (totalEnds === 0) {
      return {
        isAnomaly: false,
        type: 'NONE' as const,
        message: isRunning ? 'PROCESSING' : 'STANDBY',
        subMessage: 'Scanning frame for tube ends...',
        headsCount: 0,
        tailsCount: 0,
      };
    }

    // Accurate status when inference is active vs stopped
    const isCompleted = backendStatus === 'completed' || Number(backendStats?.progress) >= 100 || (Number(backendStats?.total_frames) > 0 && Number(backendStats?.frames_processed) >= Number(backendStats?.total_frames));
    let stoppedStatus = 'STOPPED';
    if (isCompleted) {
      stoppedStatus = 'COMPLETED';
    } else if (backendStatus === 'paused') {
      stoppedStatus = 'PAUSED';
    }

    return {
      isAnomaly: false,
      type: 'NONE' as const,
      message: isRunning ? 'PROCESSING' : stoppedStatus,
      subMessage: `${headsCount} Head${headsCount === 1 ? '' : 's'} • ${tailsCount} Tail${tailsCount === 1 ? '' : 's'} detected`,
      headsCount,
      tailsCount,
    };
  }, [
    hasActiveStream, isRunning, isStarting, isCameraSource,
    backendStatus, framesProcessed, backendStats, tubes
  ]);

  const prevCountRef = useRef({ heads: 0, tails: 0 });

  useEffect(() => {
    if (!hasActiveStream || !isRunning) {
      prevCountRef.current = { heads: 0, tails: 0 };
      return;
    }
    const { heads, tails } = prevCountRef.current;
    if (heads !== anomalyStatus.headsCount || tails !== anomalyStatus.tailsCount) {
      if (anomalyStatus.headsCount > 0 || anomalyStatus.tailsCount > 0) {
        addLog(`Detected: ${anomalyStatus.headsCount} Heads & ${anomalyStatus.tailsCount} Tails`, 'success');
      }
      prevCountRef.current = { heads: anomalyStatus.headsCount, tails: anomalyStatus.tailsCount };
    }
  }, [anomalyStatus.headsCount, anomalyStatus.tailsCount, addLog, hasActiveStream, isRunning]);

  return {
    activeTubes,
    anomalyStatus,
  };
}
