import { useState, useMemo, useRef, useEffect } from 'react';
import type { AnomalyStatus, DuckEntity, LogEntry, DetectionMetrics } from '../types';
import { playAnomalyAlertSound, playNormalSound } from '../utils/audio';

export function useAnomalyStatus({
  hasActiveStream,
  isRunning,
  isStarting,
  isCameraSource,
  framesProcessed,
  backendStats,
  addLog,
  ducks,
  expectedDucks,
}: {
  hasActiveStream: boolean;
  isRunning: boolean;
  isStarting: boolean;
  isCameraSource: boolean;
  framesProcessed: number;
  backendStats: any;
  addLog: (message: string, level?: LogEntry['level']) => void;
  ducks: DuckEntity[];
  expectedDucks: number;
}) {
  const activeDucks = useMemo(() => {
    if (!hasActiveStream && ducks.length === 0) return [];
    return ducks;
  }, [hasActiveStream, ducks]);

  const lastValidDetectedCountRef = useRef<number>(expectedDucks);

  const backendStatus = backendStats.status;

  const anomalyStatus: AnomalyStatus = useMemo(() => {
    if (!hasActiveStream && ducks.length === 0) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: isCameraSource ? 'NO CAMERA' : 'STANDBY',
        subMessage: isCameraSource
          ? 'No Luxonis OAK-D / USB camera connected'
          : 'Waiting for video stream...',
        detectedCount: 0,
        expectedCount: expectedDucks,
        difference: 0,
        foreignSpecies: [],
        foreignCount: 0,
      };
    }

    const hasInferenceResult = framesProcessed > 0 || ducks.length > 0;

    // When inference has NOT run yet: Standby / Ready / Live Stream (clean slate, zero mismatch)
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
        detectedCount: 0,
        expectedCount: expectedDucks,
        difference: 0,
        foreignSpecies: [],
        foreignCount: 0,
      };
    }

    if (isRunning && (isStarting || backendStatus === 'WARMING' || framesProcessed === 0)) {
      return {
        isAnomaly: false,
        type: 'NONE',
        message: 'WARMING',
        subMessage: 'Warming up AI engine and acquiring targets...',
        detectedCount: backendStats.detected_duck_count,
        expectedCount: expectedDucks,
        difference: 0,
        foreignSpecies: [],
        foreignCount: 0,
      };
    }

    const backendReasons: string[] = Array.isArray(backendStats.reasons) ? backendStats.reasons : [];
    const isWarmingUp = backendStatus === 'WARMING' || backendStats.anchor_locked === false;
    const hasHand = backendStatus === 'HAND' || backendStats.hand_detected === true;

    const backendDetected = Number(backendStats.detected_duck_count);
    const backendExpected = Number(backendStats.expected_duck_count);
    const expectedFromMl = Number.isFinite(backendExpected) && backendExpected > 0
      ? backendExpected
      : expectedDucks;

    let detectedCount: number;
    if (hasHand) {
      // When a hand enters the tray, the ML engine temporarily pauses detection to prevent occlusions.
      // Retain the last known valid count (or expected count) instead of collapsing to 0.
      detectedCount = lastValidDetectedCountRef.current || expectedFromMl;
    } else {
      detectedCount = Number.isFinite(backendDetected) && backendDetected > 0
        ? backendDetected
        : ducks.filter((duck) => duck.species === 'Duck' && duck.statusEvent !== 'missing').length;
      if (detectedCount > 0) {
        lastValidDetectedCountRef.current = detectedCount;
      }
    }

    const backendForeign = Number(backendStats.detected_other_toy_count);
    const foreignCount = Number.isFinite(backendForeign)
      ? backendForeign
      : ducks.filter((duck) => duck.species === 'Unknown' && !duck.provisional).length;
    const missingIds = Array.isArray(backendStats.missing_ids) ? backendStats.missing_ids : [];
    const hasMissingDuck = !isWarmingUp && !hasHand && (
      backendReasons.includes('missing_duck') ||
      missingIds.length > 0 ||
      ducks.some((duck) => !duck.provisional && duck.statusEvent === 'missing')
    );
    const difference = hasHand ? 0 : (detectedCount - expectedFromMl);
    const foreignSpecies = foreignCount > 0 ? ['Unknown'] : [];

    // BUG FIX: Trust backend-smoothed verdicts from analyzer.py (reasons / status)
    // rather than re-computing raw count mismatches per single frame. A momentary
    // single-frame occlusion or detection glitch must not trigger an instant alarm
    // before the backend's smoothing window confirms it.
    const isTooFew = !isWarmingUp && !hasHand && (backendReasons.includes('too_few_ducks') || backendReasons.includes('too_few'));
    const isTooMany = !isWarmingUp && !hasHand && (backendReasons.includes('too_many_ducks') || backendReasons.includes('too_many'));
    const isCountMismatch = !isWarmingUp && !hasHand && (isTooFew || isTooMany);
    const hasForeign = !isWarmingUp && !hasHand && (backendReasons.includes('other_species_present') || foreignCount > 0);

    // Backend-aligned anomaly verdict: single source of truth from analyzer.py
    // NOTE: Hand present is NOT an anomaly (it is a temporary inspection pause).
    const isAnomaly = !isWarmingUp && !hasHand && (
      backendStatus === 'ANOMALY' ||
      isCountMismatch ||
      hasMissingDuck ||
      hasForeign
    );
    let message = hasHand ? 'HAND DETECTED' : isAnomaly ? 'ANOMALY' : isWarmingUp ? 'WARMING' : 'NORMAL';
    let subMessage = hasHand
      ? 'Hand detected in frame. Evaluation paused until hand is removed.'
      : isWarmingUp
        ? 'Warming up AI engine and acquiring targets...'
        : `${detectedCount} ducks detected in target area. Count matches expected (${expectedFromMl}).`;
    let type: AnomalyStatus['type'] = 'NONE';

    if (hasHand) {
      type = 'NONE';
      message = 'HAND DETECTED';
      subMessage = 'Hand detected in frame. Evaluation paused until hand is removed.';
    } else if (isAnomaly) {
      if (hasForeign && isCountMismatch) {
        type = 'FOREIGN_SPECIES';
        subMessage = `${foreignCount} non-duck detected & duck count: ${detectedCount}/${expectedFromMl} (${difference > 0 ? `+${difference}` : difference})`;
      } else if (hasForeign) {
        type = 'FOREIGN_SPECIES';
        subMessage = `Duck count normal (${detectedCount}/${expectedFromMl}), but ${foreignCount} non-duck detected.`;
      } else if (isCountMismatch) {
        if (isTooMany || difference > 0) {
          type = 'OVER_COUNT';
          const excessList: any[] = (Array.isArray(backendStats.excess_ids) && backendStats.excess_ids.length > 0)
            ? backendStats.excess_ids
            : (backendStats.added_ids || []);
          const excessStr = excessList.length > 0 ? ` (Excess: ${excessList.join(', ')})` : '';
          subMessage = `+${Math.max(1, difference)} above expected count (${detectedCount} detected, ${expectedFromMl} expected)${excessStr}`;
        } else {
          type = 'UNDER_COUNT';
          const missing = missingIds.length > 0 ? ` (Missing: ${missingIds.join(', ')})` : '';
          subMessage = `${Math.abs(difference) || 1} missing ducks (${detectedCount} detected, ${expectedFromMl} expected)${missing}`;
        }
      } else if (hasMissingDuck) {
        type = 'MISSING_DUCK';
        const missing = missingIds.length > 0 ? `: ${missingIds.join(', ')}` : '';
        subMessage = `A tracked duck is missing from this frame${missing}.`;
      } else {
        type = 'FOREIGN_SPECIES';
        subMessage = `${foreignCount} non-duck detected.`;
      }
    }

    return {
      isAnomaly,
      type,
      message,
      subMessage,
      detectedCount,
      expectedCount: expectedFromMl,
      difference,
      foreignSpecies: hasHand ? [] : foreignSpecies,
      foreignCount: hasHand ? 0 : foreignCount,
    };
  }, [
    hasActiveStream, isRunning, isStarting, expectedDucks, isCameraSource,
    backendStatus, framesProcessed, backendStats, ducks.length
  ]);

  const prevAnomalyRef = useRef(anomalyStatus.isAnomaly);
  const isAnomaly = anomalyStatus.isAnomaly;
  const subMessage = anomalyStatus.subMessage;

  useEffect(() => {
    if (!hasActiveStream || !isRunning) {
      prevAnomalyRef.current = false;
      return;
    }
    if (prevAnomalyRef.current !== isAnomaly) {
      if (isAnomaly) {
        playAnomalyAlertSound();
        addLog(`Anomalous Activity: ${subMessage}`, 'anomaly');
      } else {
        playNormalSound();
        addLog(`Status Normalized: Expected (${anomalyStatus.expectedCount}) matches Detected (${anomalyStatus.detectedCount})`, 'success');
      }
      prevAnomalyRef.current = isAnomaly;
    }
  }, [isAnomaly, subMessage, anomalyStatus.expectedCount, anomalyStatus.detectedCount, addLog, hasActiveStream, isRunning]);

  return {
    activeDucks,
    anomalyStatus,
  };
}
