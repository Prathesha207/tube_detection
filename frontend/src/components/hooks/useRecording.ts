import { useState, useRef, useCallback, useEffect } from 'react';
import { useInferenceStore } from '../../store/inferenceStore';
import { recordingService, StopRecordingResponse } from '../service/recordingService';
import { showToast } from '../../lib/toast';

export function useRecording() {
  const [isRecording, setIsRecording] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState<string | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);

  const sessionIdRef = useRef<string | null>(null);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number | null>(null);

  const setIsRecordingStore = useInferenceStore((state) => state.setIsRecording);

  useEffect(() => {
    setIsRecordingStore(isRecording);
  }, [isRecording, setIsRecordingStore]);

  useEffect(() => {
    return () => {
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
    };
  }, []);

  const startRecording = useCallback(async (recordingFormat: string = 'MP4') => {
    if (isRecording || isSaving) return false;

    const newSessionId = `rec_${Date.now()}`;
    sessionIdRef.current = newSessionId;

    try {
      const res = await recordingService.startRecording(newSessionId, recordingFormat);
      if (res && res.status === 'recording') {
        setIsRecording(true);
        setRecordingDuration(0);
        startTimeRef.current = Date.now();
        if (durationIntervalRef.current) clearInterval(durationIntervalRef.current);
        durationIntervalRef.current = setInterval(() => {
          if (startTimeRef.current) {
            setRecordingDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
          }
        }, 1000);
        return true;
      } else {
        showToast('error', 'Failed to start camera recording on backend.');
        return false;
      }
    } catch (err: any) {
      console.error('[RECORD] startRecording error:', err);
      showToast('error', err?.message || 'Failed to start recording');
      return false;
    }
  }, [isRecording, isSaving]);

  const stopRecording = useCallback(async (): Promise<StopRecordingResponse | null> => {
    if (!isRecording && !sessionIdRef.current) return null;

    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }
    setIsRecording(false);
    setIsSaving(true);

    const activeSessionId = sessionIdRef.current || `rec_${Date.now()}`;
    try {
      const res = await recordingService.stopRecording(activeSessionId);
      setIsSaving(false);
      sessionIdRef.current = null;
      if (res && res.status === 'done' && res.session_id) {
        return res;
      } else {
        showToast('warning', res?.message || 'Recording finished, but file was not found.');
        return null;
      }
    } catch (err: any) {
      console.error('[RECORD] stopRecording error:', err);
      setIsSaving(false);
      sessionIdRef.current = null;
      showToast('error', err?.message || 'Failed to stop recording');
      return null;
    }
  }, [isRecording]);

  const clearRecording = useCallback(() => {
    if (recordedVideoUrl) URL.revokeObjectURL(recordedVideoUrl);
    setRecordedVideoUrl(null);
    setRecordedFile(null);
    setRecordingDuration(0);
    sessionIdRef.current = null;
  }, [recordedVideoUrl]);

  return {
    isRecording,
    isSaving,
    recordedVideoUrl,
    recordedFile,
    recordingDuration,
    startRecording,
    stopRecording,
    clearRecording,
  };
}
