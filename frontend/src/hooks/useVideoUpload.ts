import { useState, useEffect, useRef } from 'react';
import { getApiBaseUrl } from '../lib/api';
import { useInferenceStore } from '../store/inferenceStore';
import { showToast } from '../lib/toast';

export const useVideoUpload = (
  fileInputRef: React.RefObject<HTMLInputElement | null>,
  onVideoUploaded?: (videoUrl: string, fileName: string, sessionId?: string, isCameraRecording?: boolean) => void | Promise<void>,
  recordedFile?: File | null,
  clearRecording?: () => void,
  initialUploadFile?: File
) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [isSelectingVideo, setIsSelectingVideo] = useState(false);

  const blobUrlRef = useRef<string | null>(null);

  const processUploadedFile = async (file: File, isRecorded: boolean = false) => {
    setUploadProgress(0);
    useInferenceStore.getState().setVideoLoading(true);

    const isRec = Boolean(isRecorded);
    const formData = new FormData();

    // When running in Electron / Desktop EXE, the File object has a direct local path (.path)
    // Passing this allows the backend to read directly from disk with 0 upload overhead and 0 temp files!
    const localPath = (file as any).path || '';
    if (localPath && typeof localPath === 'string' && localPath.length > 3 && !isRec) {
      formData.append('file_path', localPath);
    }

    formData.append('file', file);
    formData.append('is_camera_recording', isRec ? 'true' : 'false');
    const fpsMatch = file.name.match(/_(\d+)fps/i);
    if (fpsMatch) {
      formData.append('fps', fpsMatch[1]);
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${getApiBaseUrl()}/video/upload`);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        setUploadProgress(Math.min(99, percent));
      }
    };

    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          setUploadProgress(100);
          const baseUrl = getApiBaseUrl();
          if (onVideoUploaded) {
            const streamUrl = `${baseUrl}/video/stream/${data.session_id}`;
            const isCameraRec = Boolean(data.is_camera_recording ?? isRec);
            await onVideoUploaded(streamUrl, file.name, data.session_id, isCameraRec);
          }
        } catch (e) {
          console.error('Error starting inference after upload:', e);
        } finally {
          setUploadProgress(null);
          useInferenceStore.getState().setVideoLoading(false);
        }
      } else {
        setUploadProgress(null);
        useInferenceStore.getState().setVideoLoading(false);
        try {
          const errData = JSON.parse(xhr.responseText || '{}');
          showToast('error', errData.message || `Upload failed (Status: ${xhr.status})`);
        } catch {
          showToast('error', `Upload failed with status: ${xhr.status}`);
        }
        console.error('Upload failed with status:', xhr.status);
      }
    };

    xhr.onerror = () => {
      setUploadProgress(null);
      useInferenceStore.getState().setVideoLoading(false);
      showToast('error', 'Network error during video upload');
      console.error('Network error during video upload');
    };

    xhr.send(formData);
  };

  useEffect(() => {
    if (recordedFile) {
      processUploadedFile(recordedFile, true);
      if (clearRecording) {
        clearRecording(); // Ensure it is consumed
      }
    }
  }, [recordedFile]);

  useEffect(() => {
    if (initialUploadFile) {
      processUploadedFile(initialUploadFile);
    }
  }, [initialUploadFile]);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }
    };
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processUploadedFile(file);
    }
    e.target.value = '';
  };

  const handleSelectVideoAndStart = async () => {
    const electronApi = (window as any).electronAPI;
    if (electronApi && typeof electronApi.selectFile === 'function') {
      try {
        setIsSelectingVideo(true);
        const filePath = await electronApi.selectFile();
        if (!filePath) {
          setIsSelectingVideo(false);
          return;
        }

        useInferenceStore.getState().setVideoLoading(true);
        const baseUrl = getApiBaseUrl();
        const res = await fetch(`${baseUrl}/video/inference/path`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            video_path: filePath,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.message || 'Failed to start path inference');
        }

        const data = await res.json();
        const filename = data.video_name || filePath.split(/[/\\]/).pop() || 'video.mp4';

        if (onVideoUploaded) {
          const streamUrl = `${baseUrl}/video/stream/${data.session_id}`;
          await onVideoUploaded(streamUrl, filename, data.session_id);
        }
      } catch (err: any) {
        console.error('Desktop video selection error:', err);
      } finally {
        setIsSelectingVideo(false);
        useInferenceStore.getState().setVideoLoading(false);
      }
      return;
    }

    fileInputRef.current?.click();
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent, isVideoSource: boolean) => {
    e.preventDefault();
    setIsDragOver(false);
    if (!isVideoSource) return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      processUploadedFile(file);
    }
  };

  return {
    isDragOver,
    uploadProgress,
    isSelectingVideo,
    processUploadedFile,
    handleFileInputChange,
    handleSelectVideoAndStart,
    handleDragOver,
    handleDragLeave,
    handleDrop
  };
};
