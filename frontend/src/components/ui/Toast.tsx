import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, Info, X } from 'lucide-react';

export type ToastType = 'error' | 'success' | 'info';

export interface ToastProps {
  type: ToastType;
  message: string;
  onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ type, message, onClose }) => {
  const typeStyles: Record<ToastType, string> = {
    error: 'bg-[var(--status-anomaly-bg)] text-[var(--status-anomaly-text)] border-[var(--status-anomaly-border)]',
    success: 'bg-[var(--status-normal-bg)] text-[var(--status-normal-text)] border-[var(--status-normal-border)]',
    info: 'bg-[var(--bg-card)] text-[var(--text-primary)] border-[var(--border-color)]',
  };

  const icons: Record<ToastType, React.ReactNode> = {
    error: <AlertCircle className="w-4 h-4 shrink-0 text-[var(--status-anomaly-text)]" />,
    success: <CheckCircle2 className="w-4 h-4 shrink-0 text-[var(--status-normal-text)]" />,
    info: <Info className="w-4 h-4 shrink-0 text-[var(--accent-pond)]" />,
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 animate-in fade-in slide-in-from-bottom-3 duration-200 pointer-events-auto">
      <div
        className={`flex items-center gap-2.5 px-4 py-3 rounded-2xl shadow-xl border backdrop-blur-md ${typeStyles[type]}`}
      >
        {icons[type]}
        <span className="text-xs font-bold tracking-wide">{message}</span>
        <button
          onClick={onClose}
          aria-label="Dismiss notification"
          className="ml-2 hover:opacity-80 transition-opacity p-0.5 rounded cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
