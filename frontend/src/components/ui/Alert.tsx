import React from 'react';
import { AlertTriangle, AlertCircle, CheckCircle2, Info } from 'lucide-react';

export type AlertVariant = 'anomaly' | 'warning' | 'success' | 'info';

export interface AlertProps {
  variant?: AlertVariant;
  title?: React.ReactNode;
  children: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}

export const Alert: React.FC<AlertProps> = ({
  variant = 'info',
  title,
  children,
  icon,
  className = '',
  action,
}) => {
  const variantStyles: Record<AlertVariant, string> = {
    anomaly: 'bg-[var(--status-anomaly-bg)] border-[var(--status-anomaly-border)] text-[var(--status-anomaly-text)]',
    warning: 'bg-[var(--status-warn-bg)] border-[var(--status-warn-border)] text-[var(--status-warn-text)]',
    success: 'bg-[var(--status-normal-bg)] border-[var(--status-normal-border)] text-[var(--status-normal-text)]',
    info: 'bg-[var(--bg-card-subtle)] border-[var(--border-color)] text-[var(--text-primary)]',
  };

  const defaultIcons: Record<AlertVariant, React.ReactNode> = {
    anomaly: <AlertTriangle className="w-4 h-4 text-[var(--status-anomaly-text)] shrink-0" />,
    warning: <AlertTriangle className="w-4 h-4 text-[var(--status-warn-text)] shrink-0" />,
    success: <CheckCircle2 className="w-4 h-4 text-[var(--status-normal-text)] shrink-0" />,
    info: <Info className="w-4 h-4 text-[var(--text-secondary)] shrink-0" />,
  };

  return (
    <div
      role="alert"
      className={`flex items-start gap-2.5 p-3 rounded-xl border text-xs leading-relaxed ${variantStyles[variant]} ${className}`}
    >
      <div className="mt-0.5 shrink-0">{icon || defaultIcons[variant]}</div>
      <div className="flex-1 min-w-0">
        {title && <div className="font-bold text-xs mb-0.5 tracking-tight">{title}</div>}
        <div className="opacity-95 font-medium">{children}</div>
      </div>
      {action && <div className="shrink-0 ml-2">{action}</div>}
    </div>
  );
};
