import React from 'react';

export type BadgeVariant = 'normal' | 'warning' | 'anomaly' | 'offline' | 'active' | 'neutral' | 'info';
export type BadgeSize = 'sm' | 'md';

export interface BadgeProps {
  variant?: BadgeVariant;
  size?: BadgeSize;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
  dotColor?: string;
  onClick?: () => void;
  title?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  size = 'md',
  icon,
  children,
  className = '',
  dot = false,
  dotColor,
  onClick,
  title,
}) => {
  const variantStyles: Record<BadgeVariant, string> = {
    normal: 'bg-emerald-600 text-white border-emerald-500 font-bold shadow-xs',
    warning: 'bg-[var(--status-warn-bg)] text-[var(--status-warn-text)] border-[var(--status-warn-border)]',
    anomaly: 'bg-[var(--status-anomaly-bg)] text-[var(--status-anomaly-text)] border-[var(--status-anomaly-border)]',
    offline: 'bg-[var(--bg-card-subtle)] text-[var(--text-muted)] border-[var(--border-color)]',
    active: 'bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] border-[var(--accent-pond)]',
    neutral: 'bg-[var(--btn-secondary-bg)] text-[var(--text-primary)] border-[var(--btn-secondary-border)]',
    info: 'bg-[var(--accent-pond-subtle)] text-[var(--text-primary)] border-[var(--border-color)]',
  };

  const defaultDotColors: Record<BadgeVariant, string> = {
    normal: 'bg-emerald-300',
    warning: 'bg-[var(--status-warn-text)]',
    anomaly: 'bg-[var(--status-anomaly-text)]',
    offline: 'bg-[var(--text-muted)]',
    active: 'bg-[var(--accent-pond)] animate-pulse',
    neutral: 'bg-[var(--accent-pond)]',
    info: 'bg-[var(--accent-pond)]',
  };

  const sizeStyles: Record<BadgeSize, string> = {
    sm: 'h-6 px-2.5 text-[10px] gap-1.5 rounded-xl',
    md: 'h-7 px-3 text-[11px] gap-1.5 rounded-xl',
  };

  const Component = onClick ? 'button' : 'span';

  return (
    <Component
      onClick={onClick}
      title={title}
      className={`inline-flex flex-row items-center whitespace-nowrap border font-semibold tracking-tight transition-all select-none shrink-0 ${sizeStyles[size]} ${
        onClick ? 'cursor-pointer hover:bg-[var(--btn-secondary-hover)] active:scale-98' : ''
      } ${variantStyles[variant]} ${className}`}
    >
      {dot && (
        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotColor || defaultDotColors[variant]}`} />
      )}
      {icon && <span className="inline-flex items-center shrink-0 [&>svg]:w-3.5 [&>svg]:h-3.5">{icon}</span>}
      <span className="inline-flex flex-row items-center gap-1.5 whitespace-nowrap">{children}</span>
    </Component>
  );
};
