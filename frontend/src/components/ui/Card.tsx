import React from 'react';

export type CardVariant = 'default' | 'elevated' | 'subtle' | 'anomaly' | 'normal' | 'glass';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  header?: React.ReactNode;
  footer?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  variant = 'default',
  padding = 'md',
  header,
  footer,
  children,
  className = '',
  ...props
}) => {
  const variantStyles: Record<CardVariant, string> = {
    default: 'bg-[var(--bg-card)] border border-[var(--border-color)] shadow-xs text-[var(--text-primary)]',
    elevated: 'bg-[var(--bg-card)] border border-[var(--border-color)] shadow-md text-[var(--text-primary)]',
    subtle: 'bg-[var(--bg-card-subtle)] border border-[var(--border-color)] text-[var(--text-primary)]',
    anomaly: 'bg-[var(--status-anomaly-bg)] border border-[var(--status-anomaly-border)] text-[var(--status-anomaly-text)] shadow-sm',
    normal: 'bg-[var(--status-normal-bg)] border border-[var(--status-normal-border)] text-[var(--status-normal-text)] shadow-sm',
    glass: 'bg-[var(--glass-bg)] border border-[var(--border-color)] text-[var(--text-primary)] shadow-sm',
  };

  const paddingStyles: Record<string, string> = {
    none: 'p-0',
    sm: 'p-2 sm:p-2.5',
    md: 'p-3 sm:p-3.5',
    lg: 'p-4 sm:p-5',
  };

  return (
    <div
      className={`rounded-2xl ${variantStyles[variant]} ${paddingStyles[padding]} ${className}`}
      {...props}
    >
      {header && (
        <div className="pb-2.5 mb-2.5 border-b border-[var(--border-color)]">
          {header}
        </div>
      )}
      {children}
      {footer && (
        <div className="pt-2.5 mt-2.5 border-t border-[var(--border-color)]">
          {footer}
        </div>
      )}
    </div>
  );
};
