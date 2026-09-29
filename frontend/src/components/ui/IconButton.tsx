import React from 'react';

export type IconButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type IconButtonSize = 'sm' | 'md' | 'lg';

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: IconButtonVariant;
  size?: IconButtonSize;
  icon: React.ReactNode;
  'aria-label': string;
  active?: boolean;
}

export const IconButton: React.FC<IconButtonProps> = ({
  variant = 'secondary',
  size = 'md',
  icon,
  'aria-label': ariaLabel,
  active = false,
  className = '',
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center rounded-xl select-none cursor-pointer focus:outline-hidden disabled:opacity-50 disabled:cursor-not-allowed active:scale-95 shrink-0 shadow-2xs';

  const sizeStyles: Record<IconButtonSize, string> = {
    sm: 'w-7 h-7 text-xs [&>svg]:w-3.5 [&>svg]:h-3.5',
    md: 'w-7.5 sm:w-8 h-7.5 sm:h-8 text-xs [&>svg]:w-3.5 sm:[&>svg]:w-4 [&>svg]:h-3.5 sm:[&>svg]:h-4',
    lg: 'w-9 h-9 text-sm [&>svg]:w-4.5 [&>svg]:h-4.5',
  };

  const variantStyles: Record<IconButtonVariant, string> = {
    primary: 'bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] border border-[var(--border-color)] shadow-xs',
    secondary: 'bg-[var(--btn-secondary-bg)] border border-[var(--btn-secondary-border)] text-[var(--btn-secondary-text)] hover:bg-[var(--btn-secondary-hover)]',
    ghost: 'bg-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]',
    danger: 'bg-[var(--status-anomaly-bg)] border border-[var(--status-anomaly-border)] text-[var(--status-anomaly-text)] hover:opacity-90',
  };

  const activeStyles = active ? 'ring-2 ring-[var(--accent-pond)]/40 !bg-[var(--accent-pond-subtle)] !border-[var(--accent-pond)] !text-[var(--text-primary)]' : '';

  return (
    <button
      aria-label={ariaLabel}
      title={ariaLabel}
      className={`${baseStyles} ${sizeStyles[size]} ${variantStyles[variant]} ${activeStyles} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon}
    </button>
  );
};
