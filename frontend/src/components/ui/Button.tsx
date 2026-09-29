import React from 'react';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'outline' | 'danger' | 'success';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: React.ReactNode;
  iconRight?: React.ReactNode;
  loading?: boolean;
  active?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'secondary',
  size = 'md',
  icon,
  iconRight,
  loading = false,
  active = false,
  children,
  className = '',
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-bold tracking-tight rounded-xl select-none cursor-pointer focus:outline-hidden disabled:opacity-50 disabled:cursor-not-allowed active:scale-98 shrink-0';

  const sizeStyles: Record<ButtonSize, string> = {
    sm: 'h-7 px-2.5 text-[11px] gap-1.5',
    md: 'h-8 px-3 text-xs gap-1.5',
    lg: 'h-9 px-4 text-xs sm:text-sm gap-2',
  };

  const variantStyles: Record<ButtonVariant, string> = {
    primary: 'bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] border border-[var(--border-color)] shadow-xs',
    secondary: 'bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--btn-secondary-border)] text-[var(--btn-secondary-text)] shadow-2xs',
    ghost: 'bg-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]',
    outline: 'bg-transparent border border-[var(--border-color)] text-[var(--text-primary)] hover:bg-[var(--btn-secondary-bg)]',
    danger: 'bg-[var(--status-anomaly-bg)] hover:opacity-90 border border-[var(--status-anomaly-border)] text-[var(--status-anomaly-text)] shadow-xs',
    success: 'bg-[var(--status-normal-bg)] hover:opacity-90 border border-[var(--status-normal-border)] text-[var(--status-normal-text)] shadow-xs',
  };

  const activeStyles = active ? 'ring-2 ring-[var(--accent-pond)]/40 !bg-[var(--accent-pond-subtle)] !border-[var(--accent-pond)] !text-[var(--text-primary)]' : '';

  return (
    <button
      className={`${baseStyles} ${sizeStyles[size]} ${variantStyles[variant]} ${activeStyles} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin shrink-0" />
      ) : icon ? (
        <span className="shrink-0 [&>svg]:w-3.5 [&>svg]:h-3.5">{icon}</span>
      ) : null}
      {children && <span className="whitespace-nowrap truncate">{children}</span>}
      {iconRight && <span className="shrink-0 [&>svg]:w-3.5 [&>svg]:h-3.5">{iconRight}</span>}
    </button>
  );
};
