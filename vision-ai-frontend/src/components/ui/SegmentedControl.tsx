import React from 'react';

export interface SegmentOption<T extends string> {
  value: T;
  label: React.ReactNode;
  icon?: React.ReactNode;
}

export interface SegmentedControlProps<T extends string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  size?: 'sm' | 'md';
  className?: string;
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
  className = '',
}: SegmentedControlProps<T>) {
  const sizeStyles = {
    sm: 'h-6 sm:h-7 p-0.5 rounded-lg text-[10px] sm:text-xs',
    md: 'h-7 sm:h-8 p-0.5 rounded-xl text-xs',
  };

  const itemSizeStyles = {
    sm: 'h-5 sm:h-6 px-2 sm:px-2.5 rounded-md gap-1',
    md: 'h-6 sm:h-7 px-2.5 sm:px-3 rounded-lg gap-1.5',
  };

  return (
    <div
      className={`inline-flex items-center bg-[var(--btn-secondary-bg)] border border-[var(--btn-secondary-border)] shadow-2xs shrink-0 select-none ${sizeStyles[size]} ${className}`}
    >
      {options.map((opt) => {
        const isSelected = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => {
              onChange(opt.value);
            }}
            className={`flex items-center justify-center font-bold tracking-tight cursor-pointer ${itemSizeStyles[size]} ${
              isSelected
                ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] shadow-xs'
                : 'text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)]'
            }`}
          >
            {opt.icon && <span className="shrink-0 [&>svg]:w-3 sm:[&>svg]:w-3.5 [&>svg]:h-3 sm:[&>svg]:h-3.5">{opt.icon}</span>}
            <span className="whitespace-nowrap">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
