import React from 'react';
import { Plus, Minus } from 'lucide-react';

export interface NumberStepperProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  size?: 'sm' | 'md';
  className?: string;
}

export const NumberStepper: React.FC<NumberStepperProps> = ({
  value,
  onChange,
  min = 0,
  max = 999,
  step = 1,
  size = 'md',
  className = '',
}) => {
  const handleIncrement = () => {
    if (value + step <= max) {
      onChange(value + step);
    }
  };

  const handleDecrement = () => {
    if (value - step >= min) {
      onChange(value - step);
    }
  };

  const sizeStyles = {
    sm: 'h-6 sm:h-7 p-0.5 rounded-lg text-xs',
    md: 'h-7 sm:h-8 p-0.5 rounded-xl text-xs sm:text-sm',
  };

  const btnSizeStyles = {
    sm: 'w-5 h-5 sm:w-6 sm:h-6 rounded-md',
    md: 'w-6 h-6 sm:w-7 sm:h-7 rounded-lg',
  };

  return (
    <div
      className={`inline-flex items-center bg-[var(--btn-secondary-bg)] border border-[var(--btn-secondary-border)] shadow-2xs shrink-0 select-none ${sizeStyles[size]} ${className}`}
    >
      <button
        type="button"
        onClick={handleDecrement}
        aria-label="Decrease value"
        disabled={value <= min}
        className={`flex items-center justify-center text-[var(--accent-pond)] hover:bg-[var(--btn-secondary-hover)] active:scale-95 font-bold cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed ${btnSizeStyles[size]}`}
      >
        <Minus className="w-3.5 h-3.5" />
      </button>

      <input
        type="number"
        value={value}
        onChange={(e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val) && val >= min && val <= max) {
            onChange(val);
          }
        }}
        className="w-7 sm:w-9 text-center font-black text-[var(--text-primary)] bg-transparent focus:outline-hidden p-0 m-0 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
      />

      <button
        type="button"
        onClick={handleIncrement}
        aria-label="Increase value"
        disabled={value >= max}
        className={`flex items-center justify-center text-[var(--accent-pond)] hover:bg-[var(--btn-secondary-hover)] active:scale-95 font-bold cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed ${btnSizeStyles[size]}`}
      >
        <Plus className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};
