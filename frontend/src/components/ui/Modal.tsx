import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { lockScroll, unlockScroll } from '../../utils/scrollLock';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  description,
  icon,
  children,
  footer,
  maxWidth = 'md',
  className = '',
}) => {
  useEffect(() => {
    if (isOpen && typeof document !== 'undefined') {
      lockScroll();
      return () => {
        unlockScroll();
      };
    }
  }, [isOpen]);

  if (!isOpen || typeof document === 'undefined') return null;

  const maxWidthStyles: Record<string, string> = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
  };

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[999999] flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs select-none"
      onClick={handleBackdropClick}
    >
      <div
        className={`relative w-full ${maxWidthStyles[maxWidth]} max-h-[90dvh] rounded-3xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-2xl overflow-hidden flex flex-col ${className}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        {(title || icon) && (
          <div className="flex items-center justify-between p-5 pb-4 border-b border-[var(--border-color)] shrink-0">
            <div className="flex items-center gap-2.5 min-w-0">
              {icon && (
                <div className="w-8 h-8 rounded-xl bg-[var(--accent-pond-subtle)] text-[var(--accent-pond)] flex items-center justify-center border border-[var(--border-color)] shrink-0">
                  {icon}
                </div>
              )}
              <div className="min-w-0">
                {title && (
                  <h2 className="font-bold text-sm sm:text-base text-[var(--text-primary)] truncate">
                    {title}
                  </h2>
                )}
                {description && (
                  <p className="text-[11px] sm:text-xs text-[var(--text-secondary)] truncate">
                    {description}
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={() => {
                onClose();
              }}
              aria-label="Close dialog"
              className="p-1.5 rounded-xl text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--btn-secondary-hover)] transition-colors cursor-pointer shrink-0 ml-2"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto flex-1 min-h-0 text-[var(--text-primary)]">{children}</div>

        {/* Modal Footer */}
        {footer && (
          <div className="p-4 bg-[var(--bg-card-subtle)] border-t border-[var(--border-color)] shrink-0 flex items-center justify-end gap-2">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body
  );
};
