export function lockScroll() {
  if (typeof document === 'undefined') return;
  document.body.classList.add('modal-open');
  document.documentElement.classList.add('modal-open');
}

export function unlockScroll() {
  if (typeof document === 'undefined') return;
  document.body.classList.remove('modal-open');
  document.documentElement.classList.remove('modal-open');
}
