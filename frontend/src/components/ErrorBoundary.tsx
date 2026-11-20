import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an unhandled render error:', error, errorInfo);
    this.setState({ errorInfo });
    try {
      if (typeof window !== 'undefined' && (window as any).electron?.writeLog) {
        (window as any).electron.writeLog(`[RENDER_CRASH] ${error.name}: ${error.message}\n${error.stack || ''}`);
      }
    } catch { }
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleReset = () => {
    try {
      sessionStorage.clear();
      localStorage.removeItem('vision_monitor_session_initialized');
      localStorage.removeItem('vision_monitor_initialized');
    } catch { }
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="w-screen h-screen bg-[#0B1020] text-slate-100 flex items-center justify-center p-6 select-none">
          <div className="max-w-lg w-full bg-[#131B31] border border-red-500/30 rounded-2xl p-6 sm:p-8 shadow-2xl flex flex-col items-center text-center animate-in fade-in zoom-in-95 duration-200">
            <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 flex items-center justify-center mb-5 shadow-inner">
              <AlertTriangle className="w-8 h-8" />
            </div>

            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white mb-2">
              Workspace Render Exception
            </h2>

            <p className="text-sm text-slate-300 mb-4 max-w-sm">
              An unexpected component error occurred. The application stopped rendering to prevent inconsistent system state.
            </p>

            <div className="w-full bg-[#0A0E1A] border border-slate-800 rounded-xl p-3 text-left mb-6 max-h-40 overflow-y-auto">
              <p className="text-xs font-mono text-red-300 break-words font-semibold">
                {this.state.error?.name || 'Error'}: {this.state.error?.message || 'Unknown render exception'}
              </p>
              {this.state.error?.stack && (
                <pre className="text-[10px] font-mono text-slate-500 mt-2 whitespace-pre-wrap leading-relaxed">
                  {this.state.error.stack.split('\n').slice(0, 5).join('\n')}
                </pre>
              )}
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 w-full">
              <button
                onClick={this.handleReload}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm transition-all duration-150 flex items-center justify-center gap-2 cursor-pointer shadow-md active:scale-95"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Reload Page</span>
              </button>
              <button
                onClick={this.handleReset}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm border border-slate-700 transition-all duration-150 flex items-center justify-center gap-2 cursor-pointer active:scale-95"
              >
                <Home className="w-4 h-4" />
                <span>Reset to Landing</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
