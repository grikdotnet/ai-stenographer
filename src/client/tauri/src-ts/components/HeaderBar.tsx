import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import type { MouseEvent } from "react";
import type { ReactElement } from "react";
import type { HeaderStatus } from "../types/viewState";

interface HeaderBarProps {
  title: string;
  subtitle: string;
  status: HeaderStatus;
}

/**
 * Renders the static application heading and passive shell status.
 */
export function HeaderBar({ title, subtitle, status }: HeaderBarProps): ReactElement {
  async function handleDragRegionMouseDown(event: MouseEvent<HTMLDivElement>): Promise<void> {
    if (event.button !== 0) {
      return;
    }

    const target = event.target;
    if (target instanceof Element && target.closest("button")) {
      return;
    }

    await getCurrentWebviewWindow().startDragging();
  }

  async function handleMinimize(): Promise<void> {
    await getCurrentWebviewWindow().minimize();
  }

  async function handleToggleMaximize(): Promise<void> {
    const currentWindow = getCurrentWebviewWindow();
    const isMaximized = await currentWindow.isMaximized();
    if (isMaximized) {
      await currentWindow.unmaximize();
      return;
    }
    await currentWindow.maximize();
  }

  async function handleClose(): Promise<void> {
    await getCurrentWebviewWindow().close();
  }

  return (
    <header className="header-bar" aria-label="Application header">
      <div
        className="header-bar__drag-region"
        data-tauri-drag-region
        onMouseDown={(event) => void handleDragRegionMouseDown(event)}
      >
        <div className="header-bar__brand" aria-hidden="true">
          <svg className="brand-mic" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="8" y="2" width="8" height="13" rx="4" fill="url(#mic-body)" />
            <path
              d="M5 11a7 7 0 0 0 14 0"
              stroke="url(#mic-arc)"
              strokeWidth="1.75"
              strokeLinecap="round"
              fill="none"
            />
            <line x1="12" y1="18" x2="12" y2="21" stroke="#c8944a" strokeWidth="1.75" strokeLinecap="round" />
            <line x1="9" y1="21" x2="15" y2="21" stroke="#c8944a" strokeWidth="1.75" strokeLinecap="round" />
            <defs>
              <linearGradient id="mic-body" x1="12" y1="2" x2="12" y2="15" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#e8b46a" />
                <stop offset="100%" stopColor="#a06830" />
              </linearGradient>
              <linearGradient id="mic-arc" x1="5" y1="11" x2="19" y2="11" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#d4984a" />
                <stop offset="100%" stopColor="#a06830" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div className="header-bar__titles">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        <div className="header-bar__status" aria-label={`Status: ${status}`}>
          <span className="status-pill">
            <span className={`status-pill__dot status-pill__dot--${status}`} aria-hidden="true" />
            <span className="status-pill__label">{status}</span>
          </span>
        </div>
      </div>
      <div className="window-controls" aria-label="Window controls">
        <button
          type="button"
          className="window-controls__button"
          aria-label="Minimize window"
          onClick={() => void handleMinimize()}
        >
          <span className="window-controls__glyph window-controls__glyph--minimize" />
        </button>
        <button
          type="button"
          className="window-controls__button"
          aria-label="Maximize window"
          onClick={() => void handleToggleMaximize()}
        >
          <span className="window-controls__glyph window-controls__glyph--maximize" />
        </button>
        <button
          type="button"
          className="window-controls__button window-controls__button--close"
          aria-label="Close window"
          onClick={() => void handleClose()}
        >
          <span className="window-controls__glyph window-controls__glyph--close" />
        </button>
      </div>
    </header>
  );
}
