import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import type { MouseEvent } from "react";
import type { ReactElement } from "react";
import appIconUrl from "../assets/app-icon.png";
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
          <img className="brand-icon" src={appIconUrl} alt="" />
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
