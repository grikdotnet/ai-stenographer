import React from "react";
import ReactDOM from "react-dom/client";
import { useState, useEffect } from "react";
import type { MouseEvent } from "react";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import "./quickentry.css";

function QuickEntryApp(): React.ReactElement {
  const [text, setText] = useState<string>("");

  useEffect(() => {
    const unlisteners: Array<() => void> = [];

    async function setup(): Promise<void> {
      unlisteners.push(await listen<{ text: string }>(
        "stt://quickentry-text",
        (event) => { setText(event.payload.text); }
      ));

      unlisteners.push(await listen<{ visible: boolean }>(
        "stt://quickentry-visibility",
        (event) => {
          if (!event.payload.visible) {
            setText("");
          }
        }
      ));
    }

    setup();
    return () => { unlisteners.forEach((fn) => fn()); };
  }, []);

  async function handleTitlebarMouseDown(event: MouseEvent<HTMLDivElement>): Promise<void> {
    if (event.button !== 0) {
      return;
    }
    await getCurrentWebviewWindow().startDragging();
  }

  return (
    <div className="quickentry">
      <div
        className="quickentry__titlebar"
        onMouseDown={(event) => void handleTitlebarMouseDown(event)}
      >
        <span className="quickentry__title">Quick Entry</span>
      </div>
      <div className="quickentry__content">
        {text || <span className="quickentry__placeholder">Listening...</span>}
      </div>
      <div className="quickentry__hints">
        <span>Enter to send</span>
        <span className="quickentry__separator">·</span>
        <span>Esc to cancel</span>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <QuickEntryApp />
  </React.StrictMode>
);
