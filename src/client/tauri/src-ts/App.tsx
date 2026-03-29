import { useState, useEffect } from "react";
import type { ReactElement } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type { AppViewState, FinalizedUtterance, HeaderStatus } from "./types/viewState";
import { AppShell } from "./components/AppShell";

interface ConnectionStatusPayload {
  connected: boolean;
  error?: string;
}

const INITIAL_VIEW_STATE: AppViewState = {
  title: "Speech-to-Text",
  subtitle: "Real-time Recognition",
  status: "connecting",
  utterances: [],
  preliminaryText: "",
  isPaused: false,
};

/**
 * Hosts Tauri event-driven UI state for the desktop client.
 *
 * Subscribes to backend events for transcript updates, state changes,
 * connection status, and insertion state. Delegates user actions to
 * Tauri commands.
 */
function App(): ReactElement {
  const [viewState, setViewState] = useState<AppViewState>(INITIAL_VIEW_STATE);

  useEffect(() => {
    const unlisteners: Array<() => void> = [];

    async function setupListeners(): Promise<void> {
      unlisteners.push(
        await listen<{ action: string; utterances: FinalizedUtterance[]; preliminary: string }>(
          "stt://transcript-update",
          (event) => {
            setViewState((prev) => ({
              ...prev,
              utterances: event.payload.utterances,
              preliminaryText: event.payload.preliminary,
            }));
          }
        )
      );

      unlisteners.push(
        await listen<{ old: string; new: string }>(
          "stt://state-changed",
          (event) => {
            const statusMap: Record<string, HeaderStatus> = {
              Starting: "connecting",
              Running: "listening",
              Paused: "paused",
              Shutdown: "error",
            };
            setViewState((prev) => ({
              ...prev,
              status: statusMap[event.payload.new] ?? "connecting",
              isPaused: event.payload.new === "Paused",
            }));
          }
        )
      );

      unlisteners.push(
        await listen<{ connected: boolean; error?: string }>(
          "stt://connection-status",
          (event) => {
            setViewState((prev) => ({
              ...prev,
              status: event.payload.connected
                ? "listening"
                : event.payload.error
                  ? "error"
                  : "connecting",
              connectionError: event.payload.error,
            }));
          }
        )
      );

    }

    void (async () => {
      await setupListeners();
      syncStateFromRust();
    })();

    function syncStateFromRust(): void {
      const statusMap: Record<string, HeaderStatus> = {
        Starting: "connecting",
        Running: "listening",
        Paused: "paused",
        Shutdown: "error",
      };
      void Promise.all([
        invoke<string>("get_state"),
        invoke<ConnectionStatusPayload>("get_connection_status"),
      ]).then(([currentState, connectionStatus]) => {
        setViewState((prev) => {
          const connectionError = connectionStatus.error ?? prev.connectionError;
          const status: HeaderStatus = connectionStatus.connected
            ? "listening"
            : connectionStatus.error
              ? "error"
              : connectionError && currentState === "Starting"
                ? "error"
                : statusMap[currentState] ?? "connecting";
          return {
            ...prev,
            status,
            isPaused: currentState === "Paused",
            connectionError,
          };
        });
        if (currentState === "Starting" && !connectionStatus.error) {
          setTimeout(syncStateFromRust, 500);
        }
      });
    }

    return () => {
      unlisteners.forEach((fn) => fn());
    };
  }, []);

  async function handlePauseToggle(): Promise<void> {
    if (viewState.isPaused) {
      await invoke("resume");
    } else {
      await invoke("pause");
    }
  }

  async function handleClear(): Promise<void> {
    await invoke("clear");
  }

  return (
    <AppShell
      viewState={viewState}
      onPauseToggle={() => void handlePauseToggle()}
      onClear={() => void handleClear()}
    />
  );
}

export default App;
