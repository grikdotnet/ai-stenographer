import { useState, useEffect } from "react";
import type { ReactElement } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type {
  AppViewState,
  DownloadProgress,
  FinalizedUtterance,
  HeaderStatus,
  ModelInfo,
} from "./types/viewState";
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
  models: [],
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
                ? prev.serverState === "waiting_for_model"
                  ? "waiting"
                  : prev.status
                : event.payload.error
                  ? "error"
                  : "connecting",
              connectionError: event.payload.error,
            }));
          }
        )
      );

      unlisteners.push(
        await listen<{ state: string }>(
          "stt://server-state",
          (event) => {
            if (event.payload.state === "waiting_for_model") {
              void invoke("list_models");
            }
            setViewState((prev) => ({
              ...prev,
              serverState: event.payload.state,
              downloadDialogDismissed:
                event.payload.state === "running" ? false : prev.downloadDialogDismissed,
              status:
                event.payload.state === "running"
                  ? prev.isPaused
                    ? "paused"
                    : "listening"
                  : event.payload.state === "waiting_for_model"
                    ? "waiting"
                    : event.payload.state === "shutdown"
                      ? "error"
                    : "connecting",
            }));
          }
        )
      );

      unlisteners.push(
        await listen<{ models: ModelInfo[] }>(
          "stt://model-list",
          (event) => {
            setViewState((prev) => ({
              ...prev,
              models: event.payload.models,
            }));
          }
        )
      );

      unlisteners.push(
        await listen<{ status: string; request_id?: string }>(
          "stt://model-status",
          (event) => {
            if (event.payload.status === "downloading" || event.payload.status === "ready") {
              void invoke("list_models");
            }
          }
        )
      );

      unlisteners.push(
        await listen<DownloadProgress>(
          "stt://download-progress",
          (event) => {
            setViewState((prev) => ({
              ...prev,
              downloadProgress: event.payload,
              downloadDialogDismissed:
                event.payload.status === "downloading" || event.payload.status === "error"
                  ? false
                  : prev.downloadDialogDismissed,
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
        WaitingForServer: "connecting",
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

  async function handleRefreshModels(): Promise<void> {
    await invoke("list_models");
  }

  async function handleDownloadModel(modelName: string): Promise<void> {
    await invoke("download_model", { modelName });
  }

  function handleCloseModelDialog(): void {
    setViewState((prev) => ({
      ...prev,
      downloadDialogDismissed: true,
    }));
  }

  return (
    <AppShell
      viewState={viewState}
      onPauseToggle={() => void handlePauseToggle()}
      onClear={() => void handleClear()}
      onRefreshModels={() => void handleRefreshModels()}
      onDownloadModel={(modelName) => void handleDownloadModel(modelName)}
      onCloseModelDialog={handleCloseModelDialog}
    />
  );
}

export default App;
