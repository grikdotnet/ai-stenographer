import type { ReactElement } from "react";
import type { AppViewState } from "../types/viewState";
import { ButtonPanel } from "./ButtonPanel";
import { HeaderBar } from "./HeaderBar";
import { ModelDownloadDialog } from "./ModelDownloadDialog";
import { TranscriptPanel } from "./TranscriptPanel";

interface AppShellProps {
  viewState: AppViewState;
  onPauseToggle: () => void;
  onClear: () => void;
  onDownloadModel: (modelName: string) => void;
  onCancelDownload: () => void;
  onCloseModelDialog: () => void;
}

/**
 * Composes the Tauri shell layout for the standalone desktop client.
 */
export function AppShell({
  viewState,
  onPauseToggle,
  onClear,
  onDownloadModel,
  onCancelDownload,
  onCloseModelDialog,
}: AppShellProps): ReactElement {
  const downloadStatus = viewState.downloadProgress?.status;
  const hasReadyModel =
    viewState.models.some((model) => model.status === "downloaded") ||
    downloadStatus === "complete";
  const waitingForMissingModel =
    viewState.serverState === "waiting_for_model" && !hasReadyModel;
  const shouldShowModelDialog =
    viewState.serverState !== "running" &&
    (downloadStatus === "downloading" ||
      downloadStatus === "error" ||
      downloadStatus === "complete" ||
      (waitingForMissingModel && !viewState.downloadDialogDismissed));

  return (
    <main className="app-shell">
      <div className="app-shell__backdrop" aria-hidden="true" />
      <section className="app-shell__window">
        <HeaderBar
          title={viewState.title}
          subtitle={viewState.subtitle}
          status={viewState.status}
        />
        <ButtonPanel
          isPaused={viewState.isPaused}
          onPauseToggle={onPauseToggle}
          onClear={onClear}
        />
        <TranscriptPanel
          utterances={viewState.utterances}
          preliminaryText={viewState.preliminaryText}
          status={viewState.status}
          connectionError={viewState.connectionError}
        />
        <ModelDownloadDialog
          isOpen={shouldShowModelDialog}
          models={viewState.models}
          downloadProgress={viewState.downloadProgress}
          onDownloadModel={onDownloadModel}
          onCancelDownload={onCancelDownload}
          onClose={onCloseModelDialog}
        />
      </section>
    </main>
  );
}
