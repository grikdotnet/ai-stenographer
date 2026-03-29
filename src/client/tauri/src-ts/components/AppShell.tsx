import type { ReactElement } from "react";
import type { AppViewState } from "../types/viewState";
import { ButtonPanel } from "./ButtonPanel";
import { HeaderBar } from "./HeaderBar";
import { TranscriptPanel } from "./TranscriptPanel";

interface AppShellProps {
  viewState: AppViewState;
  onPauseToggle: () => void;
  onClear: () => void;
}

/**
 * Composes the Tauri shell layout for the standalone desktop client.
 */
export function AppShell({
  viewState,
  onPauseToggle,
  onClear,
}: AppShellProps): ReactElement {
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
      </section>
    </main>
  );
}
