import type { ReactElement, ReactNode } from "react";
import type { FinalizedUtterance, HeaderStatus } from "../types/viewState";

interface TranscriptPanelProps {
  utterances: FinalizedUtterance[];
  preliminaryText: string;
  status: HeaderStatus;
  connectionError?: string;
}

// 2.0 s silence threshold that separates spoken paragraphs
const PARAGRAPH_GAP_SECONDS = 2.0;

function buildFormattedNodes(utterances: FinalizedUtterance[]): ReactNode[] {
  if (utterances.length === 0) return [];

  const nodes: ReactNode[] = [utterances[0].text];
  for (let i = 1; i < utterances.length; i++) {
    const gap = utterances[i].start_time - utterances[i - 1].end_time;
    if (gap > PARAGRAPH_GAP_SECONDS) {
      nodes.push(<br key={`br1-${i}`} />);
      nodes.push(<br key={`br2-${i}`} />);
    } else {
      nodes.push(" ");
    }
    nodes.push(utterances[i].text);
  }
  return nodes;
}

/**
 * Displays scrollable transcript content with an empty-state placeholder.
 *
 * Finalized utterances are rendered with paragraph breaks inserted between
 * utterances separated by more than PARAGRAPH_GAP_SECONDS seconds of silence.
 * Preliminary (in-progress) segments appear in gray italic immediately after.
 */
const EMPTY_STATE_MESSAGES: Record<HeaderStatus, string> = {
  connecting: "Connecting to recognition service…",
  waiting: "Model download required before recognition can start.",
  listening: "Start speaking — transcript will appear here.",
  paused: "Recognition paused.",
  error: "Transcript will appear here once recognition is connected.",
};

export function TranscriptPanel({
  utterances,
  preliminaryText,
  status,
  connectionError,
}: TranscriptPanelProps): ReactElement {
  const hasText = utterances.length > 0 || preliminaryText.trim().length > 0;
  const nodes = buildFormattedNodes(utterances);

  return (
    <section className="transcript-panel" aria-label="Transcript section">
      <div className="transcript-panel__header">
        <span className="transcript-panel__eyebrow">Transcription</span>
      </div>
      <div className={`transcript-panel__surface${connectionError ? " transcript-panel__surface--error" : ""}`}>
        {connectionError ? (
          <div className="transcript-panel__error" role="alert" aria-label={connectionError}>
            <svg className="transcript-error__icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <path d="M12 9v4M12 17h.01" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            </svg>
            <span className="transcript-error__message">{connectionError}</span>
          </div>
        ) : !hasText ? (
          <div className="transcript-panel__empty" aria-hidden="true">
            <div className="empty-glyph" />
            <p>{EMPTY_STATE_MESSAGES[status]}</p>
          </div>
        ) : null}
        <div
          className="transcript-panel__content"
          role="log"
          aria-live="polite"
          aria-label="Transcript text"
          tabIndex={0}
        >
          {nodes}
          {preliminaryText ? (
            <span className="transcript-panel__preliminary">{preliminaryText}</span>
          ) : null}
        </div>
      </div>
    </section>
  );
}
