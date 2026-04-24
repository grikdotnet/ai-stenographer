import type { ReactElement } from "react";

interface ButtonPanelProps {
  isPaused: boolean;
  onPauseToggle: () => void;
  onClear: () => void;
}

/**
 * Renders the action row for transcript controls.
 */
export function ButtonPanel({
  isPaused,
  onPauseToggle,
  onClear,
}: ButtonPanelProps): ReactElement {
  return (
    <section className="button-panel" aria-label="Controls">
      <div className="button-panel__group">
        <button type="button" className="button button--secondary" onClick={onPauseToggle}>
          {isPaused ? (
            <svg className="button__icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M3 2.5a.5.5 0 0 1 .765-.424l9 5a.5.5 0 0 1 0 .848l-9 5A.5.5 0 0 1 3 13.5v-11z"/>
            </svg>
          ) : (
            <svg className="button__icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path d="M5.5 3.5A1.5 1.5 0 0 1 7 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5zm5 0A1.5 1.5 0 0 1 12 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5z"/>
            </svg>
          )}
          {isPaused ? "Resume" : "Pause"}
        </button>
        <button type="button" className="button button--danger" onClick={onClear}>
          <svg className="button__icon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
            <path fillRule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
          </svg>
          Clear
        </button>
      </div>
    </section>
  );
}
