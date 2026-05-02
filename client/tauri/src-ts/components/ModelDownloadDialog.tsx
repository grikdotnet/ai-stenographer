import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { useEffect, useState, type MouseEvent, type ReactElement } from "react";
import { BACKDROP_TYPEWRITER_TEXTS } from "../content/backdropTypewriterTexts";
import type { DownloadProgress, ModelInfo } from "../types/viewState";

interface ModelDownloadDialogProps {
  isOpen: boolean;
  models: ModelInfo[];
  downloadProgress?: DownloadProgress;
  typewriterText?: string;
  onDownloadModel: (modelName: string) => void;
  onCancelDownload: () => void;
  onClose: () => void;
}
const TYPEWRITER_START_DELAY_MS = 250;
const TYPEWRITER_CADENCE_MS = [80, 58, 112, 52, 120, 74, 104, 58, 88, 76, 125, 50];
const TYPEWRITER_JITTER_MS = 50;
const TYPEWRITER_HESITATION_CHANCE = 0.14;
const TYPEWRITER_HESITATION_MS = 100;

function getTypewriterDelayMs(text: string, nextCharacterIndex: number): number {
  const previousCharacter = text[nextCharacterIndex - 1];
  const nextCharacter = text[nextCharacterIndex];

  if (previousCharacter === ",") return 250;
  if (previousCharacter === "/") return 200;
  if (nextCharacter === " ") return 60;
  if (previousCharacter === " ") return 120;

  const baseDelay = TYPEWRITER_CADENCE_MS[nextCharacterIndex % TYPEWRITER_CADENCE_MS.length];
  const jitter = Math.round(Math.random() * TYPEWRITER_JITTER_MS);
  const hesitation = Math.random() < TYPEWRITER_HESITATION_CHANCE ? TYPEWRITER_HESITATION_MS : 0;
  return baseDelay + jitter + hesitation;
}

function selectPrimaryModel(
  models: ModelInfo[],
  downloadProgress?: DownloadProgress
): ModelInfo | undefined {
  if (downloadProgress?.model_name) {
    const matching = models.find((model) => model.name === downloadProgress.model_name);
    if (matching) return matching;
  }

  return (
    models.find((model) => model.status === "downloading") ??
    models.find((model) => model.status === "missing") ??
    models[0]
  );
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }

  const formatted = Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
  return `${formatted} ${units[unitIndex]}`;
}

function getStatusText(model?: ModelInfo, downloadProgress?: DownloadProgress): string {
  if (downloadProgress?.status === "downloading") {
    return "Downloading speech model.";
  }
  if (downloadProgress?.status === "complete") {
    return "Model is ready or finishing setup.";
  }
  if (downloadProgress?.status === "error") {
    return downloadProgress.error_message ?? "Download failed. Try again.";
  }
  if (model?.status === "downloaded") {
    return "Model is ready or finishing setup.";
  }
  if (model?.status === "downloading") {
    return "Downloading speech model.";
  }
  return "Model must be downloaded before recognition can start.";
}

/**
 * Presents the missing-model download flow as a blocking desktop dialog.
 */
export function ModelDownloadDialog({
  isOpen,
  models,
  downloadProgress,
  typewriterText = BACKDROP_TYPEWRITER_TEXTS.waitingForModel,
  onDownloadModel,
  onCancelDownload,
  onClose,
}: ModelDownloadDialogProps): ReactElement | null {
  const [cancelRequested, setCancelRequested] = useState(false);
  const [backdropTypewriterText, setBackdropTypewriterText] = useState("");

  async function handleBackdropMouseDown(event: MouseEvent<HTMLDivElement>): Promise<void> {
    if (event.button !== 0 || event.target !== event.currentTarget) {
      return;
    }

    await getCurrentWebviewWindow().startDragging();
  }

  useEffect(() => {
    if (downloadProgress?.status !== "downloading") {
      setCancelRequested(false);
    }
  }, [downloadProgress?.status]);

  useEffect(() => {
    let nextCharacterTimer: number | undefined;
    let characterCount = 0;

    function clearTimers(): void {
      if (nextCharacterTimer !== undefined) {
        window.clearTimeout(nextCharacterTimer);
      }
    }

    function scheduleTyping(delayMs: number): void {
      nextCharacterTimer = window.setTimeout(typeNextCharacter, delayMs);
    }

    function typeNextCharacter(): void {
      characterCount += 1;
      setBackdropTypewriterText(typewriterText.slice(0, characterCount));

      if (characterCount < typewriterText.length) {
        scheduleTyping(getTypewriterDelayMs(typewriterText, characterCount));
        return;
      }
    }

    if (!isOpen) {
      setBackdropTypewriterText("");
      return clearTimers;
    }

    setBackdropTypewriterText("");
    nextCharacterTimer = window.setTimeout(typeNextCharacter, TYPEWRITER_START_DELAY_MS);

    return clearTimers;
  }, [isOpen, typewriterText]);

  if (!isOpen) return null;

  const primaryModel = selectPrimaryModel(models, downloadProgress);
  const primaryModelName = primaryModel?.name ?? downloadProgress?.model_name;
  const displayName = primaryModel?.display_name ?? primaryModelName ?? "Speech model";
  const isDownloading = downloadProgress?.status === "downloading" || primaryModel?.status === "downloading";
  const isReady = downloadProgress?.status === "complete" || primaryModel?.status === "downloaded";
  const canDownload = Boolean(primaryModelName) && !isDownloading && !isReady;
  const isError = downloadProgress?.status === "error";
  const canClose = !isDownloading && !isError;
  const headerActionLabel = isDownloading
    ? cancelRequested
      ? "Cancelling..."
      : "Cancel"
    : "Close";
  const headerActionDisabled = isDownloading ? cancelRequested : !canClose;
  const handleHeaderAction = isDownloading
    ? () => {
        setCancelRequested(true);
        onCancelDownload();
      }
    : onClose;
  const progressPercent =
    downloadProgress?.progress === undefined
      ? undefined
      : Math.max(0, Math.min(100, Math.round(downloadProgress.progress * 100)));
  const byteText =
    downloadProgress?.downloaded_bytes !== undefined && downloadProgress.total_bytes !== undefined
      ? `${formatBytes(downloadProgress.downloaded_bytes)} / ${formatBytes(downloadProgress.total_bytes)}`
      : undefined;

  return (
    <div
      className="model-dialog__backdrop"
      onMouseDown={(event) => void handleBackdropMouseDown(event)}
    >
      <p className="model-dialog__typewriter" aria-hidden="true">
        <span>{backdropTypewriterText}</span>
        <span className="model-dialog__typewriter-caret" />
      </p>
      <section
        className="model-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-dialog-title"
      >
        <header className="model-dialog__header">
          <div>
            <p className="model-dialog__eyebrow">Recognition setup</p>
            <h2 id="model-dialog-title">Speech model required</h2>
          </div>
          <button
            type="button"
            className="button button--secondary model-dialog__close"
            onClick={handleHeaderAction}
            disabled={headerActionDisabled}
          >
            {headerActionLabel}
          </button>
        </header>

        <div className="model-dialog__body">
          {primaryModelName || downloadProgress ? (
            <div className="model-dialog__model">
              <div>
                <span className="model-dialog__model-name">{displayName}</span>
                {primaryModel?.size_description ? (
                  <span className="model-dialog__model-size">{primaryModel.size_description}</span>
                ) : null}
              </div>
              <p className={isError ? "model-dialog__status model-dialog__status--error" : "model-dialog__status"}>
                {getStatusText(primaryModel, downloadProgress)}
              </p>
            </div>
          ) : (
            <div className="model-dialog__model">
              <span className="model-dialog__model-name">Loading available models</span>
              <p className="model-dialog__status">Loading model list before recognition can start.</p>
            </div>
          )}

          {isDownloading ? (
            <div className="model-dialog__progress">
              {progressPercent === undefined ? (
                <div
                  className="model-dialog__progress-track model-dialog__progress-track--indeterminate"
                  role="progressbar"
                  aria-label="Model download progress"
                />
              ) : (
                <progress
                  aria-label="Model download progress"
                  value={progressPercent}
                  max={100}
                />
              )}
              <div className="model-dialog__progress-meta">
                <span>{progressPercent === undefined ? "Preparing download" : `${progressPercent}%`}</span>
                {byteText ? <span>{byteText}</span> : null}
              </div>
            </div>
          ) : null}
        </div>

        {primaryModelName ? (
          <div className="model-dialog__actions">
            <button
              type="button"
              className="button model-dialog__download"
              onClick={() => onDownloadModel(primaryModelName)}
              disabled={!canDownload}
            >
              {isError ? "Retry" : "Download"}
            </button>
          </div>
        ) : null}
      </section>
    </div>
  );
}
