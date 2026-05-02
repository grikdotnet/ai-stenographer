import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { vi } from "vitest";
import { BACKDROP_TYPEWRITER_TEXTS } from "../content/backdropTypewriterTexts";
import { ModelDownloadDialog } from "./ModelDownloadDialog";
import type { DownloadProgress, ModelInfo } from "../types/viewState";

const missingModel: ModelInfo = {
  name: "parakeet",
  display_name: "Parakeet ASR",
  size_description: "1.25 GB",
  status: "missing",
};
const backdropTypewriterText = BACKDROP_TYPEWRITER_TEXTS.waitingForModel;

function renderDialog(options: {
  isOpen?: boolean;
  models?: ModelInfo[];
  downloadProgress?: DownloadProgress;
  typewriterText?: string;
  onDownloadModel?: (modelName: string) => void;
  onCancelDownload?: () => void;
  onClose?: () => void;
} = {}) {
  const props = {
    isOpen: options.isOpen ?? true,
    models: options.models ?? [missingModel],
    downloadProgress: options.downloadProgress,
    typewriterText: options.typewriterText,
    onDownloadModel: options.onDownloadModel ?? vi.fn(),
    onCancelDownload: options.onCancelDownload ?? vi.fn(),
    onClose: options.onClose ?? vi.fn(),
  };

  const renderResult = render(<ModelDownloadDialog {...props} />);
  return { ...props, ...renderResult };
}

describe("ModelDownloadDialog", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("does not render when closed", () => {
    renderDialog({ isOpen: false });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders missing model name and size when open", () => {
    const { container } = renderDialog();

    expect(screen.getByRole("dialog", { name: "Speech model required" })).toBeInTheDocument();
    const backdropText = container.querySelector(".model-dialog__typewriter");
    expect(backdropText).toHaveAttribute("aria-hidden", "true");
    expect(container.querySelector(".model-dialog__typewriter-caret")).toBeInTheDocument();
    expect(screen.getByText("Parakeet ASR")).toBeInTheDocument();
    expect(screen.getByText("1.25 GB")).toBeInTheDocument();
    expect(
      screen.getByText("Model must be downloaded before recognition can start.")
    ).toBeInTheDocument();
  });

  it("types the decorative backdrop sentence over time", () => {
    vi.useFakeTimers();
    const { container } = renderDialog({ typewriterText: "Dear Sir." });
    const backdropText = container.querySelector(".model-dialog__typewriter");

    expect(backdropText).toHaveTextContent("");

    act(() => {
      vi.advanceTimersByTime(250);
    });
    expect(backdropText).toHaveTextContent("D");

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(backdropText?.textContent?.length).toBeGreaterThan(1);

    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(backdropText).toHaveTextContent("Dear Sir.");

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(backdropText).toHaveTextContent("Dear Sir.");

    vi.useRealTimers();
  });

  it("calls onDownloadModel with the primary model name when Download is clicked", async () => {
    const user = userEvent.setup();
    const onDownloadModel = vi.fn();
    renderDialog({ onDownloadModel });

    await user.click(screen.getByRole("button", { name: "Download" }));

    expect(onDownloadModel).toHaveBeenCalledWith("parakeet");
  });

  it("starts window dragging when the empty backdrop is pressed with the left mouse button", () => {
    const currentWindow = getCurrentWebviewWindow();
    const { container } = renderDialog();
    const backdrop = container.querySelector(".model-dialog__backdrop");

    backdrop?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, button: 0 }));

    expect(currentWindow.startDragging).toHaveBeenCalledOnce();
  });

  it("does not start window dragging when the dialog card is pressed", () => {
    const currentWindow = getCurrentWebviewWindow();
    renderDialog();

    screen
      .getByRole("dialog", { name: "Speech model required" })
      .dispatchEvent(new MouseEvent("mousedown", { bubbles: true, button: 0 }));

    expect(currentWindow.startDragging).not.toHaveBeenCalled();
  });

  it("does not start window dragging for non-left mouse buttons on the backdrop", () => {
    const currentWindow = getCurrentWebviewWindow();
    const { container } = renderDialog();
    const backdrop = container.querySelector(".model-dialog__backdrop");

    backdrop?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, button: 1 }));
    backdrop?.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, button: 2 }));

    expect(currentWindow.startDragging).not.toHaveBeenCalled();
  });

  it("shows loading state without Download when no model is available yet", () => {
    renderDialog({ models: [] });

    expect(screen.getByText("Loading available models")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download" })).not.toBeInTheDocument();
  });

  it("does not render Refresh when a missing model is available", () => {
    renderDialog();

    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download" })).toHaveClass("model-dialog__download");
  });

  it("disables Download while the selected model is downloading", () => {
    renderDialog({
      models: [{ ...missingModel, status: "downloading" }],
    });

    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
  });

  it("disables Download while progress status is downloading", () => {
    renderDialog({
      downloadProgress: {
        model_name: "parakeet",
        status: "downloading",
      },
    });

    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
  });

  it("turns header Close into Cancel during download and waits for confirmation", async () => {
    const user = userEvent.setup();
    const onCancelDownload = vi.fn();
    renderDialog({
      onCancelDownload,
      downloadProgress: {
        model_name: "parakeet",
        status: "downloading",
        progress: 0.42,
      },
    });

    expect(screen.queryByRole("button", { name: "Close" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancelDownload).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: "Cancelling..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("shows percentage progress when reported", () => {
    renderDialog({
      downloadProgress: {
        model_name: "parakeet",
        status: "downloading",
        progress: 0.42,
      },
    });

    expect(screen.getByRole("progressbar", { name: "Model download progress" })).toHaveAttribute(
      "value",
      "42"
    );
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("shows formatted bytes when byte counts are reported", () => {
    renderDialog({
      downloadProgress: {
        model_name: "parakeet",
        status: "downloading",
        downloaded_bytes: 536870912,
        total_bytes: 1073741824,
      },
    });

    expect(screen.getByText("512 MB / 1 GB")).toBeInTheDocument();
  });

  it("shows error text and retry action when progress status is error", async () => {
    const user = userEvent.setup();
    const onDownloadModel = vi.fn();
    renderDialog({
      onDownloadModel,
      downloadProgress: {
        model_name: "parakeet",
        status: "error",
        error_message: "network disconnected",
      },
    });

    expect(screen.getByText("network disconnected")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(onDownloadModel).toHaveBeenCalledWith("parakeet");
  });
});
