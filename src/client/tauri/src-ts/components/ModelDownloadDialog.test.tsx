import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { ModelDownloadDialog } from "./ModelDownloadDialog";
import type { DownloadProgress, ModelInfo } from "../types/viewState";

const missingModel: ModelInfo = {
  name: "parakeet",
  display_name: "Parakeet ASR",
  size_description: "1.25 GB",
  status: "missing",
};

function renderDialog(options: {
  isOpen?: boolean;
  models?: ModelInfo[];
  downloadProgress?: DownloadProgress;
  onRefreshModels?: () => void;
  onDownloadModel?: (modelName: string) => void;
  onClose?: () => void;
} = {}) {
  const props = {
    isOpen: options.isOpen ?? true,
    models: options.models ?? [missingModel],
    downloadProgress: options.downloadProgress,
    onRefreshModels: options.onRefreshModels ?? vi.fn(),
    onDownloadModel: options.onDownloadModel ?? vi.fn(),
    onClose: options.onClose ?? vi.fn(),
  };

  render(<ModelDownloadDialog {...props} />);
  return props;
}

describe("ModelDownloadDialog", () => {
  it("does not render when closed", () => {
    renderDialog({ isOpen: false });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders missing model name and size when open", () => {
    renderDialog();

    expect(screen.getByRole("dialog", { name: "Speech model required" })).toBeInTheDocument();
    expect(screen.getByText("Parakeet ASR")).toBeInTheDocument();
    expect(screen.getByText("1.25 GB")).toBeInTheDocument();
    expect(
      screen.getByText("Model must be downloaded before recognition can start.")
    ).toBeInTheDocument();
  });

  it("calls onRefreshModels when Refresh is clicked", async () => {
    const user = userEvent.setup();
    const onRefreshModels = vi.fn();
    renderDialog({ onRefreshModels });

    await user.click(screen.getByRole("button", { name: "Refresh" }));

    expect(onRefreshModels).toHaveBeenCalledTimes(1);
  });

  it("calls onDownloadModel with the primary model name when Download is clicked", async () => {
    const user = userEvent.setup();
    const onDownloadModel = vi.fn();
    renderDialog({ onDownloadModel });

    await user.click(screen.getByRole("button", { name: "Download" }));

    expect(onDownloadModel).toHaveBeenCalledWith("parakeet");
  });

  it("shows loading state without Download when no model is available yet", () => {
    renderDialog({ models: [] });

    expect(screen.getByText("Loading available models")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download" })).not.toBeInTheDocument();
  });

  it("disables Download while the selected model is downloading", () => {
    renderDialog({
      models: [{ ...missingModel, status: "downloading" }],
    });

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

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(onDownloadModel).toHaveBeenCalledWith("parakeet");
  });
});
