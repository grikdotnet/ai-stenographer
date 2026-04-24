import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { vi } from "vitest";
import App from "./App";

const eventListeners = new Map<string, (event: { payload: unknown }) => void>();

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn((eventName: string, handler: (event: { payload: unknown }) => void) => {
    eventListeners.set(eventName, handler);
    return Promise.resolve(() => {
      eventListeners.delete(eventName);
    });
  }),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn((command: string) => {
    if (command === "get_state") {
      return Promise.resolve("Starting");
    }
    if (command === "get_connection_status") {
      return Promise.resolve({ connected: false });
    }
    return Promise.resolve(undefined);
  }),
}));

describe("App", () => {
  beforeEach(() => {
    vi.mocked(invoke).mockImplementation((command: string) => {
      if (command === "get_state") {
        return Promise.resolve("Starting");
      }
      if (command === "get_connection_status") {
        return Promise.resolve({ connected: false });
      }
      return Promise.resolve(undefined);
    });
  });

  afterEach(() => {
    eventListeners.clear();
    vi.clearAllMocks();
  });

  it("mounts the shell with header, controls, and transcript regions", () => {
    render(<App />);

    expect(screen.getByLabelText("Application header")).toBeInTheDocument();
    expect(screen.getByLabelText("Controls")).toBeInTheDocument();
    expect(screen.getByLabelText("Transcript section")).toBeInTheDocument();
  });

  it("shows connecting status on initial render", () => {
    render(<App />);

    expect(screen.getByLabelText("Status: connecting")).toBeInTheDocument();
  });

  it("syncs to listening status when get_state returns Running after mount", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    vi.mocked(invoke).mockImplementation((command: string) => {
      if (command === "get_state") {
        return Promise.resolve("Running");
      }
      if (command === "get_connection_status") {
        return Promise.resolve({ connected: true });
      }
      return Promise.resolve(undefined);
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByLabelText("Status: listening")).toBeInTheDocument();
    });
  });

  it("shows the connection error message when the backend emits a failed connection status", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    vi.mocked(invoke).mockImplementation((command: string) => {
      if (command === "get_state") {
        return Promise.resolve("Starting");
      }
      if (command === "get_connection_status") {
        return Promise.resolve({ connected: false });
      }
      return Promise.resolve(undefined);
    });

    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://connection-status")).toBe(true);
    });

    await act(async () => {
      eventListeners.get("stt://connection-status")?.({
        payload: {
          connected: false,
          error: "Connection failed: target machine actively refused it.",
        },
      });
    });

    expect(
      await screen.findByRole("alert", {
        name: "Connection failed: target machine actively refused it.",
      })
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Status: error")).toBeInTheDocument();
  });

  it("keeps the error status visible when get_state resolves after a connection failure", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    let resolveState: ((value: string) => void) | undefined;
    vi.mocked(invoke).mockImplementation(
      (command: string) => {
        if (command === "get_state") {
          return new Promise<string>((resolve) => {
            resolveState = resolve;
          });
        }
        if (command === "get_connection_status") {
          return Promise.resolve({ connected: false });
        }
        return Promise.resolve(undefined);
      }
    );

    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://connection-status")).toBe(true);
    });

    await act(async () => {
      eventListeners.get("stt://connection-status")?.({
        payload: {
          connected: false,
          error: "Connection failed: target machine actively refused it.",
        },
      });
    });

    await act(async () => {
      resolveState?.("Starting");
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Connection failed: target machine actively refused it."
    );
    expect(screen.getByLabelText("Status: error")).toBeInTheDocument();
  });

  it("shows the persisted connection error when startup failure happened before listeners subscribed", async () => {
    const { invoke } = await import("@tauri-apps/api/core");
    vi.mocked(invoke).mockImplementation((command: string) => {
      if (command === "get_state") {
        return Promise.resolve("Starting");
      }
      if (command === "get_connection_status") {
        return Promise.resolve({
          connected: false,
          error: "Connection failed before the UI subscribed.",
        });
      }
      return Promise.resolve(undefined);
    });

    render(<App />);

    expect(
      await screen.findByRole("alert", {
        name: "Connection failed before the UI subscribed.",
      })
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Status: error")).toBeInTheDocument();
  });

  it("invokes list_models when the server enters waiting_for_model", async () => {
    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://server-state")).toBe(true);
    });
    vi.mocked(invoke).mockClear();

    await act(async () => {
      eventListeners.get("stt://server-state")?.({
        payload: { state: "waiting_for_model" },
      });
    });

    expect(invoke).toHaveBeenCalledWith("list_models");
  });

  it("renders the model download dialog when a missing model list arrives", async () => {
    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://server-state")).toBe(true);
      expect(eventListeners.has("stt://model-list")).toBe(true);
    });

    await act(async () => {
      eventListeners.get("stt://server-state")?.({
        payload: { state: "waiting_for_model" },
      });
      eventListeners.get("stt://model-list")?.({
        payload: {
          models: [
            {
              name: "parakeet",
              display_name: "Parakeet ASR",
              size_description: "1.25 GB",
              status: "missing",
            },
          ],
        },
      });
    });

    expect(await screen.findByRole("dialog", { name: "Speech model required" })).toBeInTheDocument();
    expect(screen.getByText("Parakeet ASR")).toBeInTheDocument();
  });

  it("invokes download_model with the selected model when Download is clicked", async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://server-state")).toBe(true);
      expect(eventListeners.has("stt://model-list")).toBe(true);
    });

    await act(async () => {
      eventListeners.get("stt://server-state")?.({
        payload: { state: "waiting_for_model" },
      });
      eventListeners.get("stt://model-list")?.({
        payload: {
          models: [
            {
              name: "parakeet",
              display_name: "Parakeet ASR",
              size_description: "1.25 GB",
              status: "missing",
            },
          ],
        },
      });
    });

    vi.mocked(invoke).mockClear();
    await user.click(await screen.findByRole("button", { name: "Download" }));

    expect(invoke).toHaveBeenCalledWith("download_model", { modelName: "parakeet" });
  });

  it("updates visible dialog progress when download progress arrives", async () => {
    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://server-state")).toBe(true);
      expect(eventListeners.has("stt://model-list")).toBe(true);
      expect(eventListeners.has("stt://download-progress")).toBe(true);
    });

    await act(async () => {
      eventListeners.get("stt://server-state")?.({
        payload: { state: "waiting_for_model" },
      });
      eventListeners.get("stt://model-list")?.({
        payload: {
          models: [
            {
              name: "parakeet",
              display_name: "Parakeet ASR",
              size_description: "1.25 GB",
              status: "missing",
            },
          ],
        },
      });
      eventListeners.get("stt://download-progress")?.({
        payload: {
          model_name: "parakeet",
          status: "downloading",
          progress: 0.65,
        },
      });
    });

    expect(await screen.findByRole("progressbar", { name: "Model download progress" })).toHaveAttribute(
      "value",
      "65"
    );
    expect(screen.getByText("65%")).toBeInTheDocument();
  });

  it("removes the blocking dialog and returns to listening when the server starts running", async () => {
    render(<App />);

    await waitFor(() => {
      expect(eventListeners.has("stt://server-state")).toBe(true);
      expect(eventListeners.has("stt://model-list")).toBe(true);
    });

    await act(async () => {
      eventListeners.get("stt://server-state")?.({
        payload: { state: "waiting_for_model" },
      });
      eventListeners.get("stt://model-list")?.({
        payload: {
          models: [
            {
              name: "parakeet",
              display_name: "Parakeet ASR",
              size_description: "1.25 GB",
              status: "missing",
            },
          ],
        },
      });
    });

    expect(await screen.findByRole("dialog", { name: "Speech model required" })).toBeInTheDocument();

    await act(async () => {
      eventListeners.get("stt://server-state")?.({
        payload: { state: "running" },
      });
    });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Speech model required" })).not.toBeInTheDocument();
    });
    expect(screen.getByLabelText("Status: listening")).toBeInTheDocument();
  });
});
