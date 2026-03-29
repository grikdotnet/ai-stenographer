import { act, render, screen, waitFor } from "@testing-library/react";
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
});
