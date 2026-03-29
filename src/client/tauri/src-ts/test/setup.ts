import "@testing-library/jest-dom/vitest";

vi.mock("@tauri-apps/api/webviewWindow", () => ({
  getCurrentWebviewWindow: () => ({
    minimize: vi.fn(),
    maximize: vi.fn(),
    unmaximize: vi.fn(),
    isMaximized: vi.fn(() => Promise.resolve(false)),
    close: vi.fn(),
    startDragging: vi.fn(),
  }),
}));
