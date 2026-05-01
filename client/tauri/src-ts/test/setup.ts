import "@testing-library/jest-dom/vitest";

const mockWebviewWindow = vi.hoisted(() => ({
  minimize: vi.fn(),
  maximize: vi.fn(),
  unmaximize: vi.fn(),
  isMaximized: vi.fn(() => Promise.resolve(false)),
  close: vi.fn(),
  startDragging: vi.fn(),
}));

vi.mock("@tauri-apps/api/webviewWindow", () => ({
  getCurrentWebviewWindow: () => mockWebviewWindow,
}));
