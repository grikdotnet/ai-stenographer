import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        quickentry: resolve(__dirname, "quickentry.html"),
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src-ts/test/setup.ts",
    css: true,
    globals: true
  }
});
