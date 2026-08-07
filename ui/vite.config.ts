import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

export default defineConfig({
  plugins: [solid()],
  build: {
    outDir: "../server/static/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:7878",
    },
  },
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.{ts,tsx}", "packages/*/tests/**/*.test.{ts,tsx}"],
    setupFiles: ["tests/setup.ts"],
  },
});