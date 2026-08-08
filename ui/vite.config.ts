import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

export default defineConfig({
  plugins: [solid()],
  build: {
    outDir: "../server/static/dist",
    emptyOutDir: true,
    // Three.js is ~550 kB minified — that's just the library's size.
    // Set the limit high enough to avoid a false-positive warning for it.
    chunkSizeWarningLimit: 600,
    // Rolldown's manualChunks is a function (not an object like Rollup's).
    // Route "three" into its own chunk so the druid game module stays under
    // the limit.
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("node_modules/three")) return "three";
        },
      },
    },
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