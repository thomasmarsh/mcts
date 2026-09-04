// BenchApp.tsx — Top-level bench UI. Thin wrapper around the tuner UI
// (`TunerApp`), which owns its own store and hash sub-routes
// (`#/tuner/...`). The former "round-robin runs" surface (a run list +
// launch form driving the retired Optuna tuner) is gone: round-robin runs
// are launched from the CLI (`bin/bench`) and have no browser UI.

import type { Component } from "solid-js";
import { TunerApp } from "./tuner/TunerApp.js";

export const BenchApp: Component = () => (
  <div id="bench-app">
    <TunerApp />
  </div>
);
