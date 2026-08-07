// NewGameFields.tsx — Druid's new-game config editor: a board-size picker,
// matching app.js's `#new-size` select. `GameShell` owns the rest of the
// New Game dialog (seat pickers, dialog chrome) generically; this component
// only edits the game-specific `config` blob.

import type { Component } from "solid-js";
import { For } from "solid-js";
import { BOARD_SIZES, DEFAULT_SIZE, type NewGameConfig } from "./types.js";

function isNewGameConfig(config: unknown): config is NewGameConfig {
  if (!config || typeof config !== "object") return false;
  const size = (config as { size?: unknown }).size;
  return !!size && typeof size === "object" && "w" in size && "h" in size;
}

export const NewGameFields: Component<{ config: unknown; onChange: (config: unknown) => void }> = (props) => {
  const size = () => (isNewGameConfig(props.config) ? props.config.size : DEFAULT_SIZE);

  return (
    <label>
      Board size
      <select
        value={`${size().w}x${size().h}`}
        onChange={(e) => {
          const [w, h] = e.currentTarget.value.split("x").map(Number);
          if (w === undefined || h === undefined) return;
          props.onChange({ size: { w, h } } satisfies NewGameConfig);
        }}
      >
        <For each={BOARD_SIZES}>
          {(s) => <option value={`${s.w}x${s.h}`}>{s.w} × {s.h}</option>}
        </For>
      </select>
    </label>
  );
};
