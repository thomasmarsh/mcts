// NewGameFields.tsx — Margo's new-game config editor: a board-size picker,
// matching Druid's `NewGameFields.tsx` shape but over `n` (a single base
// width) instead of `Size` (`w`/`h`). `GameShell` owns the rest of the New
// Game dialog (seat pickers, dialog chrome) generically.

import type { Component } from "solid-js";
import { For } from "solid-js";
import { BOARD_SIZES, DEFAULT_N, type NewGameConfig } from "./types.js";

function isNewGameConfig(config: unknown): config is NewGameConfig {
  if (!config || typeof config !== "object") return false;
  return typeof (config as { n?: unknown }).n === "number";
}

export const NewGameFields: Component<{ config: unknown; onChange: (config: unknown) => void }> = (
  props,
) => {
  const n = () => (isNewGameConfig(props.config) ? props.config.n : DEFAULT_N);

  return (
    <label>
      Board size
      <select
        value={String(n())}
        onChange={(e) => {
          const value = Number(e.currentTarget.value);
          if (Number.isNaN(value)) return;
          props.onChange({ n: value } satisfies NewGameConfig);
        }}
      >
        <For each={BOARD_SIZES}>
          {(size) => (
            <option value={String(size)}>
              {size} × {size}
            </option>
          )}
        </For>
      </select>
    </label>
  );
};
