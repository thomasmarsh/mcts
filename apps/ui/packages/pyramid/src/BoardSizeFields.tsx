// BoardSizeFields.tsx — Shared new-game board-size picker for every
// pyramid-family game whose only config is a single base width `n` (Margo,
// Akron both mirror `pyramid::{MIN_N, MAX_N, DEFAULT_N}` = 4..=10 / 7).
// `GameShell` owns the rest of the New Game dialog (seat pickers, dialog
// chrome) generically; this is just the per-game config editor slotted into
// it via `GameKindModule.NewGameFields`.

import type { Component } from "solid-js";
import { For } from "solid-js";

export interface NConfig {
  n: number;
}

function isNConfig(config: unknown): config is NConfig {
  if (!config || typeof config !== "object") return false;
  return typeof (config as { n?: unknown }).n === "number";
}

export function boardSizes(minN: number, maxN: number): number[] {
  return Array.from({ length: maxN - minN + 1 }, (_, i) => minN + i);
}

export const BoardSizeFields: Component<{
  config: unknown;
  onChange: (config: unknown) => void;
  sizes: number[];
  defaultN: number;
}> = (props) => {
  const n = () => (isNConfig(props.config) ? props.config.n : props.defaultN);

  return (
    <label>
      Board size
      <select
        value={String(n())}
        onChange={(e) => {
          const value = Number(e.currentTarget.value);
          if (Number.isNaN(value)) return;
          props.onChange({ n: value } satisfies NConfig);
        }}
      >
        <For each={props.sizes}>
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
