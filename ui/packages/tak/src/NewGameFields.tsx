// NewGameFields.tsx — Tak's new-game config editor: a board-size picker,
// mirroring `ui/packages/druid/src/NewGameFields.tsx`. Offers the engine's
// full supported range (`games/tak/src/lib.rs`'s `State<const N: usize>`,
// N in 3..=6) even though `games/tak/src/main.rs` is hardcoded to `State<5>`
// today (`new_state` ignores `config` and always returns a 5x5 board) --
// TODO: once a future session generalizes `main.rs` to dispatch over N, this
// picker becomes fully functional with no changes here; the rest of this
// package already derives board size from the actual returned state
// (`types.ts`'s `boardSizeFromTps`), never from what was requested.

import type { Component } from "solid-js";
import { For } from "solid-js";
import { BOARD_SIZES, DEFAULT_SIZE, type NewGameConfig } from "./types.js";

function isNewGameConfig(config: unknown): config is NewGameConfig {
  if (!config || typeof config !== "object") return false;
  return typeof (config as { size?: unknown }).size === "number";
}

export const NewGameFields: Component<{ config: unknown; onChange: (config: unknown) => void }> = (props) => {
  const size = () => (isNewGameConfig(props.config) ? props.config.size : DEFAULT_SIZE);

  return (
    <label>
      Board size
      <select
        value={size()}
        onChange={(e) => props.onChange({ size: Number(e.currentTarget.value) } satisfies NewGameConfig)}
      >
        <For each={BOARD_SIZES}>{(n) => <option value={n}>{n} × {n}</option>}</For>
      </select>
    </label>
  );
};
