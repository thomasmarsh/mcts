// NewGameFields.tsx — Board-size `<select>` for goban games, mirroring
// `@mcts/druid`'s `NewGameFields.tsx` but keyed by a single `size: number`
// rather than Druid's `{ w, h }`: AtariGo/Gonnect boards are always square,
// and `WORDS` is mechanically determined by `N` server-side (see
// `games/atarigo/src/main.rs`'s `SUPPORTED_SIZES`), so there's nothing else
// for a client to choose.

import type { Component } from "solid-js";
import { For } from "solid-js";

export interface SizeConfig {
  size: number;
}

function isSizeConfig(config: unknown): config is SizeConfig {
  return !!config && typeof config === "object" && typeof (config as { size?: unknown }).size === "number";
}

/** Builds a `NewGameFields` component offering `sizes` as board-size
 * options, defaulting to `defaultSize` when `config` isn't already a
 * `SizeConfig` (e.g. the very first render, before the user has touched the
 * dropdown). */
export function createSizeField(
  sizes: number[],
  defaultSize: number,
): Component<{ config: unknown; onChange: (config: unknown) => void }> {
  return (props) => {
    const size = () => (isSizeConfig(props.config) ? props.config.size : defaultSize);

    return (
      <label>
        Board size
        <select
          value={String(size())}
          onChange={(e) => {
            props.onChange({ size: Number(e.currentTarget.value) } satisfies SizeConfig);
          }}
        >
          <For each={sizes}>
            {(s) => (
              <option value={String(s)}>
                {s} × {s}
              </option>
            )}
          </For>
        </select>
      </label>
    );
  };
}
