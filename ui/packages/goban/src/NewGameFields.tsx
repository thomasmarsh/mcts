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

/** Builds a `NewGameFields` component for a contiguous size range (e.g.
 * every N from `min` to `max`), rendered as a numeric input rather than a
 * `<select>` -- for games like Gonnect/AtariGo where every size in the
 * range is legal, so a dropdown would mean one `<option>` per size. Values
 * are clamped to `[min, max]` on change; values outside that range (or
 * non-numeric input, e.g. while the field is mid-edit) fall back to the
 * current size rather than propagating an invalid config. */
export function createSizeRangeField(
  min: number,
  max: number,
  defaultSize: number,
): Component<{ config: unknown; onChange: (config: unknown) => void }> {
  return (props) => {
    const size = () => (isSizeConfig(props.config) ? props.config.size : defaultSize);

    return (
      <label>
        Board size ({min}–{max})
        <input
          type="number"
          min={min}
          max={max}
          step={1}
          value={size()}
          onChange={(e) => {
            const n = Number(e.currentTarget.value);
            if (!Number.isInteger(n)) return;
            const clamped = Math.min(max, Math.max(min, n));
            props.onChange({ size: clamped } satisfies SizeConfig);
          }}
        />
      </label>
    );
  };
}
