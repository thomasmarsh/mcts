// NewGameFields.tsx — Akron's new-game config editor: `@mcts/pyramid`'s
// shared board-size picker, bound to Akron's own `MIN_N`/`MAX_N`/`DEFAULT_N`
// range (identical to Margo's).

import type { Component } from "solid-js";
import { BoardSizeFields, boardSizes } from "@mcts/pyramid";
import { DEFAULT_N, MAX_N, MIN_N } from "./types.js";

const SIZES = boardSizes(MIN_N, MAX_N);

export const NewGameFields: Component<{ config: unknown; onChange: (config: unknown) => void }> = (
  props,
) => (
  <BoardSizeFields
    config={props.config}
    onChange={props.onChange}
    sizes={SIZES}
    defaultN={DEFAULT_N}
  />
);
