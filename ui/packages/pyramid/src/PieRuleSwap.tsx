// PieRuleSwap.tsx — The pie-rule "swap colours" HUD panel, shared by every
// pyramid-family game whose `Action` includes a `Swap` variant (Margo,
// Akron): a fixed overlay offering the one-off recolour reply to the
// opening placement. Purely presentational -- the caller decides whether
// `Swap` is currently legal (via `canSwap`) and supplies the action itself
// via `onSwap`.

import type { Component } from "solid-js";
import "./pyramid.css";

export const PieRuleSwap: Component<{ canSwap: boolean; busy: boolean; onSwap: () => void }> = (
  props,
) => (
  <>
    {props.canSwap && (
      <div class="pyramid-swap-panel">
        <span class="pyramid-swap-title">Pie rule</span>
        <button
          type="button"
          class="pyramid-swap-button"
          disabled={props.busy}
          onClick={() => props.onSwap()}
        >
          Swap colours
        </button>
      </div>
    )}
  </>
);
