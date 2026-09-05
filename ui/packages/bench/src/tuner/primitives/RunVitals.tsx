// RunVitals — the run header's "how long, how far, how healthy" panel.
// Layout only; every number is pre-computed and pre-formatted by
// `summarizeVitals` (which composes `deriveVitals`). Shown above the
// science / evidence views so an operator never has to open them to see
// the ETA, the budget burn, or the performance vitals.

import { Show, type Component } from "solid-js";
import { KpiRow } from "./KpiRow.js";
import type { VitalsView } from "../models/telemetry-model.js";

export const RunVitals: Component<{ view: VitalsView }> = (props) => {
  const v = (): VitalsView => props.view;
  return (
    <section class="tuner-run-vitals" data-testid="run-vitals">
      <div class="tuner-run-vitals-head">
        <h3>Vitals</h3>
        <Show when={v().etaLabel} fallback={<span class="tuner-vitals-eta tuner-vitals-eta--none">ETA —</span>}>
          <span class="tuner-vitals-eta" data-testid="run-vitals-eta">
            ETA {v().etaLabel}
          </span>
        </Show>
      </div>

      <Show when={v().progressFraction !== null}>
        <div
          class="tuner-vitals-bar"
          role="progressbar"
          data-testid="run-vitals-bar"
          aria-valuenow={Math.round(v().progressFraction! * 100)}
        >
          <div
            class="tuner-vitals-fill"
            style={{ width: `${v().progressFraction! * 100}%` }}
          />
        </div>
      </Show>

      <KpiRow items={v().kpis} testid="run-vitals-kpis" />
    </section>
  );
};
