// ProgressRail — the "where is this run" strip shown on every run view.
// All projection arithmetic is in `deriveProgress`; this component is layout
// only. The optional `live` prop (the pure event fold from the evidence
// stream) drives the phase label, the cohort chip, and the pair counter
// *between* projection refreshes — the compute ledger stays the authority
// once a refresh lands.

import { Show, type Component } from "solid-js";
import {
  deriveProgress,
  formatWall,
  type ProgressInput,
} from "../models/progress-model.js";
import type { LiveProgress } from "../tuner-types.js";

export const ProgressRail: Component<ProgressInput & { live?: LiveProgress | null }> = (props) => {
  const p = () => deriveProgress(props);
  // The compute ledger has landed once the phase is a real phase name.
  const ledgerReady = (): boolean => !["starting", "—"].includes(p().phase);
  const phaseLabel = (): string =>
    ledgerReady() || !props.live ? p().phase : props.live.phase;
  const budgetTotal = (): number => {
    const compute = props.compute ?? [];
    return compute.reduce((sum, c) => sum + c.pair_attempts, 0);
  };
  const burn = (): number => {
    const compute = props.compute ?? [];
    const done = compute.reduce((sum, c) => sum + c.completed_pairs, 0);
    return done > 0 ? done : (props.live?.pairs.completed ?? 0);
  };

  return (
    <div class="tuner-progress-rail" data-testid="progress-rail">
      <div class="tuner-progress-head">
        <span class="tuner-progress-phase">{phaseLabel()}</span>
        <Show when={props.live && props.live.cohortIndex !== null}>
          <span class="tuner-progress-cohort" data-testid="progress-cohort">
            cohort {props.live!.cohortIndex}
          </span>
        </Show>
        <span class="tuner-progress-wall">{formatWall(p().wallMs)}</span>
      </div>
      <div
        class="tuner-progress-bar"
        role="progressbar"
        aria-valuenow={Math.round(p().fraction * 100)}
      >
        <div
          class="tuner-progress-fill"
          style={{ width: `${Math.min(100, p().fraction * 100)}%` }}
        />
      </div>
      <Show when={ledgerReady()}>
        <div class="tuner-progress-counts">
          {p().pairs.completed} / {p().pairs.attempted} pairs
          {p().pairs.failed > 0 ? ` · ${p().pairs.failed} failed` : ""}
          {p().pairs.censored > 0 ? ` · ${p().pairs.censored} censored` : ""}
        </div>
      </Show>
      <Show when={props.live && !ledgerReady()}>
        <div class="tuner-progress-live-counts" data-testid="progress-live-counts">
          {phaseLabel()}: {props.live!.pairs.completed} pairs done
          {props.live!.pairs.started > props.live!.pairs.completed
            ? ` · ${props.live!.pairs.started - props.live!.pairs.completed} in flight`
            : ""}
          {props.live!.pairs.failed > 0 ? ` · ${props.live!.pairs.failed} failed` : ""}
          {" · "}
          {props.live!.lastEventSeq} evidence lines ingested so far
        </div>
      </Show>
      <Show when={props.live && budgetTotal() > 0}>
        <div class="tuner-progress-burn" data-testid="progress-burn">
          <div
            class="tuner-progress-burn-fill"
            style={{ width: `${Math.min(100, (burn() / budgetTotal()) * 100)}%` }}
          />
          <span class="tuner-progress-burn-label">
            {burn()} / {budgetTotal()} budget
          </span>
        </div>
      </Show>
    </div>
  );
};
