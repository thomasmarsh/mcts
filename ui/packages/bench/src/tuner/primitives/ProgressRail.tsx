// ProgressRail — the "where is this run" strip shown on every run view.
// All arithmetic is in `deriveProgress`; this component is layout only.

import { type Component } from "solid-js";
import {
  deriveProgress,
  formatWall,
  type ProgressInput,
} from "../models/progress-model.js";

export const ProgressRail: Component<ProgressInput> = (props) => {
  const p = () => deriveProgress(props);
  return (
    <div class="tuner-progress-rail" data-testid="progress-rail">
      <div class="tuner-progress-head">
        <span class="tuner-progress-phase">{p().phase}</span>
        <span class="tuner-progress-wall">{formatWall(p().wallMs)}</span>
      </div>
      <div class="tuner-progress-bar" role="progressbar" aria-valuenow={Math.round(p().fraction * 100)}>
        <div class="tuner-progress-fill" style={{ width: `${Math.min(100, p().fraction * 100)}%` }} />
      </div>
      <div class="tuner-progress-counts">
        {p().pairs.completed} / {p().pairs.attempted} pairs
        {p().pairs.failed > 0 ? ` · ${p().pairs.failed} failed` : ""}
        {p().pairs.censored > 0 ? ` · ${p().pairs.censored} censored` : ""}
      </div>
    </div>
  );
};
