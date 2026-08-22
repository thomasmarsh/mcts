/** TunerSummaryStats — header stats row for a tuner run: total trial count,
 * best score (`mu − 3σ`), best trial number, evaluation count for the best
 * group, and the 95% confidence interval (`mu ± 2σ`). */

import { Show, type Component } from "solid-js";
import type { TrialRow } from "../index.js";
import { trialScore } from "./tuner-helpers.js";
import type { TrialGroup } from "./tuner-helpers.js";

function fmtCiValue(v: number): string {
  return v.toFixed(3);
}

export const TunerSummaryStats: Component<{
  totalTrials: number;
  bestTrial: TrialRow | null;
  bestGroup: TrialGroup | null;
}> = (props) => {
  const bestScoreDisplay = (): string => {
    if (!props.bestTrial) return "—";
    const s = trialScore(props.bestTrial);
    if (s !== null) return s.toFixed(3);
    // Fallback: cost = -(mu - 3σ), so -cost recovers the score
    if (props.bestTrial.cost !== null) return (-props.bestTrial.cost).toFixed(3);
    return "—";
  };

  return (
    <div id="tuner-stats-row">
      <div class="tuner-stat">
        <span class="tuner-stat-value">{props.totalTrials}</span>
        <span class="tuner-stat-label">Trials</span>
      </div>
      <div class="tuner-stat">
        <span class="tuner-stat-value">{bestScoreDisplay()}</span>
        <span class="tuner-stat-label">Best score (mu − 3σ)</span>
      </div>
      <div class="tuner-stat">
        <span class="tuner-stat-value">
          #{props.bestTrial?.trial_id ?? "—"}
        </span>
        <span class="tuner-stat-label">Best trial</span>
      </div>
      <Show when={props.bestGroup}>
        {(group) => (
          <>
            <div class="tuner-stat">
              <span class="tuner-stat-value">{group().trials.length}</span>
              <span class="tuner-stat-label">Evaluations</span>
            </div>
            <div class="tuner-stat">
              <span class="tuner-stat-value">
                {fmtCiValue(group().ci.lower)} – {fmtCiValue(group().ci.upper)}
              </span>
              <span class="tuner-stat-label">95% CI (mu ± 2σ)</span>
            </div>
          </>
        )}
      </Show>
    </div>
  );
};