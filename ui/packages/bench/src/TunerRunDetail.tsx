/** TunerRunDetail — Trial history for an open `kind: "tuner"` run.
 *
 * Composes the tuner detail panel from smaller, focused sub-components:
 *   - TunerIncumbentRow — incumbent config display with copy buttons
 *   - TunerSummaryStats — headline stats (trials, best score, CI)
 *   - TunerScoreChart — SVG score-over-trials chart
 *   - TunerConfigDiff — parameter tables (incumbent vs baseline)
 *   - TunerTrialsTable — scrollable per-scored-trial listing
 *
 * Pure presentational component: `trials`/`tuner`/`launchConfig` are read
 * from BenchState by RunDetailPanel.tsx (which owns the tail loop that
 * keeps `trials` current via reducer.ts's `tailTick`) and passed down,
 * same convention as WinRateChart/LeaderboardTable reading their slice of
 * BenchState directly, except this one takes props since it's nested
 * inside another store-reading component rather than mounted as its own
 * tab.
 *
 * Scoring model (optuna):
 *   Each trial plays a matchmaking sequence (ladder of trash) against the
 *   dynamic opponent pool. The primary metric is `mu - 3*sigma` (higher
 *   is better), stored in `trial.extra.mu`/`trial.extra.sigma`, with
 *   `cost = -(mu - 3*sigma)` on the wire. Confidence is expressed as
 *   `mu ± 2*sigma` (the OpenSkill 95% band). */

import { createMemo, Show, type Component } from "solid-js";
import type {
  ChainedTrial,
  ChainRung,
  IncumbentInfo,
  TrialRow,
  TunerInfo,
} from "./index.js";
import {
  buildGroups,
  groupKey,
  trialScore,
} from "./tuner/tuner-helpers.js";
import { TunerIncumbentRow } from "./tuner/TunerIncumbentRow.js";
import { TunerSummaryStats } from "./tuner/TunerSummaryStats.js";
import { TunerScoreChart } from "./tuner/TunerScoreChart.js";
import { TunerConfigDiff } from "./tuner/TunerConfigDiff.js";
import { TunerTrialsTable } from "./tuner/TunerTrialsTable.js";

export const TunerRunDetail: Component<{
  trials: TrialRow[];
  chain: ChainRung[];
  chainedTrials: ChainedTrial[];
  tuner: TunerInfo | null;
  launchConfig?: unknown;
  incumbent?: IncumbentInfo | null;
}> = (props) => {
  const effectiveEntries = createMemo((): ChainedTrial[] => {
    if (props.chainedTrials.length > 0) {
      return [...props.chainedTrials].sort(
        (a, b) => a.rungIndex - b.rungIndex || a.trial.trial_id - b.trial.trial_id,
      );
    }
    return [...props.trials]
      .sort((a, b) => a.trial_id - b.trial_id)
      .map((trial) => ({ rungIndex: 0, trial }));
  });

  const scoredEntries = createMemo(() =>
    effectiveEntries().filter((e) => e.trial.cost !== null),
  );
  const scored = createMemo(() => scoredEntries().map((e) => e.trial));

  // Best trial: highest score (mu - 3σ), tracked as (rungIndex, trial_id)
  // since ids restart at 1 in every rung.
  const bestEntry = createMemo((): ChainedTrial | null => {
    let best: ChainedTrial | null = null;
    for (const e of scoredEntries()) {
      if (best === null) { best = e; continue; }
      const score = trialScore(e.trial);
      const bestScore = trialScore(best.trial);
      if (score !== null && bestScore !== null && score > bestScore) best = e;
    }
    return best;
  });

  const bestTrial = createMemo((): TrialRow | null => bestEntry()?.trial ?? null);

  const groups = createMemo(() => buildGroups(scored()));

  const bestGroup = createMemo(() => {
    const best = bestTrial();
    return best ? groups().get(groupKey(best)) ?? null : null;
  });

  const supportsTranspositions = createMemo(
    () => props.tuner?.parameters.some((p) => p.name === "mcgs") ?? false,
  );

  return (
    <div id="tuner-run-detail">
      <TunerIncumbentRow
        incumbent={props.incumbent ?? null}
        supportsTranspositions={supportsTranspositions()}
      />

      <Show
        when={scored().length > 0}
        fallback={<div class="log-empty">No scored trials yet.</div>}
      >
        <TunerSummaryStats
          totalTrials={effectiveEntries().length}
          bestTrial={bestTrial()}
          bestGroup={bestGroup()}
        />

        <TunerScoreChart
          scoredEntries={scoredEntries()}
          scored={scored()}
          chain={props.chain}
          bestTrial={bestTrial()}
        />

        <TunerConfigDiff
          incumbent={props.incumbent ?? null}
          bestTrial={bestTrial()}
          launchConfig={props.launchConfig}
        />
      </Show>

      <TunerTrialsTable
        entries={effectiveEntries()}
        chain={props.chain}
        bestEntry={bestEntry()}
      />
    </div>
  );
};