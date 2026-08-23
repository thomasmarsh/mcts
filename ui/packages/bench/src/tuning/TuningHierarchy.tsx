import { Index, Show, type Accessor, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type { TuningNavigationAction } from "../tuning-navigation.js";
import type { TuningAttempt, TuningGame, TuningPair, TuningSessionDetail, TuningTrial } from "../types.js";
import {
  configurationSummary,
  formatRating,
  formatScore,
  opponentLabel,
  pairEvidence,
  trialsForAttempt,
} from "./tuning-view-model.js";

interface TreeProps {
  store: Store<BenchState, BenchAction>;
  detail: TuningSessionDetail;
}

interface NodeProps extends TreeProps {
  expanded: (id: string) => boolean;
}

function send(store: Store<BenchState, BenchAction>, action: TuningNavigationAction): void {
  store.dispatch({ tag: "tuningNavigation", action });
}

function trialSummary(trial: TuningTrial): string {
  const values = [trial.status, configurationSummary(trial.config), `score ${formatScore(trial.score)}`];
  if (trial.mu !== null && trial.sigma !== null) values.push(`rating ${formatRating(trial.mu, trial.sigma)}`);
  if (trial.failure) values.push(`failure: ${trial.failure}`);
  return values.join(" · ");
}

function pairSummary(pair: TuningPair): string {
  const values = [pairEvidence(pair), `before ${formatRating(pair.rating_before.mu, pair.rating_before.sigma)}`];
  if (pair.rating_after) values.push(`after ${formatRating(pair.rating_after.mu, pair.rating_after.sigma)}`);
  if (pair.score !== null) values.push(`score ${formatScore(pair.score)}`);
  return values.join(" · ");
}

function gameSummary(game: TuningGame): string {
  const trace = game.trace_game_seq === null ? "trace unavailable" : `trace #${game.trace_game_seq}`;
  return `${game.outcome} · seed ${game.seed} · ${game.plies} plies · ${game.elapsed_ms} ms · candidate ${game.candidate.iterations_total} iter/${game.candidate.move_time_ms} ms · opponent ${game.baseline.iterations_total} iter/${game.baseline.move_time_ms} ms · ${trace}`;
}

const Toggle: Component<{ store: Store<BenchState, BenchAction>; id: string; expanded: boolean; label: string }> = (props) => (
  <button
    class="tuning-node-toggle"
    aria-label={`${props.expanded ? "Collapse" : "Expand"} ${props.label}`}
    onClick={() => send(props.store, { tag: "toggleExpanded", id: props.id })}
  >
    {props.expanded ? "−" : "+"}
  </button>
);

const GameNode: Component<NodeProps & { game: Accessor<TuningGame> }> = (props) => {
  const selected = () => props.store.getState()().tuningNavigation.selection.gameId === props.game().game_id;
  return (
    <li class="tuning-tree-node" role="treeitem" aria-selected={selected()}>
      <div class="tuning-node-controls">
        <button class="tuning-node-select" onClick={() => send(props.store, { tag: "selectGame", gameId: props.game().game_id })}>
          Game · candidate {props.game().candidate_side}
          <span class="tuning-node-status">{gameSummary(props.game())}</span>
        </button>
      </div>
    </li>
  );
};

const PairNode: Component<NodeProps & { pair: Accessor<TuningPair> }> = (props) => {
  const id = () => `pair:${props.pair().pair_id}`;
  const selected = () => props.store.getState()().tuningNavigation.selection.pairId === props.pair().pair_id;
  return (
    <li class="tuning-tree-node" role="treeitem" aria-selected={selected()} aria-expanded={props.expanded(id())}>
      <div class="tuning-node-controls">
        <Toggle store={props.store} id={id()} expanded={props.expanded(id())} label={`pair ${props.pair().pair_index + 1}`} />
        <button class="tuning-node-select" onClick={() => send(props.store, { tag: "selectPair", pairId: props.pair().pair_id })}>
          Pair {props.pair().pair_index + 1} · {opponentLabel(props.pair())}
          <span class="tuning-node-status">{pairSummary(props.pair())}</span>
        </button>
      </div>
      <Show when={props.expanded(id())}>
        <ul role="group">
          <Index each={props.pair().games}>{(game) => <GameNode {...props} game={game} />}</Index>
        </ul>
      </Show>
    </li>
  );
};

const TrialNode: Component<NodeProps & { trial: Accessor<TuningTrial> }> = (props) => {
  const id = () => `trial:${props.trial().trial_id}`;
  const selected = () => props.store.getState()().tuningNavigation.selection.trialId === props.trial().trial_id;
  return (
    <li class="tuning-tree-node" role="treeitem" aria-selected={selected()} aria-expanded={props.expanded(id())}>
      <div class="tuning-node-controls">
        <Toggle store={props.store} id={id()} expanded={props.expanded(id())} label={`trial ${props.trial().trial_number}`} />
        <button class="tuning-node-select" onClick={() => send(props.store, { tag: "selectTrial", trialId: props.trial().trial_id })}>
          Trial #{props.trial().trial_number}
          <span class="tuning-node-status">{trialSummary(props.trial())} · {props.trial().pairs.length} pairs</span>
        </button>
      </div>
      <Show when={props.expanded(id())}>
        <ul role="group">
          <Index each={props.trial().pairs}>{(pair) => <PairNode {...props} pair={pair} />}</Index>
        </ul>
      </Show>
    </li>
  );
};

const AttemptNode: Component<NodeProps & { attempt: Accessor<TuningAttempt> }> = (props) => {
  const id = () => `attempt:${props.attempt().attempt_id}`;
  const trials = () => trialsForAttempt(props.detail, props.attempt().attempt_id);
  const selected = () => props.store.getState()().tuningNavigation.selection.attemptId === props.attempt().attempt_id;
  return (
    <li class="tuning-tree-node" role="treeitem" aria-selected={selected()} aria-expanded={props.expanded(id())}>
      <div class="tuning-node-controls">
        <Toggle store={props.store} id={id()} expanded={props.expanded(id())} label={`attempt ${props.attempt().attempt_id}`} />
        <button class="tuning-node-select" onClick={() => send(props.store, { tag: "selectAttempt", attemptId: props.attempt().attempt_id })}>
          Attempt {props.attempt().attempt_id.slice(0, 12)}
          <span class="tuning-node-status">{props.attempt().status} · {trials().length} trials</span>
        </button>
      </div>
      <Show when={props.expanded(id())}>
        <ul role="group">
          <Index each={trials()}>{(trial) => <TrialNode {...props} trial={trial} />}</Index>
        </ul>
      </Show>
    </li>
  );
};

export const TuningHierarchy: Component<TreeProps> = (props) => {
  const expanded = (id: string): boolean => props.store.getState()().tuningNavigation.expandedIds.includes(id);
  return (
    <section class="tuning-panel" aria-labelledby="tuning-hierarchy-heading">
      <h4 id="tuning-hierarchy-heading">Attempts and evidence</h4>
      <ul class="tuning-tree" role="tree" aria-label="Tuning evidence hierarchy">
        <Index each={props.detail.attempts}>{(attempt) => <AttemptNode {...props} expanded={expanded} attempt={attempt} />}</Index>
      </ul>
    </section>
  );
};
