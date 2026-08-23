import { createMemo, Show, type Component } from "solid-js";
import { Dynamic } from "solid-js/web";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type {
  BenchSpectatorProps,
  TuningAttempt,
  TuningGame,
  TuningPair,
  TuningSessionDetail,
  TuningSessionListItem,
  TuningTrial,
} from "../types.js";
import {
  formatRating,
  formatScore,
  formatTimestamp,
  jsonText,
  opponentLabel,
  pairEvidence,
  replayTarget,
  selectedAttempt,
  selectedGame,
  selectedPair,
  selectedTrial,
} from "./tuning-view-model.js";

const Field: Component<{ label: string; value: string | number | null }> = (props) => (
  <><dt>{props.label}</dt><dd>{props.value ?? "not recorded"}</dd></>
);

const SessionEvidence: Component<{ detail: TuningSessionDetail }> = (props) => (
  <>
    <h4>Session evidence</h4>
    <dl class="tuning-evidence-grid">
      <Field label="Status" value={props.detail.summary.status} />
      <Field label="Session ID" value={props.detail.summary.session_id} />
      <Field label="Target trials" value={props.detail.summary.target_trial_count} />
      <Field label="Manifest fingerprint" value={props.detail.fingerprint} />
      <Field label="Lifecycle sequence" value={props.detail.cursor.session_sequence} />
    </dl>
  </>
);

const AttemptEvidence: Component<{ attempt: TuningAttempt }> = (props) => (
  <>
    <h4>Attempt evidence</h4>
    <dl class="tuning-evidence-grid">
      <Field label="Status" value={props.attempt.status} />
      <Field label="Attempt ID" value={props.attempt.attempt_id} />
      <Field label="Physical run" value={props.attempt.bench_run_id} />
      <Field label="Started" value={formatTimestamp(props.attempt.started_at)} />
      <Field label="Ended" value={props.attempt.ended_at ? formatTimestamp(props.attempt.ended_at) : null} />
      <Field label="Failure" value={props.attempt.failure} />
    </dl>
  </>
);

const TrialEvidence: Component<{ trial: TuningTrial }> = (props) => (
  <>
    <h4>Trial #{props.trial.trial_number}</h4>
    <dl class="tuning-evidence-grid">
      <Field label="Status" value={props.trial.status} />
      <Field label="Trial ID" value={props.trial.trial_id} />
      <Field label="Score" value={formatScore(props.trial.score)} />
      <Field label="Rating μ ± σ" value={formatRating(props.trial.mu, props.trial.sigma)} />
      <Field label="Failure" value={props.trial.failure} />
    </dl>
    <h4>Candidate configuration</h4>
    <pre class="tuning-json">{jsonText(props.trial.config)}</pre>
  </>
);

const PairEvidence: Component<{ pair: TuningPair }> = (props) => (
  <>
    <h4>Pair {props.pair.pair_index + 1}</h4>
    <dl class="tuning-evidence-grid">
      <Field label="Status" value={pairEvidence(props.pair)} />
      <Field label="Opponent" value={opponentLabel(props.pair)} />
      <Field label="Opponent rating" value={formatRating(props.pair.opponent.mu, props.pair.opponent.sigma)} />
      <Field label="Rating before" value={formatRating(props.pair.rating_before.mu, props.pair.rating_before.sigma)} />
      <Field label="Rating after" value={props.pair.rating_after ? formatRating(props.pair.rating_after.mu, props.pair.rating_after.sigma) : null} />
      <Field label="Score" value={formatScore(props.pair.score)} />
      <Field label="Seed / round" value={`${props.pair.seed} / ${props.pair.round}`} />
      <Field label="Failure" value={props.pair.failure} />
    </dl>
    <h4>Opponent configuration</h4>
    <pre class="tuning-json">{jsonText(props.pair.opponent.config)}</pre>
  </>
);

const GameEvidence: Component<{ game: TuningGame }> = (props) => (
  <>
    <h4>Game evidence</h4>
    <dl class="tuning-evidence-grid">
      <Field label="Outcome" value={props.game.outcome} />
      <Field label="Candidate seat" value={props.game.candidate_side} />
      <Field label="Seed / round" value={`${props.game.seed} / ${props.game.round}`} />
      <Field label="Plies" value={props.game.plies} />
      <Field label="Elapsed" value={`${props.game.elapsed_ms} ms`} />
      <Field label="Trace sequence" value={props.game.trace_game_seq} />
      <Field label="Candidate search" value={`${props.game.candidate.iterations_total} iterations · ${props.game.candidate.move_time_ms} ms`} />
      <Field label="Candidate first half" value={`${props.game.candidate.iterations_first_half} iterations`} />
      <Field label="Opponent search" value={`${props.game.baseline.iterations_total} iterations · ${props.game.baseline.move_time_ms} ms`} />
      <Field label="Opponent first half" value={`${props.game.baseline.iterations_first_half} iterations`} />
    </dl>
  </>
);

const Replay: Component<{
  detail: TuningSessionDetail;
  session: TuningSessionListItem | null;
  store: Store<BenchState, BenchAction>;
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const target = createMemo(() => replayTarget(props.detail, props.session, props.store.getState()().tuningNavigation.selection));
  const available = () => {
    const value = target();
    return typeof value === "string" ? null : value;
  };
  return (
    <section class="tuning-replay" aria-label="Selected game replay">
      <Show when={props.Spectator} fallback={<><button disabled>Replay unavailable</button><div class="tuning-replay-reason">No spectator component is available.</div></>}>
        {(Spectator) => (
          <Show when={available()} fallback={<><button disabled>Replay unavailable</button><div class="tuning-replay-reason">{String(target())}</div></>}>
            {(value) => <Dynamic component={Spectator()} runId={value().runId} game={value().game} kind="tuner" live={value().live} initialGameSeq={value().gameSeq} />}
          </Show>
        )}
      </Show>
    </section>
  );
};

export const TuningEvidenceDetail: Component<{
  detail: TuningSessionDetail;
  session: TuningSessionListItem | null;
  store: Store<BenchState, BenchAction>;
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const selection = () => props.store.getState()().tuningNavigation.selection;
  const attempt = createMemo(() => selectedAttempt(props.detail, selection()));
  const trial = createMemo(() => selectedTrial(props.detail, selection()));
  const pair = createMemo(() => selectedPair(props.detail, selection()));
  const game = createMemo(() => selectedGame(props.detail, selection()));
  return (
    <section class="tuning-panel" aria-live="polite">
      <Show when={game()} fallback={<Show when={pair()} fallback={<Show when={trial()} fallback={<Show when={attempt()} fallback={<SessionEvidence detail={props.detail} />}>{(value) => <AttemptEvidence attempt={value()} />}</Show>}>{(value) => <TrialEvidence trial={value()} />}</Show>}>{(value) => <PairEvidence pair={value()} />}</Show>}>
        {(value) => <GameEvidence game={value()} />}
      </Show>
      <Replay detail={props.detail} session={props.session} store={props.store} Spectator={props.Spectator} />
    </section>
  );
};
