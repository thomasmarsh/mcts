import { createMemo, For, Show, type Component } from "solid-js";
import { Dynamic } from "solid-js/web";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type {
  BenchSpectatorProps,
  TuningSessionListItem,
  TuningTrialDetailGame,
} from "../types.js";
import type { TuningNavigationAction } from "../tuning-navigation.js";
import { formatRating, formatScore } from "./tuning-view-model.js";

function send(store: Store<BenchState, BenchAction>, action: TuningNavigationAction): void {
  store.dispatch({ tag: "tuningNavigation", action });
}

const GameReplay: Component<{
  game: TuningTrialDetailGame | null;
  session: TuningSessionListItem | null;
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const replay = () => props.game?.replay ?? null;
  const unavailable = () => {
    if (!props.game) return "Select a recorded game to inspect its replay.";
    if (!replay()) return "Replay unavailable: this game has no replay reference.";
    if (!replay()!.has_renderer_trace)
      return "Replay unavailable: renderer trace was not retained.";
    if (!props.session?.game)
      return "Replay unavailable: the session did not record its game kind.";
    return null;
  };
  return (
    <section class="tuning-replay" aria-label="Selected game replay">
      <Show
        when={props.Spectator}
        fallback={
          <>
            <button disabled>Replay unavailable</button>
            <div class="tuning-replay-reason">No spectator component is available.</div>
          </>
        }
      >
        {(Spectator) => (
          <Show
            when={!unavailable()}
            fallback={
              <>
                <button disabled>Replay unavailable</button>
                <div class="tuning-replay-reason">{unavailable()}</div>
              </>
            }
          >
            <Dynamic
              component={Spectator()}
              runId={replay()!.run_id}
              game={props.session!.game!}
              kind="tuner"
              live={props.session!.status === "active"}
              initialGameSeq={replay()!.game_seq}
            />
          </Show>
        )}
      </Show>
    </section>
  );
};

export const TuningGameEvidence: Component<{
  store: Store<BenchState, BenchAction>;
  session: TuningSessionListItem | null;
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const state = props.store.getState();
  const navigation = () => state().tuningNavigation;
  const page = () => navigation().trialPage.snapshot;
  const selectedDetail = () => {
    const trialId = navigation().selection.trialId;
    return trialId ? (navigation().trialDetails[trialId]?.snapshot?.trial ?? null) : null;
  };
  const selectedPair = createMemo(
    () =>
      selectedDetail()?.pairs.find((pair) => pair.pair_id === navigation().selection.pairId) ??
      null,
  );
  const selectedGame = createMemo(
    () =>
      selectedPair()?.games.find((game) => game.game_id === navigation().selection.gameId) ?? null,
  );
  const trialState = () =>
    navigation().selection.trialId
      ? navigation().trialDetails[navigation().selection.trialId!]
      : null;
  return (
    <section class="tuning-game-evidence" aria-labelledby="tuning-game-heading">
      <header class="tuning-trials-heading">
        <div>
          <h4 id="tuning-game-heading">Game</h4>
          <p>Choose one retained trial; its pairs and games load on demand.</p>
        </div>
      </header>
      <Show
        when={page()}
        fallback={
          <div class="loading-bench" role="status">
            Loading recorded trials…
          </div>
        }
      >
        {(value) => (
          <section class="tuning-game-trials" aria-label="Recorded trials">
            <Show
              when={value().trials.length > 0}
              fallback={
                <p class="tuning-empty">No recorded trials are available for game evidence.</p>
              }
            >
              <For each={value().trials}>
                {(trial) => (
                  <button
                    type="button"
                    class="tuning-game-trial"
                    classList={{
                      "tuning-game-trial-selected":
                        navigation().selection.trialId === trial.trial_id,
                    }}
                    aria-pressed={navigation().selection.trialId === trial.trial_id}
                    disabled={!trial.has_detail}
                    title={
                      trial.has_detail ? "" : "Not recorded — detail is unavailable for this trial."
                    }
                    onClick={() =>
                      send(props.store, { tag: "selectTrial", trialId: trial.trial_id })
                    }
                  >
                    Trial #{trial.trial_number} · {trial.state} · {trial.pair_count} pairs ·{" "}
                    {formatScore(trial.score)}
                  </button>
                )}
              </For>
            </Show>
            <Show when={value().next_cursor !== null}>
              <p class="tuning-not-recorded">More trials are available in the Trials tab.</p>
            </Show>
          </section>
        )}
      </Show>
      <Show when={navigation().trialPage.status === "error" && !page()}>
        <div class="tuning-load-error" role="alert">
          Could not load recorded trials: {navigation().trialPage.error}
        </div>
      </Show>
      <Show when={trialState()?.status === "loading"}>
        <div role="status">Loading recorded game evidence…</div>
      </Show>
      <Show when={trialState()?.status === "error"}>
        <div class="tuning-load-error" role="alert">
          Could not load recorded game evidence: {trialState()?.error}
        </div>
      </Show>
      <Show when={selectedDetail()}>
        {(trial) => (
          <section
            class="tuning-game-detail"
            aria-label={`Trial ${trial().trial_number} game evidence`}
          >
            <dl class="tuning-evidence-grid">
              <dt>Trial</dt>
              <dd>
                #{trial().trial_number} · {trial().state}
              </dd>
              <dt>Score / rating</dt>
              <dd>
                {formatScore(trial().score)} /{" "}
                {trial().rating
                  ? formatRating(trial().rating!.mu, trial().rating!.sigma)
                  : "Not recorded"}
              </dd>
              <dt>Reason</dt>
              <dd>{trial().reason ?? "Not recorded"}</dd>
            </dl>
            <div class="tuning-game-pairs" role="list" aria-label="Recorded pairs">
              <For each={trial().pairs}>
                {(pair) => (
                  <section role="listitem" class="tuning-game-pair">
                    <button
                      type="button"
                      onClick={() => send(props.store, { tag: "selectPair", pairId: pair.pair_id })}
                      aria-pressed={navigation().selection.pairId === pair.pair_id}
                    >
                      Pair {pair.pair_index + 1} · {pair.state} · {pair.games.length} games
                    </button>
                    <Show when={navigation().selection.pairId === pair.pair_id}>
                      <div class="tuning-game-list">
                        <For each={pair.games}>
                          {(game, index) => (
                            <button
                              type="button"
                              onClick={() =>
                                send(props.store, { tag: "selectGame", gameId: game.game_id })
                              }
                              aria-pressed={navigation().selection.gameId === game.game_id}
                            >
                              Game {index() + 1} · candidate {game.candidate_side} · {game.outcome}{" "}
                              · {game.plies} plies
                            </button>
                          )}
                        </For>
                      </div>
                    </Show>
                  </section>
                )}
              </For>
            </div>
          </section>
        )}
      </Show>
      <GameReplay game={selectedGame()} session={props.session} Spectator={props.Spectator} />
    </section>
  );
};
