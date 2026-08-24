import { For, Show, createMemo, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type { TuningNavigationAction } from "../tuning-navigation.js";
import type { TuningTrialDetail, TuningTrialDetailGame, TuningTrialDetailPair, TuningTrialDetailView, TuningTrialPageQuery, TuningTrialSummary } from "../types.js";
import { opponentDistances, trialPageSummary } from "./analysis-models.js";
import { buildPresetSpec, candidatePresetSource, opponentPresetSource } from "./preset-copy.js";
import { formatRating, formatScore, jsonText } from "./tuning-view-model.js";
import { PresetCopyAction } from "./PresetCopyAction.js";

const PAGE_SIZES = [50, 100, 200] as const;

function send(store: Store<BenchState, BenchAction>, action: TuningNavigationAction): void {
  store.dispatch({ tag: "tuningNavigation", action });
}

function recorded(value: string | number | null, explanation: string): string {
  return value === null ? `Not recorded — ${explanation}` : String(value);
}

function detailState(state: string): string {
  return state.replaceAll("_", " ");
}

const CandidateConfig: Component<{ trial: TuningTrialDetailView }> = (props) => {
  const build = createMemo(() => buildPresetSpec(candidatePresetSource(props.trial)));
  return (
    <section class="tuning-trial-config" aria-label="Candidate configuration">
      <h5>Candidate configuration</h5>
      <Show when={props.trial.config !== null} fallback={<p class="tuning-not-recorded">Not recorded — this legacy trial did not record a candidate configuration.</p>}>
        <pre class="tuning-json">{jsonText(props.trial.config)}</pre>
      </Show>
      <PresetCopyAction label="candidate preset" build={build()} />
    </section>
  );
};

const PolicyReports: Component<{ trial: TuningTrialDetailView }> = (props) => (
  <section class="tuning-trial-reports" aria-label="Ordered policy reports">
    <h5>Policy reports</h5>
    <Show when={props.trial.reports.length > 0} fallback={<p class="tuning-not-recorded">Not recorded — this session has no retained trial reports.</p>}>
      <ol>
        <For each={props.trial.reports}>{(report) => (
          <li>
            After {report.completed_pairs} pairs: {detailState(report.decision.outcome)} / {report.decision.reason}; score {formatScore(report.score)}; rating {formatRating(report.rating.mu, report.rating.sigma)}; bracket {recorded(report.decision.bracket_id, "no bracket was assigned")}; resource {recorded(report.decision.rung_resource, "no resource rung was recorded")}.
          </li>
        )}</For>
      </ol>
    </Show>
  </section>
);

const ReplayLinks: Component<{ game: TuningTrialDetailGame }> = (props) => {
  const replay = () => props.game.replay;
  const href = () => `/api/bench/runs/${encodeURIComponent(replay()!.run_id)}/games/${replay()!.game_seq}/moves`;
  return (
    <span class="tuning-game-links">
      <Show when={replay()} fallback={<span class="tuning-not-recorded">Not recorded — this game has no replay reference.</span>}>
        {(value) => <>
          <Show when={value().has_renderer_trace} fallback={<span class="tuning-not-recorded">Not recorded — renderer trace was not retained.</span>}><a href={href()}>Replay game</a></Show>
          <Show when={value().has_search_reports} fallback={<span class="tuning-not-recorded">Not recorded — search reports were not retained.</span>}><a href={href()}>Search reports</a></Show>
        </>}
      </Show>
    </span>
  );
};

const PairDetail: Component<{ pair: TuningTrialDetailPair; candidate: TuningTrialDetailView }> = (props) => {
  const revision = () => props.pair.pool_revision;
  const anchor = () => props.pair.opponent;
  const copyBuild = createMemo(() => {
    const poolAnchor = revision()?.anchors.find((value) => value.anchor_id === anchor().anchor_id);
    return poolAnchor && revision()
      ? buildPresetSpec(opponentPresetSource(poolAnchor, revision()!))
      : { enabled: false as const, reason: { code: "legacy_missing_config" as const, message: "This opponent's immutable pool snapshot was not recorded." } };
  });
  const distance = createMemo(() => opponentDistances(props.candidate).find((value) => value.pairId === props.pair.pair_id) ?? null);
  return (
    <section class="tuning-trial-pair" aria-label={`Pair ${props.pair.pair_index + 1}`}>
      <h5>Pair {props.pair.pair_index + 1}: {detailState(props.pair.state)}</h5>
      <dl class="tuning-evidence-grid">
        <dt>Opponent snapshot</dt><dd>{anchor().label ?? anchor().anchor_id} ({anchor().anchor_id}) · {formatRating(anchor().mu, anchor().sigma)}</dd>
        <dt>Pool revision</dt><dd>{revision() ? `${revision()!.display_ordinal} · ${revision()!.pool_snapshot_fingerprint}` : "Not recorded — this legacy pair has no immutable pool revision."}</dd>
        <dt>Rating update</dt><dd>{formatRating(props.pair.rating_before.mu, props.pair.rating_before.sigma)} → {props.pair.rating_after ? formatRating(props.pair.rating_after.mu, props.pair.rating_after.sigma) : "Not recorded — no post-pair rating was recorded."}</dd>
        <dt>Opponent distance</dt><dd>{distance() ? `Δμ ${distance()!.deltaMu.toFixed(3)} · |Δμ| ${distance()!.absoluteMuDistance.toFixed(3)}` : "Not recorded — the pair has no comparable rating snapshot."}</dd>
        <dt>Pair score</dt><dd>{formatScore(props.pair.score)}</dd>
      </dl>
      <pre class="tuning-json">{jsonText(anchor().config)}</pre>
      <PresetCopyAction label="opponent preset" build={copyBuild()} />
      <ul class="tuning-pair-games" aria-label={`Games for pair ${props.pair.pair_index + 1}`}>
        <For each={props.pair.games}>{(game, index) => <li>
          Game {index() + 1}: candidate {game.candidate_side}, {game.outcome}, seed {game.seed}, {game.plies} plies, {game.elapsed_ms} ms. <ReplayLinks game={game} />
        </li>}</For>
      </ul>
    </section>
  );
};

const TrialDetail: Component<{ detail: TuningTrialDetail }> = (props) => {
  const trial = () => props.detail.trial;
  return (
    <section class="tuning-trial-detail" aria-label={`Trial ${trial().trial_number} detail`}>
      <dl class="tuning-evidence-grid">
        <dt>State / reason</dt><dd>{detailState(trial().state)} / {recorded(trial().reason, "no terminal reason was recorded")}</dd>
        <dt>Score</dt><dd>{formatScore(trial().score)}</dd>
        <dt>Rating μ ± σ</dt><dd>{trial().rating ? formatRating(trial().rating!.mu, trial().rating!.sigma) : "Not recorded — no terminal rating was recorded."}</dd>
        <dt>Failure</dt><dd>{recorded(trial().failure, "no failure was recorded")}</dd>
      </dl>
      <CandidateConfig trial={trial()} />
      <PolicyReports trial={trial()} />
      <section aria-label="Pair evidence"><h5>Pair evidence</h5><For each={trial().pairs}>{(pair) => <PairDetail pair={pair} candidate={trial()} />}</For></section>
    </section>
  );
};

export const TuningTrialsView: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const navigation = () => state().tuningNavigation;
  const page = () => navigation().trialPage.snapshot;
  const summary = createMemo(() => page() ? trialPageSummary(page()!, navigation().selection.trialId) : null);
  const pageNumber = () => navigation().trialPage.previousCursors.length + 1;
  const expanded = (trialId: string) => navigation().expandedIds.includes(`trial:${trialId}`);
  function setFilter(field: "state" | "bracket" | "reason" | "family" | "q", value: string): void {
    send(props.store, { tag: "setTrialFilters", filters: { [field]: value.trim() || null } });
  }
  function select(row: TuningTrialSummary): void { send(props.store, { tag: "selectTrial", trialId: row.trial_id }); }
  function setSort(sort: NonNullable<TuningTrialPageQuery["sort"]>): void {
    send(props.store, { tag: "setTrialSort", sort: { ...navigation().sort, sort } });
  }
  function setDirection(direction: NonNullable<TuningTrialPageQuery["direction"]>): void {
    send(props.store, { tag: "setTrialSort", sort: { ...navigation().sort, direction } });
  }
  function toggle(row: TuningTrialSummary): void {
    if (!row.has_detail) return;
    const id = `trial:${row.trial_id}`;
    const opening = !expanded(row.trial_id);
    send(props.store, { tag: "toggleExpanded", id });
    if (opening) {
      select(row);
      send(props.store, { tag: "trialDetailRequest", sessionId: navigation().selection.sessionId!, trialId: row.trial_id });
    }
  }
  return (
    <section class="tuning-trials" aria-labelledby="tuning-trials-heading">
      <div class="tuning-trials-heading"><div><h4 id="tuning-trials-heading">Trials</h4><p>{page() ? `${page()!.total_count} results · page ${pageNumber()} · ${summary()!.returnedCount} rendered` : "Loading trial results…"}</p></div></div>
      <fieldset class="tuning-trial-controls" disabled={navigation().trialPage.status === "loading" && page() === null}>
        <legend>Filter and sort trials</legend>
        <label>Search <input aria-label="Search trials" value={navigation().filters.q ?? ""} onInput={(event) => setFilter("q", event.currentTarget.value)} /></label>
        <label>State <input aria-label="Filter state" value={navigation().filters.state ?? ""} onInput={(event) => setFilter("state", event.currentTarget.value)} /></label>
        <label>Bracket <input aria-label="Filter bracket" value={navigation().filters.bracket ?? ""} onInput={(event) => setFilter("bracket", event.currentTarget.value)} /></label>
        <label>Reason <input aria-label="Filter reason" value={navigation().filters.reason ?? ""} onInput={(event) => setFilter("reason", event.currentTarget.value)} /></label>
        <label>Family <input aria-label="Filter family" value={navigation().filters.family ?? ""} onInput={(event) => setFilter("family", event.currentTarget.value)} /></label>
        <label>Sort <select aria-label="Sort trials" value={navigation().sort.sort} onChange={(event) => setSort(event.currentTarget.value as NonNullable<TuningTrialPageQuery["sort"]>)}><For each={["trial", "state", "score", "mu", "sigma", "resource", "family"]}>{(value) => <option value={value}>{value}</option>}</For></select></label>
        <label>Direction <select aria-label="Sort direction" value={navigation().sort.direction} onChange={(event) => setDirection(event.currentTarget.value as NonNullable<TuningTrialPageQuery["direction"]>)}><option value="desc">Descending</option><option value="asc">Ascending</option></select></label>
        <label>Rows per page <select aria-label="Rows per page" value={navigation().trialPageLimit} onChange={(event) => send(props.store, { tag: "setTrialPageLimit", limit: Number(event.currentTarget.value) })}><For each={PAGE_SIZES}>{(value) => <option value={value}>{value}</option>}</For></select></label>
      </fieldset>
      <Show when={navigation().trialPage.status === "error" && !page()}><div class="tuning-load-error" role="alert">Could not load trials: {navigation().trialPage.error}</div></Show>
      <Show when={page()}>{(value) => <>
        <Show when={navigation().trialPage.status === "loading"}><div class="tuning-page-refresh" role="status">Refreshing trial results…</div></Show>
        <Show when={value().trials.length > 0} fallback={<p class="tuning-empty">No trials match these filters.</p>}>
          <div class="tuning-trials-table-wrap"><table class="tuning-trials-table"><thead><tr><th scope="col">State / reason</th><th scope="col">Bracket / resource</th><th scope="col">Score / μ / σ</th><th scope="col">Family</th><th scope="col">Pairs</th><th scope="col">W / L / D</th><th scope="col">Compute</th><th scope="col">Select</th><th scope="col">Expand</th><th scope="col">Copy</th></tr></thead><tbody>
            <For each={summary()!.rows}>{(row) => <>
              <tr classList={{ "tuning-trial-selected": row.selected }} aria-selected={row.selected}>
                <td><strong>#{row.trial_number} {detailState(row.state)}</strong><br /><span>{recorded(row.reason, "no reason recorded")}</span></td>
                <td>{recorded(row.bracket_id, "unassigned")} / {recorded(row.resource, "no resource recorded")}</td>
                <td>{formatScore(row.score)} / {row.rating ? `${row.rating.mu.toFixed(3)} / ${row.rating.sigma.toFixed(3)}` : "Not recorded"}</td>
                <td>{recorded(row.family, "no family recorded")}</td><td>{row.pair_count}</td><td>{row.wld.wins} / {row.wld.losses} / {row.wld.draws}</td>
                <td>{row.compute.elapsedMs} ms · {row.compute.searchIterationsTotal} iter · {row.compute.searchMoveTimeMs} ms/move</td>
                <td><button type="button" aria-label={`Select trial ${row.trial_number}`} onClick={() => select(row)}>Select</button></td>
                <td><button type="button" aria-label={`${expanded(row.trial_id) ? "Collapse" : "Expand"} trial ${row.trial_number}`} aria-expanded={expanded(row.trial_id)} disabled={!row.has_detail} title={row.has_detail ? "" : "Not recorded — detail is unavailable for this trial."} onClick={() => toggle(row)}>{expanded(row.trial_id) ? "Collapse" : "Expand"}</button></td>
                <td>{row.has_detail ? "Available when expanded" : "Not recorded"}</td>
              </tr>
              <Show when={expanded(row.trial_id)}><tr class="tuning-trial-detail-row"><td colSpan={10}><Show when={navigation().trialDetails[row.trial_id]?.status === "loading"}><div role="status">Loading trial detail…</div></Show><Show when={navigation().trialDetails[row.trial_id]?.status === "error"}><div class="tuning-load-error" role="alert">Could not load trial detail: {navigation().trialDetails[row.trial_id]?.error}</div></Show><Show when={navigation().trialDetails[row.trial_id]?.snapshot}>{(detail) => <TrialDetail detail={detail()} />}</Show></td></tr></Show>
            </>}</For>
          </tbody></table></div>
        </Show>
        <footer class="tuning-trial-pagination" aria-label="Trial page navigation"><span>{value().total_count} results · showing {summary()!.returnedCount} on page {pageNumber()} (limit {value().limit})</span><button type="button" onClick={() => send(props.store, { tag: "previousTrialPage" })} disabled={navigation().trialPage.previousCursors.length === 0}>Previous page</button><button type="button" onClick={() => send(props.store, { tag: "nextTrialPage" })} disabled={value().next_cursor === null}>Next page</button></footer>
      </>}</Show>
    </section>
  );
};
