import { createMemo, createUniqueId, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type { TuningNavigationAction } from "../tuning-navigation.js";
import { candidateRatingTrajectory, ladderAnchorRows, ladderMuDomain, opponentDistances, poolRevisionCoverage, type LadderAnchorRow, type OpponentDistance } from "./analysis-models.js";
import { PresetCopyAction } from "./PresetCopyAction.js";
import { buildPresetSpec, opponentPresetSource } from "./preset-copy.js";
import { jsonText } from "./tuning-view-model.js";

const WIDTH = 680;
const LEFT = 130;
const RIGHT = 20;
const TOP = 34;
const BOTTOM = 56;

function send(store: Store<BenchState, BenchAction>, action: TuningNavigationAction): void {
  store.dispatch({ tag: "tuningNavigation", action });
}

function numberText(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function rowLabel(row: LadderAnchorRow): string {
  return `Select anchor ${row.anchorId}, revision ${row.revisionOrdinal}, μ ${numberText(row.mu)}, σ ${numberText(row.sigma)}, ${row.provenance || "provenance not recorded"}, ${row.insertionReason || "insertion reason not recorded"}.`;
}

function poolJoinText(revision: { display_ordinal: number; pool_snapshot_fingerprint: string } | null | undefined): string {
  return revision ? `${revision.display_ordinal} · ${revision.pool_snapshot_fingerprint}` : "Not recorded — immutable pool revision was not retained.";
}

const LadderMap: Component<{
  anchors: LadderAnchorRow[];
  allRevisionOrdinals: number[];
  candidate: ReturnType<typeof candidateRatingTrajectory>;
  opponents: OpponentDistance[];
  selectedAnchorKey: string | null;
  onSelect: (key: string) => void;
}> = (props) => {
  const id = createUniqueId();
  const height = createMemo(() => Math.max(240, TOP + BOTTOM + (props.allRevisionOrdinals.length + 1) * 52));
  const domain = createMemo(() => ladderMuDomain(props.anchors, props.candidate, props.opponents));
  const x = (mu: number) => LEFT + ((mu - domain()[0]) / (domain()[1] - domain()[0])) * (WIDTH - LEFT - RIGHT);
  const y = (revision: number) => TOP + Math.max(0, props.allRevisionOrdinals.indexOf(revision)) * 52 + 18;
  const candidateY = (resource: number) => {
    const resources = [...new Set(props.candidate.map((point) => point.resource))].sort((a, b) => a - b);
    if (resources.length < 2) return height() - BOTTOM + 12;
    return height() - BOTTOM + ((resource - resources[0]!) / (resources.at(-1)! - resources[0]!)) * 24;
  };
  const candidatePath = createMemo(() => props.candidate.length < 2 ? "" : props.candidate.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.mu)},${candidateY(point.resource)}`).join(" "));
  const keydown = (event: KeyboardEvent, key: string) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); props.onSelect(key); }
  };
  return (
    <svg class="tuning-ladder-map" viewBox={`0 0 ${WIDTH} ${height()}`} role="img" aria-labelledby={`${id}-title ${id}-description`}>
      <title id={`${id}-title`}>Immutable opponent-pool rating map</title>
      <desc id={`${id}-description`}>Anchors are positioned by their recorded session-local μ. Horizontal lines show μ plus or minus twice their recorded σ. Marker shapes identify provenance; the lower lane shows the selected candidate's recorded rating reports.</desc>
      <line class="tuning-progress-axis" x1={LEFT} y1={TOP - 12} x2={LEFT} y2={height() - BOTTOM + 30} />
      <For each={props.allRevisionOrdinals}>{(revision) => <><line class="tuning-ladder-revision-line" x1={LEFT} y1={y(revision)} x2={WIDTH - RIGHT} y2={y(revision)} /><text class="tuning-ladder-revision-label" x={LEFT - 8} y={y(revision) + 4} text-anchor="end">revision {revision}</text></>}</For>
      <text class="tuning-progress-y-label" x={LEFT} y={TOP - 18}>{numberText(domain()[0])} μ</text>
      <text class="tuning-progress-y-label" x={WIDTH - RIGHT} y={TOP - 18} text-anchor="end">{numberText(domain()[1])} μ</text>
      <For each={props.anchors}>{(anchor) => <g classList={{ "tuning-ladder-anchor": true, "tuning-ladder-selected-anchor": anchor.key === props.selectedAnchorKey }} data-testid="ladder-anchor" data-anchor-key={anchor.key} role="button" tabindex="0" aria-label={rowLabel(anchor)} onClick={() => props.onSelect(anchor.key)} onKeyDown={(event) => keydown(event, anchor.key)}>
        <title>{rowLabel(anchor)}</title>
        <line class="tuning-ladder-interval" x1={x(anchor.lower)} y1={y(anchor.revisionOrdinal)} x2={x(anchor.upper)} y2={y(anchor.revisionOrdinal)} />
        <Show when={anchor.provenance === "candidate"} fallback={<Show when={anchor.provenance === "baseline"} fallback={<circle cx={x(anchor.mu)} cy={y(anchor.revisionOrdinal)} r="5" />}><rect x={x(anchor.mu) - 5} y={y(anchor.revisionOrdinal) - 5} width="10" height="10" /></Show>}><polygon points={`${x(anchor.mu)},${y(anchor.revisionOrdinal) - 6} ${x(anchor.mu) + 6},${y(anchor.revisionOrdinal)} ${x(anchor.mu)},${y(anchor.revisionOrdinal) + 6} ${x(anchor.mu) - 6},${y(anchor.revisionOrdinal)}`} /></Show>
        <text class="tuning-ladder-anchor-label" x={x(anchor.mu)} y={y(anchor.revisionOrdinal) - 9} text-anchor="middle">{anchor.anchorId}</text>
      </g>}</For>
      <For each={props.opponents}>{(opponent) => <g class="tuning-ladder-opponent" aria-label={`Encountered opponent ${opponent.opponentId}, delta mu ${numberText(opponent.deltaMu)}`}><line x1={x(opponent.opponentMu) - 4} y1={height() - BOTTOM + 8} x2={x(opponent.opponentMu) + 4} y2={height() - BOTTOM + 16} /><line x1={x(opponent.opponentMu) - 4} y1={height() - BOTTOM + 16} x2={x(opponent.opponentMu) + 4} y2={height() - BOTTOM + 8} /></g>}</For>
      <Show when={candidatePath()}><path class="tuning-ladder-candidate-path" d={candidatePath()} /></Show>
      <For each={props.candidate}>{(point) => <circle class="tuning-ladder-candidate-point" cx={x(point.mu)} cy={candidateY(point.resource)} r="4" />}</For>
      <text class="tuning-ladder-candidate-label" x={LEFT - 8} y={height() - BOTTOM + 15} text-anchor="end">selected candidate</text>
    </svg>
  );
};

export const TuningLadderView: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const navigation = () => state().tuningNavigation;
  const overview = () => navigation().overview.snapshot;
  const coverage = createMemo(() => overview() ? poolRevisionCoverage(overview()!) : null);
  const anchors = createMemo(() => overview() ? ladderAnchorRows(overview()!, navigation().ladderRevision, navigation().ladderAnchorKey) : []);
  const selectedAnchor = createMemo(() => anchors().find((anchor) => anchor.key === navigation().ladderAnchorKey) ?? null);
  const selectedDetail = createMemo(() => {
    const trialId = navigation().selection.trialId;
    return trialId ? navigation().trialDetails[trialId]?.snapshot?.trial ?? null : null;
  });
  const candidate = createMemo(() => candidateRatingTrajectory(selectedDetail()));
  const opponents = createMemo(() => selectedDetail() ? opponentDistances(selectedDetail()!) : []);
  const snapshot = createMemo(() => {
    const selected = selectedAnchor();
    if (!selected || !overview()) return null;
    const revision = overview()!.pool_revisions.find((row) => row.pool_snapshot_fingerprint === selected.revisionFingerprint);
    const anchor = revision?.anchors.find((row) => row.anchor_id === selected.anchorId);
    return revision && anchor ? { revision, anchor } : null;
  });
  const selectAnchor = (anchorKey: string) => send(props.store, { tag: "selectLadderAnchor", anchorKey });
  return (
    <section class="tuning-ladder" aria-labelledby="tuning-ladder-heading">
      <header class="tuning-trials-heading"><div><h4 id="tuning-ladder-heading">Ladder</h4><p>Immutable opponent-pool snapshots and session-local rating context.</p></div></header>
      <Show when={navigation().overview.status === "error" && !overview()}><div class="tuning-load-error" role="alert">Could not load ladder evidence: {navigation().overview.error}</div></Show>
      <Show when={overview()} fallback={<p class="tuning-empty">Loading ladder evidence…</p>}>{(value) => <div data-cursor={value().cursor.session_sequence}>
        <Show when={navigation().overview.status === "loading"}><p class="tuning-page-refresh" role="status">Refreshing ladder evidence…</p></Show>
        <p class="tuning-ladder-session">Ratings are session-local. Session fingerprint: <strong>{navigation().detail.snapshot?.fingerprint ?? "Not recorded"}</strong>.</p>
        <Show when={coverage()!.revisionCount > 0} fallback={<section class="tuning-progress-legacy" aria-label="Missing immutable pool evidence"><p>Not recorded — this session has no immutable pool revisions. The current or newest pool is not substituted for legacy evidence.</p></section>}>
          <fieldset class="tuning-ladder-controls"><legend>Pool revision</legend><label>Revision <select aria-label="Ladder revision" value={navigation().ladderRevision ?? ""} onChange={(event) => send(props.store, { tag: "setLadderRevision", revision: event.currentTarget.value === "" ? null : Number(event.currentTarget.value) })}><option value="">All stored revisions</option><For each={coverage()!.revisions}>{(revision) => <option value={revision.display_ordinal}>Revision {revision.display_ordinal} · {revision.anchorCount} anchors</option>}</For></select></label></fieldset>
          <p class="tuning-ladder-coverage">{coverage()!.revisionCount} stored revisions · {coverage()!.anchorCount} immutable anchor snapshots · {coverage()!.pairCount} joined pairs · {coverage()!.unmatchedPoolRevisions} unmatched legacy pair revisions.</p>
          <div class="tuning-ladder-layout">
            <section class="tuning-ladder-map-panel" aria-label="Pool rating map"><LadderMap anchors={anchors()} allRevisionOrdinals={coverage()!.revisions.map((revision) => revision.display_ordinal)} candidate={candidate()} opponents={opponents()} selectedAnchorKey={navigation().ladderAnchorKey} onSelect={selectAnchor} /><p class="tuning-ladder-legend"><span><i class="tuning-ladder-candidate-symbol" /> candidate provenance</span><span><i class="tuning-ladder-baseline-symbol" /> baseline provenance</span><span><i class="tuning-ladder-other-symbol" /> other provenance</span><span><i class="tuning-ladder-opponent-symbol" /> encountered opponent</span></p></section>
            <section class="tuning-ladder-selection" aria-label="Selected immutable snapshot"><h5>Selected immutable snapshot</h5><Show when={snapshot()} fallback={<p class="tuning-not-recorded">Select an anchor to inspect its immutable configuration. A snapshot absent from the selected revision is Not recorded.</p>}>{(selected) => <><dl><dt>Anchor / revision</dt><dd>{selected().anchor.anchor_id} / {selected().revision.display_ordinal}</dd><dt>Fingerprint</dt><dd>{selected().revision.pool_snapshot_fingerprint}</dd><dt>Provenance / insertion</dt><dd>{selected().anchor.provenance} / {selected().anchor.insertion_reason}</dd><dt>Promotion source</dt><dd>{selected().anchor.source_trial_id ?? "Not recorded"}</dd></dl><pre class="tuning-json">{jsonText(selected().anchor.config)}</pre><PresetCopyAction label="opponent preset" build={buildPresetSpec(opponentPresetSource(selected().anchor, selected().revision))} /></>}</Show></section>
          </div>
          <div class="tuning-ladder-table-wrap"><table aria-label="Exact immutable pool anchors"><thead><tr><th>Revision</th><th>Anchor</th><th>μ ± 2σ</th><th>Family</th><th>Provenance / insertion</th><th>Champion / history source</th><th>Select</th></tr></thead><tbody><For each={anchors()}>{(anchor) => <tr classList={{ "tuning-ladder-selected-row": anchor.selected }}><td>{anchor.revisionOrdinal}</td><td>{anchor.anchorId}</td><td>{numberText(anchor.lower)} – {numberText(anchor.upper)}</td><td>{anchor.family ?? "Not recorded"}</td><td>{anchor.provenance} / {anchor.insertionReason}</td><td>{anchor.sourceTrialId ? `Source trial ${anchor.sourceTrialId}; ` : "No promotion source recorded; "}snapshots {anchor.historyOrdinals.join(", ")}</td><td><button type="button" onClick={() => selectAnchor(anchor.key)} aria-label={`Select anchor ${anchor.anchorId} revision ${anchor.revisionOrdinal}`}>Select</button></td></tr>}</For></tbody></table></div>
          <section class="tuning-ladder-opponents" aria-labelledby="tuning-ladder-opponents-heading">
            <h5 id="tuning-ladder-opponents-heading">Selected trial matchmaking</h5>
            <Show when={navigation().selection.trialId} fallback={<p class="tuning-not-recorded">Select a trial to overlay its recorded rating trajectory and opponents.</p>}>
              <Show when={selectedDetail()} fallback={<p class="tuning-not-recorded">Loading selected trial detail for immutable matchup evidence…</p>}>
                {(detail) => <Show when={opponents().length > 0} fallback={<p class="tuning-not-recorded">Not recorded — this selected trial has no retained pair evidence.</p>}>
                  <div class="tuning-ladder-table-wrap" data-trial-id={detail().trial_id}><table aria-label="Exact selected trial opponents"><thead><tr><th>Pair</th><th>Opponent</th><th>Candidate μ</th><th>Opponent μ</th><th>Δμ / |Δμ|</th><th>Pool revision join</th></tr></thead><tbody><For each={opponents()}>{(opponent) => <tr><td>{opponent.pairIndex + 1}</td><td>{opponent.opponentId}</td><td>{numberText(opponent.candidateMu)}</td><td>{numberText(opponent.opponentMu)}</td><td>{numberText(opponent.deltaMu)} / {numberText(opponent.absoluteMuDistance)}</td><td>{poolJoinText(detail().pairs.find((pair) => pair.pair_id === opponent.pairId)?.pool_revision)}</td></tr>}</For></tbody></table></div>
                </Show>}
              </Show>
            </Show>
          </section>
        </Show>
      </div>}</Show>
    </section>
  );
};
