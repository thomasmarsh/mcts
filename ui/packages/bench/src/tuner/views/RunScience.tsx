// RunScience — the irace-style scientific report for one run. Each section
// maps to one `report.json` key, renders one or two chart primitives with a
// plain-language caption, and hides its raw numbers behind a "show numbers"
// toggle. Section collapse state is remembered per-viewer in localStorage.
// This slice covers convergence, the proposal-search funnel, the cohort
// race, and per-cohort observations; the remaining sections are later
// slices.

import { createMemo, createSignal, For, Show, type Component, type JSX } from "solid-js";
import type { Store } from "@mcts/core";
import { peek, isLoading } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import { deriveProposalFunnel } from "../models/funnel-model.js";
import { deriveCohortRace } from "../models/race-model.js";
import { deriveConvergence, deriveObservations } from "../models/science-models.js";
import { deriveElimination } from "../models/elimination-model.js";
import { deriveOpponentResponse, type OpponentRow } from "../models/opponent-model.js";
import { deriveDiagnosticGraph } from "../models/diagnostic-model.js";
import { deriveComputeLedger } from "../models/compute-model.js";
import { StepLine } from "../primitives/StepLine.js";
import { FunnelBars } from "../primitives/FunnelBars.js";
import { KpiRow } from "../primitives/KpiRow.js";
import { RaceStrip } from "../primitives/RaceStrip.js";
import { Forest } from "../primitives/Forest.js";
import { DataTable } from "../primitives/DataTable.js";
import { Heatmap } from "../primitives/Heatmap.js";
import { CycleGraph } from "../primitives/CycleGraph.js";
import { Treemap } from "../primitives/Treemap.js";

const STORAGE_KEY = "tuner.science.collapsed";

function loadCollapsed(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? (parsed as Record<string, boolean>) : {};
  } catch {
    return {};
  }
}

function saveCollapsed(map: Record<string, boolean>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // per-viewer convenience only — a blocked store is fine.
  }
}

const Section: Component<{
  id: string;
  title: string;
  caption: string;
  collapsed: Record<string, boolean>;
  toggle: (id: string) => void;
  children: JSX.Element;
  numbers?: JSX.Element;
}> = (props) => {
  const [showNumbers, setShowNumbers] = createSignal(false);
  const open = (): boolean => !props.collapsed[props.id];
  return (
    <section class="tuner-science-section" data-testid={`science-${props.id}`}>
      <button class="tuner-science-heading" onClick={() => props.toggle(props.id)}>
        <span class="tuner-science-caret">{open() ? "▾" : "▸"}</span>
        <h3>{props.title}</h3>
      </button>
      <Show when={open()}>
        <p class="tuner-science-caption">{props.caption}</p>
        {props.children}
        <Show when={props.numbers}>
          <button class="tuner-science-numbers-toggle" onClick={() => setShowNumbers((v) => !v)}>
            {showNumbers() ? "Hide numbers" : "Show numbers"}
          </button>
          <Show when={showNumbers()}>{props.numbers}</Show>
        </Show>
      </Show>
    </section>
  );
};

export const RunScience: Component<{
  store: Store<TunerState, TunerAction>;
  runId: string;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const report = createMemo(() => peek(state().report));
  const candidates = createMemo(() => peek(state().candidates));
  const reportPending = createMemo(() => isLoading(state().report) && !report());

  const funnel = createMemo(() => deriveProposalFunnel(report()));
  const race = createMemo(() => deriveCohortRace(report(), candidates()));
  const convergence = createMemo(() => deriveConvergence(report()));
  const observations = createMemo(() => deriveObservations(report()));
  const elimination = createMemo(() => deriveElimination(report()));
  const opponents = createMemo(() => deriveOpponentResponse(report()));
  const diagnostic = createMemo(() => deriveDiagnosticGraph(report()));
  const compute = createMemo(() => deriveComputeLedger(report()));

  const [collapsed, setCollapsed] = createSignal<Record<string, boolean>>(loadCollapsed());
  const toggle = (id: string): void => {
    setCollapsed((prev) => {
      const next = { ...prev, [id]: !prev[id] };
      saveCollapsed(next);
      return next;
    });
  };

  const openCandidate = (candidateId: string): void =>
    props.navigate({ view: "run", runId: props.runId, tab: "science", candidate: candidateId });

  return (
    <div class="tuner-run-science" data-testid="tuner-run-science">
      <div class="tuner-run-overview-header">
        <button
          class="tuner-back"
          onClick={() => props.navigate({ view: "run", runId: props.runId, tab: "overview" })}
        >
          ← Overview
        </button>
        <h2>{props.runId} · science</h2>
        <button
          onClick={() => dispatch({ tag: "refreshProjection" })}
          disabled={state().refreshing}
        >
          {state().refreshing ? "Refreshing…" : "Refresh science"}
        </button>
      </div>

      <Show
        when={report()}
        fallback={
          <p class="tuner-fleet-empty">
            {reportPending()
              ? "Loading the run report…"
              : "Science is available once the run's report has been projected. " +
                "Live sections populate from the projection while the run is in progress; " +
                "until then, follow the run from its overview's live event feed."}
          </p>
        }
      >
        <Section
          id="convergence"
          title="Convergence"
          caption="The leading candidate's largest margin over its cohort's elimination boundary, one step per cohort — the tuner's best-so-far signal."
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="convergence-numbers"
              rows={convergence().steps}
              rowKey={(s) => String(s.cohortIndex)}
              columns={[
                { key: "c", header: "Cohort", render: (s) => s.cohortIndex },
                { key: "leader", header: "Leader", render: (s) => s.leaderShortId ?? "—" },
                {
                  key: "m",
                  header: "Best margin",
                  align: "right",
                  render: (s) => s.bestMargin.toFixed(3),
                },
              ]}
            />
          }
        >
          <Show
            when={convergence().present}
            fallback={<p class="tuner-fleet-empty">No cohorts recorded.</p>}
          >
            <StepLine
              points={convergence().steps.map((s) => ({ x: s.x, y: s.bestMargin, label: s.label }))}
              domain={convergence().domain}
            />
          </Show>
        </Section>

        <Section
          id="proposal-search"
          title="Proposal search"
          caption="Where candidate configurations came from: configured budget, attempts made, and how many each source landed."
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="funnel-numbers"
              rows={funnel().stages}
              rowKey={(s) => s.source}
              columns={[
                { key: "s", header: "Source", render: (s) => s.label },
                { key: "cfg", header: "Configured", align: "right", render: (s) => s.configured ?? "—" },
                { key: "att", header: "Attempted", align: "right", render: (s) => s.attempted },
                { key: "acc", header: "Accepted", align: "right", render: (s) => s.accepted },
                { key: "rej", header: "Rejected", align: "right", render: (s) => s.rejected },
              ]}
            />
          }
        >
          <Show
            when={funnel().present}
            fallback={<p class="tuner-fleet-empty">No proposal-search record.</p>}
          >
            <FunnelBars
              rows={funnel().stages.map((s) => ({
                label: s.label,
                total: s.attempted,
                filled: s.accepted,
                note: s.rejected > 0 ? `${s.rejected} rejected` : undefined,
              }))}
            />
            <KpiRow items={funnel().kpis} />
          </Show>
        </Section>

        <Section
          id="cohort-race"
          title="Cohort race"
          caption={`Shadow disposition for each candidate at each common prefix. ${
            race().enforced ? "Elimination was enforced." : "Recorded but not enforced."
          }`}
          collapsed={collapsed()}
          toggle={toggle}
        >
          <Show
            when={race().present}
            fallback={<p class="tuner-fleet-empty">No shadow-elimination record.</p>}
          >
            <For each={race().cohorts}>
              {(cohort) => (
                <div class="tuner-race-cohort">
                  <h4>Cohort {cohort.cohortIndex}</h4>
                  <RaceStrip
                    columns={cohort.prefixes.map((p) => ({
                      label: `p${p.index}`,
                      title: p.prefixId,
                    }))}
                    rows={cohort.rows.map((r) => ({
                      key: r.candidateId,
                      label: r.shortId,
                      note: [
                        r.protected ? "protected" : null,
                        r.finalTopSet ? "top set" : null,
                        r.source,
                      ]
                        .filter(Boolean)
                        .join(" · "),
                      highlight: r.finalTopSet,
                      cells: r.cells,
                      onClick: () => openCandidate(r.candidateId),
                    }))}
                  />
                </div>
              )}
            </For>
            <Show when={race().dispositions.length > 0}>
              <p class="tuner-race-legend">Dispositions: {race().dispositions.join(", ")}</p>
            </Show>
          </Show>
        </Section>

        <Section
          id="observations"
          title="Observations"
          caption="Per-candidate performance across the opponent panel at the maximum tuning prefix. The bar is the envelope across opponents, not a re-estimated interval."
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="observation-numbers"
              rows={observations().rows}
              rowKey={(r) => r.candidateId}
              onRowClick={(r) => openCandidate(r.candidateId)}
              columns={[
                { key: "id", header: "Candidate", render: (r) => r.shortId },
                { key: "m", header: "Mean", align: "right", render: (r) => r.mean.toFixed(3) },
                {
                  key: "iv",
                  header: "Envelope",
                  align: "right",
                  render: (r) => `[${r.lower.toFixed(3)}, ${r.upper.toFixed(3)}]`,
                },
                { key: "opp", header: "Opponents", align: "right", render: (r) => r.opponents },
              ]}
            />
          }
        >
          <Show
            when={observations().present}
            fallback={
              <p class="tuner-fleet-empty">No opponent-response analysis in this report.</p>
            }
          >
            <Show when={observations().cohortIndex != null}>
              <h4>Cohort {observations().cohortIndex}</h4>
            </Show>
            <Forest
              domain={observations().domain}
              rows={observations().rows.map((r) => ({
                key: r.candidateId,
                label: r.shortId,
                mean: r.mean,
                lower: r.lower,
                upper: r.upper,
                onClick: () => openCandidate(r.candidateId),
              }))}
            />
          </Show>
        </Section>

        <Section
          id="elimination"
          title={elimination().enforced ? "Active elimination" : "Shadow elimination"}
          caption={
            elimination().enforced
              ? "Candidates were pruned mid-race; a randomized audit re-checks a fraction of the cuts, and a safety rule suspends pruning on an audited boundary reversal."
              : "Elimination decisions were recorded but never enforced. Calibration compares each shadow decision's predicted promotion probability against what actually happened."
          }
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="calibration-numbers"
              rows={elimination().calibrationBins}
              rowKey={(b) => `${b.lower}-${b.upper}`}
              empty="No calibration bins (active elimination records none)."
              columns={[
                { key: "band", header: "Predicted band", render: (b) => `${b.lower.toFixed(2)}–${b.upper.toFixed(2)}` },
                { key: "pred", header: "Mean prediction", align: "right", render: (b) => b.meanPrediction.toFixed(3) },
                { key: "obs", header: "Observed rate", align: "right", render: (b) => b.observedRate.toFixed(3) },
                { key: "n", header: "Count", align: "right", render: (b) => b.count },
              ]}
            />
          }
        >
          <Show
            when={elimination().present}
            fallback={<p class="tuner-fleet-empty">No elimination record in this report.</p>}
          >
            <Show when={elimination().suspended}>
              <p class="tuner-science-warn">
                Pruning suspended: {elimination().suspensionReason}
              </p>
            </Show>
            <KpiRow items={elimination().kpis} testid="elimination-kpis" />
            <Show when={elimination().calibrationBins.length > 0}>
              <Heatmap
                testid="calibration-heatmap"
                columns={elimination().calibrationBins.map((b) => ({
                  key: `${b.lower}-${b.upper}`,
                  label: `${b.lower.toFixed(1)}–${b.upper.toFixed(1)}`,
                  title: `${b.count} decisions`,
                }))}
                rows={[
                  {
                    key: "predicted",
                    label: "predicted",
                    cells: elimination().calibrationBins.map((b) => ({
                      label: b.meanPrediction.toFixed(2),
                      intensity: b.meanPrediction,
                    })),
                  },
                  {
                    key: "observed",
                    label: "observed",
                    cells: elimination().calibrationBins.map((b) => ({
                      label: b.observedRate.toFixed(2),
                      intensity: b.observedRate,
                      flag: Math.abs(b.observedRate - b.meanPrediction) > 0.1,
                    })),
                  },
                ]}
              />
            </Show>
          </Show>
        </Section>

        <Section
          id="opponent-response"
          title="Opponent response"
          caption="Each finalist's mean pair utility against every panel opponent. Flagged cells are where a pairwise interaction found a material (non-tie) contrast or a ranking reversal."
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="opponent-numbers"
              rows={opponents().rows}
              rowKey={(r) => r.candidateId}
              onRowClick={(r) => openCandidate(r.candidateId)}
              columns={[
                { key: "id", header: "Candidate", render: (r) => r.shortId },
                { key: "mean", header: "Mean", align: "right", render: (r) => r.mean.toFixed(3) },
                ...opponents().opponentIds.map((opp, i) => ({
                  key: opp,
                  header: opp,
                  align: "right" as const,
                  render: (r: OpponentRow) => {
                    const cell = r.cells[i];
                    return cell?.mean == null ? "—" : cell.mean.toFixed(3);
                  },
                })),
              ]}
            />
          }
        >
          <Show
            when={opponents().present}
            fallback={<p class="tuner-fleet-empty">No opponent-response analysis in this report.</p>}
          >
            <Heatmap
              testid="opponent-heatmap"
              columns={opponents().opponentIds.map((opp) => ({ key: opp, label: opp }))}
              rows={opponents().rows.map((r) => ({
                key: r.candidateId,
                label: r.shortId,
                onClick: () => openCandidate(r.candidateId),
                cells: r.cells.map((c) => ({
                  label: c.mean == null ? "—" : c.mean.toFixed(2),
                  title:
                    c.mean == null
                      ? "no games"
                      : `${c.mean.toFixed(3)} [${(c.lower ?? 0).toFixed(2)}, ${(c.upper ?? 0).toFixed(2)}]`,
                  intensity: c.mean ?? 0,
                  flag: c.flagged,
                })),
              }))}
            />
            <KpiRow items={opponents().kpis} testid="opponent-kpis" />
          </Show>
        </Section>

        <Section
          id="diagnostic"
          title="Diagnostic matchup graph"
          caption="Direct candidate-vs-candidate games run to resolve the objective ranking. An arrow A → B means A beat B; highlighted nodes sit in a material preference cycle."
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="diagnostic-numbers"
              rows={diagnostic().edges}
              rowKey={(e) => `${e.from}-${e.to}`}
              empty="No direct diagnostic edges."
              columns={[
                { key: "from", header: "Winner", render: (e) => e.from.replace(/^candidate-/, "").slice(0, 12) },
                { key: "to", header: "Loser", render: (e) => e.to.replace(/^candidate-/, "").slice(0, 12) },
                { key: "est", header: "Estimate", align: "right", render: (e) => (e.estimate == null ? "—" : e.estimate.toFixed(3)) },
                {
                  key: "iv",
                  header: "Interval",
                  align: "right",
                  render: (e) => (e.lower == null || e.upper == null ? "—" : `[${e.lower.toFixed(2)}, ${e.upper.toFixed(2)}]`),
                },
                { key: "pairs", header: "Pairs", align: "right", render: (e) => e.pairCount },
              ]}
            />
          }
        >
          <Show
            when={diagnostic().present}
            fallback={<p class="tuner-fleet-empty">No diagnostic matchup graph in this report.</p>}
          >
            <Show
              when={diagnostic().hasBudget}
              fallback={
                <p class="tuner-fleet-empty">
                  No diagnostic budget was spent — the objective ranking was accepted directly.
                </p>
              }
            >
              <CycleGraph
                nodes={diagnostic().nodes.map((n) => ({
                  key: n.candidateId,
                  label: n.shortId,
                  badge: `#${n.rank}`,
                  highlight: n.inCycle,
                  onClick: () => openCandidate(n.candidateId),
                }))}
                edges={diagnostic().edges.map((e) => ({
                  from: e.from,
                  to: e.to,
                  undirected: e.undirected,
                  label: e.estimate == null ? undefined : e.estimate.toFixed(3),
                }))}
              />
            </Show>
            <Show when={diagnostic().cycles.length > 0}>
              <p class="tuner-science-warn">
                Material cycle:{" "}
                {diagnostic()
                  .cycles.map((c) => c.members.join(" ⇄ "))
                  .join("; ")}
              </p>
            </Show>
            <Show when={diagnostic().shortlist.reserveDisplaced}>
              <p class="tuner-science-warn">
                Cycle reserve displaced an objective pick:{" "}
                {diagnostic().shortlist.displacedId?.replace(/^candidate-/, "").slice(0, 12)}
              </p>
            </Show>
            <KpiRow items={diagnostic().kpis} testid="diagnostic-kpis" />
          </Show>
        </Section>

        <Section
          id="compute"
          title="Compute ledger"
          caption="Where the pair-attempt budget went, per phase: completed, failed, censored, overrun, and unspent."
          collapsed={collapsed()}
          toggle={toggle}
          numbers={
            <DataTable
              testid="compute-numbers"
              rows={compute().phases}
              rowKey={(p) => p.phase}
              columns={[
                { key: "phase", header: "Phase", render: (p) => p.label },
                { key: "budget", header: "Budget", align: "right", render: (p) => p.budget },
                { key: "attempts", header: "Attempts", align: "right", render: (p) => p.pairAttempts },
                { key: "done", header: "Completed", align: "right", render: (p) => p.completedPairs },
                { key: "failed", header: "Failed", align: "right", render: (p) => p.failedAttempts },
                { key: "censored", header: "Censored", align: "right", render: (p) => p.censoredAttempts },
                { key: "overrun", header: "Overrun", align: "right", render: (p) => p.overrunPairAttempts },
                { key: "unspent", header: "Unspent", align: "right", render: (p) => p.unspentPairAttempts },
                { key: "games", header: "Games", align: "right", render: (p) => p.physicalGames },
              ]}
            />
          }
        >
          <Show
            when={compute().present}
            fallback={<p class="tuner-fleet-empty">No compute ledger in this report.</p>}
          >
            <Treemap groups={compute().treemap} />
            <KpiRow items={compute().kpis} testid="compute-kpis" />
            <Show when={compute().extensions.length > 0}>
              <ul class="tuner-science-extensions">
                <For each={compute().extensions}>
                  {(ext) => (
                    <li>
                      <strong>{ext.label}</strong> — {ext.detail}
                    </li>
                  )}
                </For>
              </ul>
            </Show>
          </Show>
        </Section>
      </Show>
    </div>
  );
};
