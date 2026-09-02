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
import { StepLine } from "../primitives/StepLine.js";
import { FunnelBars } from "../primitives/FunnelBars.js";
import { KpiRow } from "../primitives/KpiRow.js";
import { RaceStrip } from "../primitives/RaceStrip.js";
import { Forest } from "../primitives/Forest.js";
import { DataTable } from "../primitives/DataTable.js";

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
              : "Science is available once the run's report has been projected."}
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
      </Show>
    </div>
  );
};
