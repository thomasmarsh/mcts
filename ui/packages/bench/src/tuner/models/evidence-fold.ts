// evidence-fold.ts — pure tallies over the live `evidence.jsonl` stream.
// `foldEvidence` counts events and takes maxima for the progress rail;
// `tickerLines` formats each recent envelope as one human-readable line.
// No confidence interval, bootstrap, calibration, or ranking arithmetic
// lives here — every real statistic stays in the projection / report
// (AGENTS.md "no statistics in TS").
//
// `foldStep`/`freshLiveProgress` are the incremental building blocks behind
// `foldEvidence` (itself just `envelopes.reduce(foldStep, freshLiveProgress())`).
// The reducer folds new envelopes onto its *own* running `LiveProgress`
// one batch at a time, rather than recomputing over the client's bounded
// evidence ring on every update — the same "replay incrementally, not from
// scratch" fix already applied to the tuner's own projection replay. Fully
// recomputing from the ring would also be wrong past ~200 pairs into a
// phase: the ring is capped (`EVIDENCE_RING_MAX`) for ticker display, so a
// long-running phase's earlier `pair_started`/`pair_completed` events fall
// off it and a from-scratch fold would silently under-count.

import type {
  EvidenceEnvelope,
  LivePhase,
  LiveProgress,
  TickerLine,
} from "../tuner-types.js";

type Obj = Record<string, unknown>;

const asObj = (value: unknown): Obj =>
  value !== null && typeof value === "object" ? (value as Obj) : {};
const asStr = (value: unknown): string | null => (typeof value === "string" ? value : null);
const asNum = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const asArr = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

/** Trailing chunk of a `kind-<hex>` identifier, for compact ticker lines. */
export function shortId(id: string | null): string {
  if (!id) return "?";
  const tail = id.includes("-") ? id.slice(id.lastIndexOf("-") + 1) : id;
  return tail.slice(0, 7);
}

export function freshLiveProgress(): LiveProgress {
  return {
    phase: "proposal",
    cohortIndex: null,
    pairs: { started: 0, completed: 0, failed: 0 },
    proposals: {},
    bestSoFar: null,
    lastEventSeq: 0,
  };
}

/** Fold one envelope onto a running `LiveProgress`, returning a new value
 * (the input is never mutated, so it stays safe to hold onto the previous
 * value elsewhere — e.g. across a Solid store's reactivity check). */
export function foldStep(state: LiveProgress, envelope: EvidenceEnvelope): LiveProgress {
  let phase = state.phase;
  let cohortIndex = state.cohortIndex;
  let pairs = state.pairs;
  let proposals = state.proposals;
  let bestSoFar = state.bestSoFar;

  // A phase change zeroes the per-phase pair counters.
  const setPhase = (next: LivePhase): void => {
    if (next !== phase) {
      phase = next;
      pairs = { started: 0, completed: 0, failed: 0 };
    }
  };

  const p = asObj(envelope.payload);

  switch (envelope.type) {
    case "proposal_created":
    case "proposal_accepted":
    case "proposal_rejected": {
      setPhase("proposal");
      const source = asStr(p.source) ?? "unknown";
      const prior = proposals[source] ?? { created: 0, accepted: 0, rejected: 0 };
      const bucket = { ...prior };
      if (envelope.type === "proposal_created") bucket.created += 1;
      else if (envelope.type === "proposal_accepted") bucket.accepted += 1;
      else bucket.rejected += 1;
      proposals = { ...proposals, [source]: bucket };
      const cohort = asNum(p.cohort_index);
      if (cohort !== null) cohortIndex = cohort;
      break;
    }

    case "allocation_decided": {
      const allocation = asObj(p.allocation);
      const cohort = asNum(allocation.cohort_index);
      if (cohort !== null) cohortIndex = cohort;
      if (allocation.kind === "begin_validation") setPhase("validation");
      else if (allocation.kind === "evaluate_diagnostic_pair") setPhase("diagnostic");
      else setPhase("tuning");
      break;
    }

    case "pair_started":
    case "pair_completed":
    case "pair_failed": {
      setPhase(p.phase === "validation" ? "validation" : "tuning");
      if (envelope.type === "pair_started") pairs = { ...pairs, started: pairs.started + 1 };
      else if (envelope.type === "pair_failed") pairs = { ...pairs, failed: pairs.failed + 1 };
      else {
        pairs = { ...pairs, completed: pairs.completed + 1 };
        const utility = asNum(p.pair_utility);
        const candidateId = asStr(p.candidate_id);
        if (
          utility !== null &&
          candidateId !== null &&
          (bestSoFar === null || utility > bestSoFar.pairUtility)
        ) {
          bestSoFar = { candidateId, pairUtility: utility };
        }
      }
      break;
    }

    case "observation_completed":
      setPhase(p.phase === "validation" ? "validation" : "tuning");
      break;

    case "cohort_completed": {
      const cohort = asNum(p.cohort_index);
      if (cohort !== null) cohortIndex = cohort;
      setPhase("tuning");
      break;
    }

    case "diagnostic_pair_started":
    case "diagnostic_pair_completed":
    case "diagnostic_pair_failed":
      setPhase("diagnostic");
      break;

    case "finalists_selected":
    case "run_completed":
      setPhase("done");
      break;

    default:
      break;
  }

  return { phase, cohortIndex, pairs, proposals, bestSoFar, lastEventSeq: envelope.sequence };
}

export function foldEvidence(envelopes: EvidenceEnvelope[]): LiveProgress {
  return envelopes.reduce(foldStep, freshLiveProgress());
}

function signed(value: number | null): string {
  if (value === null) return "?";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

/** Render one envelope as a single human line off its shallow payload. */
export function describeEvent(envelope: EvidenceEnvelope): string {
  const p = asObj(envelope.payload);
  const allocation = asObj(p.allocation);
  switch (envelope.type) {
    case "pair_started":
      return `pair ${shortId(asStr(p.pair_id))} started · ${shortId(asStr(p.candidate_id))} vs ${asStr(p.opponent_id) ?? "?"}`;
    case "pair_completed":
      return `pair ${shortId(asStr(p.pair_id))} done · ${shortId(asStr(p.candidate_id))} vs ${asStr(p.opponent_id) ?? "?"} · ${signed(asNum(p.pair_utility))}`;
    case "pair_failed":
      return `pair ${shortId(asStr(p.pair_id))} failed · ${asStr(p.kind) ?? "error"}`;
    case "proposal_created":
      return `proposal ${asStr(p.source) ?? "?"} created (${shortId(asStr(p.candidate_id))})`;
    case "proposal_accepted":
      return `proposal ${asStr(p.source) ?? "?"} accepted (${shortId(asStr(p.candidate_id))})`;
    case "proposal_rejected":
      return `proposal ${asStr(p.source) ?? "?"} rejected (${shortId(asStr(p.candidate_id))})`;
    case "cohort_completed": {
      const promoted = asArr(p.retained_candidate_ids).length;
      const total = asArr(p.candidate_ids).length;
      return `cohort ${asNum(p.cohort_index) ?? "?"} complete — ${promoted} promoted, ${Math.max(0, total - promoted)} eliminated`;
    }
    case "observation_completed":
      return `observation for ${shortId(asStr(p.candidate_id))} (${asStr(p.phase) ?? "?"})`;
    case "allocation_decided": {
      switch (allocation.kind) {
        case "begin_validation":
          return "validation started";
        case "evaluate_diagnostic_pair":
          return `diagnostic pair allocated · cohort ${asNum(allocation.cohort_index) ?? "?"}`;
        case "introduce_candidate":
          return `candidate introduced (${asStr(allocation.source) ?? "?"})`;
        case "refill_candidate":
          return `candidate refilled (${asStr(allocation.source) ?? "?"})`;
        case "apply_elimination":
          return `elimination applied · cohort ${asNum(allocation.cohort_index) ?? "?"}`;
        case "retain_elites":
          return `elites retained · cohort ${asNum(allocation.cohort_index) ?? "?"}`;
        case "deepen_cohort":
          return `cohort deepened (block ${asNum(allocation.block_index) ?? "?"})`;
        case "suspend_active_elimination":
          return "active elimination suspended";
        default:
          return `allocation: ${asStr(allocation.kind) ?? "?"}`;
      }
    }
    case "shadow_race_decided": {
      const eliminated = asArr(p.decisions).filter(
        (d) => asObj(d).disposition === "eliminate",
      ).length;
      return `shadow race · cohort ${asNum(p.cohort_index) ?? "?"} · ${eliminated} eliminated`;
    }
    case "candidate_failed":
      return `candidate ${shortId(asStr(p.candidate_id))} failed · ${asStr(p.reason) ?? ""}`.trim();
    case "diagnostic_pair_started":
      return `diagnostic pair ${shortId(asStr(p.pair_id))} started`;
    case "diagnostic_pair_completed":
      return `diagnostic pair ${shortId(asStr(p.pair_id))} done`;
    case "diagnostic_pair_failed":
      return `diagnostic pair ${shortId(asStr(p.pair_id))} failed`;
    case "finalists_selected":
      return `finalists selected (${asArr(p.finalist_ids).length})`;
    case "run_completed":
      return "run complete";
    case "budget_extended":
      return `budget extended — ${asStr(p.reason) ?? ""}`.trim();
    case "run_interrupted":
      return `run interrupted (${asStr(p.stage) ?? ""})`.trim();
    case "run_failed":
      return `run failed — ${asStr(p.message) ?? ""}`.trim();
    default:
      return envelope.type;
  }
}

export function tickerLines(envelopes: EvidenceEnvelope[], limit: number): TickerLine[] {
  const slice = limit > 0 ? envelopes.slice(-limit) : envelopes.slice();
  return slice.map((envelope) => ({ seq: envelope.sequence, text: describeEvent(envelope) }));
}
