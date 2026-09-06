// contender-model.ts — pure derivation over the whole-run W/L/D rollup
// (`GET .../projection/runs/{id}/contenders`). Broader than
// `verdict-model.ts::deriveVerdict`, which ranks only the validation
// shortlist: this describes every candidate the run has evaluated, per
// panel opponent and per phase. It is descriptive only and must never feed
// the ship decision.

import type { ProjectionCandidate, ProjectionContenderRow } from "../tuner-types.js";
import { shortCandidateId } from "./verdict-model.js";

export interface WDL {
  wins: number;
  losses: number;
  draws: number;
}

export interface ContenderOpponent extends WDL {
  opponentId: string;
}

export interface ContenderPhase extends WDL {
  phase: string;
}

export interface Contender {
  candidateId: string;
  shortId: string;
  cohortIndex: number | null;
  /** Furthest phase this candidate has pair rows for, by the conventional
   * tuning → validation → diagnostic order; `null` with no pairs yet. */
  phaseReached: string | null;
  overall: WDL;
  /** One entry per distinct panel opponent, ordered by opponent id. */
  byOpponent: ContenderOpponent[];
  /** One entry per phase the candidate reached, in conventional order. */
  byPhase: ContenderPhase[];
  /** Sum of `pair_utility` over every pair, divided by the pair count — the
   * sort key. `null` before any pair completes. */
  meanUtility: number | null;
  /** Total completed pairs (`wins + losses + draws`). */
  games: number;
}

const PHASE_ORDER = ["tuning", "validation", "diagnostic"];
const phaseRank = (phase: string): number => {
  const i = PHASE_ORDER.indexOf(phase);
  return i === -1 ? PHASE_ORDER.length : i;
};

const emptyWDL = (): WDL => ({ wins: 0, losses: 0, draws: 0 });
const addWDL = (into: WDL, add: WDL): void => {
  into.wins += add.wins;
  into.losses += add.losses;
  into.draws += add.draws;
};
const countWDL = (w: WDL): number => w.wins + w.losses + w.draws;

/** Roll the rollup rows up per candidate. `candidates` supplies the cohort
 * index and pulls in any candidate that has no pair rows yet (shown with a
 * zero record) so the table covers every evaluated candidate, not only
 * those with results. */
export function deriveContenders(
  rows: ProjectionContenderRow[],
  candidates: ProjectionCandidate[] | undefined,
): Contender[] {
  const cohortOf = new Map((candidates ?? []).map((c) => [c.candidate_id, c.cohort_index]));
  const grouped = new Map<string, ProjectionContenderRow[]>();
  for (const row of rows) {
    const list = grouped.get(row.candidate_id);
    if (list) list.push(row);
    else grouped.set(row.candidate_id, [row]);
  }
  for (const c of candidates ?? []) {
    if (!grouped.has(c.candidate_id)) grouped.set(c.candidate_id, []);
  }

  const out: Contender[] = [];
  for (const [candidateId, crows] of grouped) {
    const overall = emptyWDL();
    const oppMap = new Map<string, ContenderOpponent>();
    const phaseMap = new Map<string, ContenderPhase>();
    let sumUtility = 0;
    for (const r of crows) {
      const wdl: WDL = { wins: r.wins, losses: r.losses, draws: r.draws };
      addWDL(overall, wdl);
      sumUtility += r.sum_utility;
      const opp = oppMap.get(r.opponent_id) ?? { opponentId: r.opponent_id, ...emptyWDL() };
      addWDL(opp, wdl);
      oppMap.set(r.opponent_id, opp);
      const ph = phaseMap.get(r.phase) ?? { phase: r.phase, ...emptyWDL() };
      addWDL(ph, wdl);
      phaseMap.set(r.phase, ph);
    }
    const games = countWDL(overall);
    const phaseReached =
      [...phaseMap.keys()].sort((a, b) => phaseRank(b) - phaseRank(a))[0] ?? null;
    out.push({
      candidateId,
      shortId: shortCandidateId(candidateId),
      cohortIndex: cohortOf.get(candidateId) ?? null,
      phaseReached,
      overall,
      byOpponent: [...oppMap.values()].sort((a, b) =>
        a.opponentId < b.opponentId ? -1 : a.opponentId > b.opponentId ? 1 : 0,
      ),
      byPhase: [...phaseMap.values()].sort((a, b) => phaseRank(a.phase) - phaseRank(b.phase)),
      meanUtility: games > 0 ? sumUtility / games : null,
      games,
    });
  }

  return out.sort((a, b) => {
    const au = a.meanUtility ?? Number.NEGATIVE_INFINITY;
    const bu = b.meanUtility ?? Number.NEGATIVE_INFINITY;
    if (au !== bu) return bu - au;
    return a.candidateId < b.candidateId ? -1 : 1;
  });
}

/** `"W / D / L"` — the order the validation table already uses. */
export function formatWDL(w: WDL): string {
  return `${w.wins} / ${w.draws} / ${w.losses}`;
}

export function findContender(
  list: Contender[],
  candidateId: string,
): Contender | undefined {
  return list.find((c) => c.candidateId === candidateId);
}
