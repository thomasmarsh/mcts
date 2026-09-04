# MCTS game tuner

`tuner` is a foreground, reproducible strategy tuner for executables that
implement the game-host `describe`, `compare validate`, and `compare eval`
protocol. It freezes an explicit deployment objective before creating an
artifact, then runs as many complete retained-elite cohorts as the declared
evaluation budget funds. The first cohort uses the bootstrap/SMAC/random-reserve
schedule; at each completed-cohort boundary the top `--finalists` candidates are
retained as elites and the next challenger cohort (filled from the frozen
challenger schedule) starts only when the remaining tuning budget can fund all
of its planned new pairs. When another whole cohort does not fit, the latest
cohort's finalists receive held-out validation.

An objective is strict JSON containing the schema default and one or more raw,
inline historical opponent configurations. The checked-in Druid deployment
objective is `tuner/objectives/druid-reference-v1.json`; Python never resolves
named Rust presets at runtime.

```bash
uv run --project tuner tuner \
  --game-binary target/release/game-druid \
  --objective-file tuner/objectives/druid-reference-v1.json \
  --run-dir /tmp/mcts-tuner-druid \
  --seed 7 --task-seed 11 --cohort-size 8 --finalists 2 \
  --bootstrap-candidates 3 --random-reserve-candidates 2 \
  --constraint '{"select": {"choices": ["ucb1", "ucb1_tuned", "rave"]}}' \
  --tuning-pairs 6 --tuning-pair-budget 132 --diagnostic-pair-budget 8 --validation-pair-budget 12 \
  --production-validation-pairs 12 \
  --tuning-max-iterations 16 --validation-max-iterations 32 \
  --production-max-iterations 64
```

`--evaluator-workers` is an operational setting and defaults to `1`. Each
evaluator runs one search thread, so the worker count cannot exceed the
available logical CPUs. Values above one execute an allocator-ordered batch of
seat-swapped pair subprocesses concurrently; starts may be batched, but terminal
evidence is committed in the same canonical order as sequential execution.
Worker count is not frozen in the manifest, so a run may resume with a different
count. Tuning pairs retry automatically after one recorded failure. After two
started, incomplete attempts for the same tuning pair, the frozen
`terminal-candidate-refill-v1` policy records `candidate_failed`, preserves the
candidate's factual work, removes it from the live cohort, and refills the
vacancy through the ordinary scheduled proposal source (or `random_reserve`
after the schedule is exhausted). Validation failures still require explicit
resume and never trigger finalist replacement. Interrupting a run cancels active
children, leaving any uncommitted starts censored; those starts count toward the
same two-attempt tuning limit on resume.

Panel weights produce a deterministic weighted-fair task order. Every task
names the exact panel opponent, canonical configuration fingerprint, seed, and
start stratum it uses. `--seed` controls proposal streams only;
`--task-seed` controls the disjoint tuning and held-out validation corpora.
`--constraint JSON` may be repeated to restrict the tuning space for this run —
an array of `{"when"?: {...}, "set": {...}}` entries, or the bare
`{name: {fix|range|choices}}` map as sugar for one un-predicated entry. Each
narrowing may only constrain (never widen) the declared schema. Constraints
apply to candidate proposals only; the frozen set is validated for resume and
does not change schema-default or inline objective opponents.
All configured task counts must be complete panel weight cycles. `--tuning-pairs`
is the maximum tuning prefix: the tuner evaluates every accepted candidate on
each cumulative complete-cycle prefix before deepening the full cohort, with no
elimination. `--finalists` is both the retained-elite count and the final
shortlist count. The selected validation corpus is always a leading prefix of the
frozen production validation corpus.

At a complete non-final tuning prefix containing at least 12 pairs, the tuner
records a deterministic paired, stratum-aware `shadow_race_decided` screening
disposition. The 12-pair minimum, practical margin, and nominal elimination
threshold are frozen in the manifest. This is evidence only: every candidate
still reaches the maximum tuning prefix, and the nominal threshold has not earned
an active-pruning safety claim. Runs with no eligible non-final prefix are valid
and record no shadow decisions.

`--shadow-policy {paired_bootstrap,successive_halving}` selects which frozen
policy records those dispositions; it is manifest-frozen and resume-sensitive.
The default `paired_bootstrap` is the stratum-aware bootstrap above.
`--active-elimination-audit-probability` accepts `paired_bootstrap` at any
setting and accepts `successive_halving` only with a positive
`--shadow-halving-spare-margin` (method version
`successive-halving-spare-near-tie-v1`), the gate-approved spare-near-tie policy;
the plain eta-2 cut stays shadow-only. `successive_halving` is
a control that, at each eligible common prefix, starts from the full
cohort roster, applies its own prior batches, ranks the surviving candidates by
their common-prefix point estimate (fingerprint breaking ties), keeps the first
`max(finalists, ceil(survivors / 2), retained elites)` of them, and marks the
rest eliminated. With a positive `--shadow-halving-spare-margin`, a would-be
eliminated candidate whose paired mean at the cut prefix is within the margin of
the last kept candidate is carried to the next look instead (`spare_margin` of
`0.0` is exactly the plain eta-2 cut). Retained elites are always protected. It
makes no confidence claim; its `--shadow-practical-margin` only defines the
audit's recovered boundary, and an explicitly non-default paired threshold is
rejected with it.
The `report.json` `shadow_elimination` section is tagged by policy: paired looks
keep their calibration and Brier score, successive-halving looks expose rank and
prior/target survivor counts with calibration fields reported as not applicable.

Passing `--active-elimination-audit-probability` with a finite value strictly
between zero and one opts into activation-validation mode. After each eligible
shadow decision the tuner records an `allocation_decided` batch that either
prunes an eliminated candidate or deterministically continues it as an audit.
Each batch action carries a typed `decision_margin`: a `paired_probability`
margin (threshold, favorable probability, and their difference) for a paired
decision, or a `successive_halving_rank` margin (rank, target survivor count,
ranks below the cutoff, and the spared-candidate count for that look) for a rank
decision. The manifest's active specification binds the selected shadow-policy
kind, exact method version, and spare margin, so a resume cannot pair an active
audit with another decision policy. Audits and recorded boundaries remain through
the maximum prefix; pruned candidates are not replaced within that cohort. The
option is frozen for resume
At the completed-cohort boundary, an audited candidate that reaches its exact
recorded boundary candidate at maximum tuning fidelity suspends later active
pruning while shadow decisions and full-cohort tuning continue.

`report.json` includes a `candidate_lifecycle` projection of this policy,
terminal failures, and replacement lineage, plus a `shadow_elimination` audit
of those frozen decisions.
It labels each candidate against the same cohort's maximum tuning-prefix top
set, while calibration and stratum reversals compare against the exact early
boundary candidate recorded in the decision. `top_set_false_elimination_rate`
uses eligible unprotected top-set paths as its denominator; `trash_precision`
uses counterfactual eliminations and calls only candidates outside that same
tuning top set “trash.” Avoided work is factual suffix work after the first
unprotected elimination, including retries and partial failed attempts. The
section also reports fixed probability-bin calibration and Brier score only for
looks an active path would reach. It never uses held-out validation, is not an
anytime-valid safety guarantee. Shadow runs retain this audit; active runs
instead expose their observed allocation batches in `active_elimination`, tagged
with the bound policy kind and method version. That section keeps projected
unique-pair savings separate from factual compute:
`gross_nominal_suffix_unique_pairs` sums the manifest tuning cases strictly after
each first nominal elimination prefix, `audit_continuation_suffix_unique_pairs`
restricts that sum to audited continuations, and
`planned_unique_pair_savings` is their difference — the unique suffix pairs
omitted for pruned candidates. It is prefix arithmetic, not observed wall time,
and does not model retries or failures; actual attempts, games, iterations, and
wall time come only from the compute ledger.

`--tuning-pair-budget` and `--validation-pair-budget` are required total
budgets over pair attempts, frozen in the manifest under the
`safe-boundary-pair-attempts-v1` policy. The initial cohort always runs; a later
challenger cohort is admitted at a completed-cohort boundary only when the
remaining tuning budget (counting every started pair attempt, including ones
that later fail or are interrupted) covers all of its planned new pairs. The
validation budget divides evenly across the finalists and derives one common
held-out prefix (`budget / finalists` pairs each, at least one complete panel
cycle, never longer than `--production-validation-pairs`). The number of
cohorts is an output of the budget, not a configuration.

`--diagnostic-pair-budget` defaults to zero. When positive, it permits direct,
seat-swapped candidate-versus-candidate pairs only after the final affordable
cohort and before finalist selection. These pairs use the frozen tuning search
effort and a deterministic graph policy; they never enter objective
observations, proposal costs, elimination, held-out estimates, or deployment
claims. The report exposes their separate compute bucket and direct matchup
graph. A 95% Hoeffding interval must establish every edge of a directed cycle
before one cycle-connected candidate outside the objective shortlist may take
the last validation slot; the objective winner is always retained. Direct-edge
intervals are per-edge and are not graph-wide multiplicity corrected.

The budgets are soft caps at scientifically safe boundaries: the tuner never
stops between the two seats of a pair or inside an admitted cohort, so a retry
after a failure or interruption may push actual attempts over the declared cap.
The report's `compute` section accounts for this truthfully — per-phase pair
attempts, completed pairs, failed and censored attempts, physical games,
actual search iterations, recorded game wall time, and unspent/overrun pair
attempts relative to the frozen budgets.

The run directory contains three version-4 artifacts:

- `manifest.json` freezes the resolved objective/panel, full corpora and
  selected prefixes, fidelity axes, mixed proposal schedule, model dependency
  versions, total compute budgets, and derived objective epoch.
- `evidence.jsonl` records append-only proposal, pair-atomic, contextual
  observation, selection, and completion evidence.
- `report.json` is a replaceable projection with proposal-search provenance, weighted held-out marginals,
  per-opponent matchup rows, matched finalist differences, unresolved ties, and
  the evidence-derived compute ledger and shadow-elimination audit.

Validation is `production` only when both axes reach their declared target:
the selected held-out validation prefix is the complete production corpus and
its search effort exactly equals the declared production effort. Every other result is
`mechanics_smoke`, with the lower axis or axes named in the report.

Resume uses the same scientific options and objective file:

```bash
uv run --project tuner tuner \
  --game-binary target/release/game-druid \
  --objective-file tuner/objectives/druid-reference-v1.json \
  --run-dir /tmp/mcts-tuner-druid --resume \
  --seed 7 --task-seed 11 --cohort-size 8 --finalists 2 \
  --bootstrap-candidates 3 --random-reserve-candidates 2 \
  --constraint '{"select": {"choices": ["ucb1", "ucb1_tuned", "rave"]}}' \
  --tuning-pairs 6 --tuning-pair-budget 132 --validation-pair-budget 12 \
  --production-validation-pairs 12 \
  --tuning-max-time-ms 16 --validation-max-time-ms 32 \
  --production-max-time-ms 64
```

Resume validates the manifest and complete evidence log before append. It
rejects changed objective content/order/weights/configurations, task corpora,
prefixes, efforts, budgets, or epoch before evaluating another game. The objective and
binary paths may move when their resolved scientific identity is unchanged;
`--pair-timeout-seconds` and `--evaluator-workers` remain operational. Resuming a completed run only
rebuilds `report.json`. A resumed run reproduces the uninterrupted run's
scientific projection, selection, and validation exactly; only the compute
ledger truthfully records the extra censored or retried attempts, including any
budget overrun they cause.

## Whole-run proposer policies and bake-offs

`--proposer-policy` selects one frozen whole-run proposal policy. Its default,
`smac_mixed`, preserves the SMAC-guided schedule. The other measured policies
are `random`, `qmc` (scrambled Sobol), and `irace_generational` (a stateless
elite-centred baseline). The selection is manifest-sensitive; it never changes
the default automatically.

Run a matched policy experiment with `tuner-proposer-bakeoff`:

```bash
uv run --project tuner tuner-proposer-bakeoff \
  --spec /tmp/druid-proposer-bakeoff.json \
  --experiment-dir /tmp/druid-proposer-bakeoff
```

The strict version-one specification fixes the four policies in this order:
`random`, `qmc`, `smac_mixed`, `irace_generational`; it also fixes at least four
proposal seeds, increasing tuning pair budgets, the task seed, objective, and
all shared run settings. The experiment directory has an immutable
`experiment.json`, ordinary replayable child run directories, and a replaceable
`results.json`. `--resume` continues incomplete children through the normal
foreground evidence path and rebuilds the result projection from completed
child artifacts.

## Elimination bake-off

`tuner-elimination-bakeoff` compares complete elimination systems at equal
declared compute. It expands each `(tuning pair budget, proposal seed)` into
three matched child runs that differ only in the elimination policy and its
active specification:

- `no_elimination` records paired shadow evidence but never enforces it;
- `paired_elimination` enforces the landed all-strata audited paired policy at
  audit probability `0.25`;
- `spare_near_tie` enforces the gate-approved audited spare-near-tie
  successive-halving policy (`successive-halving-spare-near-tie-v1`, spare margin
  `0.10`) at the same audit probability.

```bash
uv run --project tuner tuner-elimination-bakeoff \
  --spec /tmp/druid-elimination-bakeoff.json \
  --experiment-dir /tmp/druid-elimination-bakeoff
```

The strict version-1 specification fixes the three policies in that order, the
`smac_mixed` proposer, at least four distinct proposal seeds, at least two
increasing tuning budgets, zero diagnostic budget, full production validation,
and a `gate` block that must equal the landed authorization
(`task-11-successive-halving-shadow-gate.md`, `PASS`,
`successive-halving-spare-near-tie-v1`). The `results.json` reports the
Session-11a held-out quality, simple regret, and top-set recall against a
union-of-returned-finalists reference set, seed-paired policy contrasts,
per-arm active-safety summaries, budget reinvestment versus continue-all, and
one frozen largest-budget rule that emits exactly `keep_paired_elimination`,
`change_to_spare_near_tie`, or `reject_active_elimination`. An active arm is
`safe_in_bakeoff` only when every completed cell recorded zero audited boundary
reversals and no suspension. The decision is evidence, not a self-modifying
configuration; it never changes the normal tuner default. Zero audited
reversals in a finite bake-off is not a universal safety guarantee.

A tiny smoke spec (mechanics, replay, accounting, and result projection only):

```json
{
  "schema_version": 1,
  "experiment_id": "druid-elimination-smoke",
  "game_binary": "target/release/game-druid",
  "objective_file": "tuner/objectives/druid-reference-v1.json",
  "policies": ["no_elimination", "paired_elimination", "spare_near_tie"],
  "proposal_seeds": [1, 2, 3, 4],
  "task_seed": 43,
  "tuning_pair_budgets": [112, 140],
  "shared_run": {
    "proposer_policy": "smac_mixed",
    "cohort_size": 4, "finalists": 1,
    "bootstrap_candidates": 2, "random_reserve_candidates": 1,
    "tuning_pairs": 14, "validation_pair_budget": 2,
    "production_validation_pairs": 2, "diagnostic_pair_budget": 0,
    "tuning_effort": {"kind": "iterations", "value": 200},
    "validation_effort": {"kind": "iterations", "value": 1000},
    "production_effort": {"kind": "iterations", "value": 1000},
    "constraints": [{"set": {"algorithm": {"choices": ["mcts", "flat_mc"]}}}],
    "evaluator_workers": 3, "pair_timeout_seconds": 600,
    "active_audit_probability": 0.25
  },
  "decision": {
    "score_practical_margin": 0.0,
    "recall_noninferiority_margin": 0.1,
    "top_set_k": 1
  },
  "gate": {
    "document_id": "task-11-successive-halving-shadow-gate.md",
    "decision": "PASS",
    "authorized_policy_version": "successive-halving-spare-near-tie-v1"
  }
}
```

A production-equivalent run keeps the same structure with the full cohort,
finalist, and validation counts, the full production search effort on all three
phases, at least four seeds, and increasing budgets sized to admit several
cohorts. The Task-11 allocation decision requires that completed
production-equivalent `results.json`, whose experiment/child/result artifacts
are preserved outside the repository. The experiment directory has an immutable
`experiment.json`, ordinary replayable child run directories, and a replaceable
`results.json`; `--resume` continues incomplete children and rebuilds the
result projection byte-identically.
