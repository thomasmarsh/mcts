# MCTS game tuner

`tuner` is a foreground, reproducible strategy tuner for game-host executables
that implement `describe`, `compare validate`, and `compare eval`. It freezes
the deployment objective before it starts, retains the best candidates between
cohorts, and validates the final shortlist against held-out tasks.

The tuner is deliberately conservative about what its results mean. It records
the evidence needed to replay a run, keeps tuning and validation data separate,
and labels reduced-fidelity results as `mechanics_smoke`, not `production`.

## Start a run

The checked-in Druid objective is
`tuner/objectives/druid-reference-v1.json`. It contains the schema default and
raw, inline historical opponent configurations. Python does not resolve named
Rust presets at runtime.

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

The first cohort follows the bootstrap, SMAC, and random-reserve schedule. At
each completed-cohort boundary, the top `--finalists` become retained elites.
The tuner begins the next challenger cohort only when the remaining tuning
budget funds all of its planned new pairs. Otherwise, it validates the latest
cohort's finalists.

## What is frozen

The manifest freezes the objective, panel, task corpora and prefixes, search
effort, proposal schedule, model versions, budgets, and derived objective epoch.
This makes the scientific result reproducible even when a run is resumed.

- `--seed` controls proposal streams.
- `--task-seed` controls the disjoint tuning and held-out validation corpora.
- Panel weights determine a deterministic weighted-fair task order. Each task
  records its opponent, configuration fingerprint, seed, and start stratum.
- `--constraint JSON` may be repeated to narrow the proposal space. It accepts
  either `[{"when"?: {...}, "set": {...}}]` or the shorthand
  `{name: {fix|range|choices}}`. A constraint can narrow the declared schema,
  but cannot widen it. It affects proposals, not the schema default or objective
  opponents.
- All task counts must be complete panel-weight cycles. `--tuning-pairs` is the
  largest tuning prefix: every accepted candidate reaches each cumulative,
  complete-cycle prefix before the full cohort is deepened.
- `--finalists` is both the retained-elite count and the final shortlist count.
  Validation always uses a leading prefix of the frozen production corpus.

## Budgets and failures

`--tuning-pair-budget` and `--validation-pair-budget` are required total budgets
for pair attempts, frozen under `safe-boundary-pair-attempts-v1`. The initial
cohort always runs. The tuner admits a later cohort only when the remaining
tuning budget covers its planned pairs, counting every started attempt, even one
that fails or is interrupted. The number of cohorts is therefore an outcome of
the budget, not another setting.

The validation budget divides evenly over finalists. Each finalist receives the
same held-out prefix: `budget / finalists` pairs, at least one complete panel
cycle, and no more than `--production-validation-pairs`.

Budgets are soft caps at pair and cohort boundaries. The tuner never stops
between the two seats of a pair or inside an admitted cohort, so a retry may
exceed the declared cap. The compute ledger reports the actual attempts,
completed pairs, failures, censored starts, games, iterations, wall time, and
budget overrun or remainder.

`--evaluator-workers` is operational rather than scientific and defaults to
`1`. Each evaluator runs one search thread, so the value cannot exceed the
available logical CPUs. Higher values start allocator-ordered, seat-swapped pair
subprocesses concurrently, while terminal evidence is still committed in the
same canonical order as a sequential run. A resumed run may use a different
worker count.

Tuning pairs retry once after a recorded failure. After two started but
incomplete attempts, the frozen `terminal-candidate-refill-v1` policy records
`candidate_failed`, preserves the candidate's factual work, removes it from the
live cohort, and refills the vacancy from the scheduled source, or from
`random_reserve` after the schedule ends. Validation failures require an
explicit resume and never replace a finalist. Interrupting a run cancels active
children; uncommitted starts are censored and count toward the same two-attempt
limit when resumed.

## Screening and active elimination

At a complete, non-final tuning prefix of at least 12 pairs, the tuner can
record a deterministic, paired, stratum-aware `shadow_race_decided` disposition.
The minimum, practical margin, and nominal threshold are frozen in the
manifest. Shadow evidence does not prune candidates: each still reaches the
maximum tuning prefix. A run without an eligible prefix is valid and records no
shadow decision.

`--shadow-policy {paired_bootstrap,successive_halving}` chooses the frozen
screening policy and is resume-sensitive.

- `paired_bootstrap` is the default stratum-aware bootstrap policy.
- `successive_halving` ranks surviving candidates at each eligible common prefix
  by point estimate, breaking ties by fingerprint. It begins with the full cohort
  roster and applies its earlier batches before each ranking. It keeps
  `max(finalists, ceil(survivors / 2), retained elites)`, and marks the rest as
  eliminated. Retained elites are always protected.
- With a positive `--shadow-halving-spare-margin`, a candidate within that
  paired-mean margin of the last kept candidate continues to the next look.
  A margin of `0.0` is the plain eta-2 cut. This policy makes no confidence
  claim; `--shadow-practical-margin` defines only the audit's recovered
  boundary, and cannot be paired with an explicitly non-default paired
  threshold.

`--active-elimination-audit-probability` activates allocation-validation mode
only for a finite value strictly between zero and one. The tuner then records an
`allocation_decided` batch at each eligible decision: candidates are pruned or
deterministically continued as audits. `paired_bootstrap` accepts any setting.
`successive_halving` requires a positive `--shadow-halving-spare-margin`, using
`successive-halving-spare-near-tie-v1`; the plain eta-2 cut remains shadow-only.

The active specification binds the policy, method version, and spare margin, so
a resume cannot combine an audit with a different decision policy. Audits and
their boundaries remain through the maximum prefix, and pruned candidates are
not replaced within a cohort. If an audited candidate reaches its exact recorded
boundary candidate at maximum tuning fidelity, later active pruning suspends;
shadow decisions and full-cohort tuning continue.

For paired decisions, `decision_margin` records the threshold, favorable
probability, and their difference. For rank decisions, it records rank, target
survivor count, ranks below the cutoff, and the spared-candidate count.

## Reports and validation

Each run directory contains three version-4 artifacts:

- `manifest.json` is the frozen run specification.
- `evidence.jsonl` is append-only proposal, pair-atomic observation, selection,
  and completion evidence.
- `report.json` is a replaceable projection with proposal provenance, weighted
  held-out marginals, opponent matchups, finalist differences, unresolved ties,
  and the compute ledger.

`report.json` also projects candidate failures, replacements, and screening:

- `candidate_lifecycle` records policy outcomes, terminal failures, and
  replacement lineage.
- `shadow_elimination` compares candidates with the same cohort's maximum-prefix
  top set. Calibration and stratum reversals use the exact early boundary
  candidate. It never uses held-out validation and is not an anytime-valid
  safety guarantee.
- Paired looks include calibration and Brier score. Successive-halving looks
  include rank and survivor counts, with calibration fields marked not
  applicable.
- Active runs also have `active_elimination`, which distinguishes planned unique
  pair savings from factual compute. `gross_nominal_suffix_unique_pairs` is the
  suffix after every first nominal elimination;
  `audit_continuation_suffix_unique_pairs` restricts that to audits; and
  `planned_unique_pair_savings` is the difference. This prefix arithmetic
  excludes retries, failures, and wall time; those belong to the compute ledger.

`top_set_false_elimination_rate` uses eligible, unprotected top-set paths as its
denominator. `trash_precision` uses counterfactual eliminations and calls only
candidates outside that top set `trash`. Avoided work is factual suffix work
after the first unprotected elimination, including retries and partial failures.
Calibration uses fixed probability bins and Brier score only for looks an active
path would reach.

Validation is `production` only when the selected validation prefix is the whole
production corpus and its search effort equals the declared production effort.
Every other result is `mechanics_smoke`, with the lower axis or axes named in
the report.

## Diagnostic matchups

`--diagnostic-pair-budget` defaults to zero. A positive value permits direct,
seat-swapped candidate matchups after the final affordable cohort and before
finalist selection. These pairs use frozen tuning effort and a deterministic
graph policy, but never affect objective observations, proposal costs,
elimination, held-out estimates, or deployment claims.

The report exposes a separate compute bucket and direct-matchup graph. A 95%
Hoeffding interval must establish every edge in a directed cycle before one
cycle-connected candidate outside the objective shortlist can take the final
validation slot. The objective winner remains. Direct-edge intervals are
per-edge, not graph-wide multiplicity corrected.

## Resume a run

Resume with the same scientific options and objective file:

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

Before append, resume validates the manifest and the complete evidence log. It
rejects changes to objective content, order, weights, configurations, task
corpora, prefixes, effort, budgets, or epoch. Objective and binary paths may
move when their resolved scientific identity is unchanged.

`--pair-timeout-seconds` and `--evaluator-workers` remain operational. Resuming
a completed run only rebuilds `report.json`. The scientific projection,
selection, and validation match an uninterrupted run; the ledger retains any
extra censored or retried attempts and resulting budget overrun.

## Proposer bake-off

`--proposer-policy` selects a frozen whole-run proposal policy. The default,
`smac_mixed`, keeps the SMAC-guided schedule. The measured alternatives are
`random`, `qmc` (scrambled Sobol), and `irace_generational` (a stateless,
elite-centred baseline).

```bash
uv run --project tuner tuner-proposer-bakeoff \
  --spec /tmp/druid-proposer-bakeoff.json \
  --experiment-dir /tmp/druid-proposer-bakeoff
```

The version-1 spec fixes policy order as `random`, `qmc`, `smac_mixed`, and
`irace_generational`, along with at least four proposal seeds, increasing tuning
budgets, the task seed, objective, and shared run settings. The experiment has
an immutable `experiment.json`, replayable child runs, and a replaceable
`results.json`. `--resume` completes unfinished children through the usual
foreground evidence path, then rebuilds the result projection.

## Elimination bake-off

`tuner-elimination-bakeoff` compares complete elimination systems at equal
declared compute. For each `(tuning pair budget, proposal seed)`, it creates
three matched child runs that differ only in elimination policy:

- `no_elimination` records paired shadow evidence but never enforces it.
- `paired_elimination` enforces the all-strata audited paired policy at audit
  probability `0.25`.
- `spare_near_tie` enforces the audited spare-near-tie successive-halving policy
  (`successive-halving-spare-near-tie-v1`, spare margin `0.10`) at the same
  audit probability.

```bash
uv run --project tuner tuner-elimination-bakeoff \
  --spec /tmp/druid-elimination-bakeoff.json \
  --experiment-dir /tmp/druid-elimination-bakeoff
```

The version-1 spec fixes the policies in that order, the `smac_mixed` proposer,
at least four distinct proposal seeds, at least two increasing tuning budgets,
zero diagnostic budget, full production validation, and the authorization block
for `successive-halving-spare-near-tie-v1`. `results.json` reports held-out
quality, simple regret, top-set recall against a union-of-returned-finalists
reference set, seed-paired contrasts, active-safety summaries, and budget
reinvestment against continue-all. Its largest-budget rule emits exactly
`keep_paired_elimination`, `change_to_spare_near_tie`, or
`reject_active_elimination`.

An active arm is `safe_in_bakeoff` only when every completed cell has zero
audited boundary reversals and no suspension. That is evidence from a finite
bake-off, not a universal safety guarantee, and it does not alter the normal
tuner default.

Here is a small smoke specification. It covers mechanics, replay, accounting,
and result projection, rather than production-quality evidence.

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
    "constraints": [{"set": {"algorithm": {"choices": ["mcts", "bandit"]}}}],
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

A production-equivalent experiment uses the full cohort, finalist, and
validation counts; production search effort for all three phases; at least four
seeds; and increasing budgets that admit several cohorts. Its experiment and
child artifacts are kept outside the repository. As with proposer bake-offs,
the directory contains immutable `experiment.json`, replayable child runs, and
replaceable `results.json`; `--resume` finishes incomplete children and rebuilds
the projection byte-identically.
