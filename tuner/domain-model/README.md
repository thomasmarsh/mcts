# tuner-domain-model

A Haskell domain model for the model-guided paired-racing game-strategy tuner.
Types and function signatures only — no implementation.

## Module summaries

| Module | Purpose |
|---|---|
| `Json` | Canonical JSON value used at transport and fingerprinting boundaries (integers and floats kept distinct) |
| `Identity` | Deterministic identities and fingerprints — candidates, panels, tasks, prefixes, epochs, observations, pair/game/diagnostic IDs |
| `Effort` | `SearchEffort` (iterations or time per move) with strict construction, comparison, and transport codecs |
| `ConfigSpace` | The abstract typed conditional space — parameter kinds, activation conditions, forbidden combinations, relational constraints, canonicalization |
| `Schema` | The concrete game-host schema — `GameSpec`, `TuningSchema`, `ParameterSpec`, `ActivationCondition`, and the ConfigSpace bridge |
| `FamilyExclusions` | Frozen named-family exclusion policy over the schema and candidate validation |
| `Deployment` | The optimization target — `DeploymentCase`, `DeploymentDistribution`, `Opponent`/`OpponentPanel`, `ObjectiveEpoch`, `ResolvedObjective`, `TuningObjective` |
| `Statistics` | Pair-level scoring and conservative comparison rules — Hoeffding intervals, paired differences, percentile bootstrap |
| `Evaluation` | The atomic evidence unit — `TaskCase`, `TaskCorpus`, `TaskPrefix`, `PairTask`, `GameResult`/`PairResult`, diagnostic pair tasks/results |
| `Candidate` | Immutable canonical candidates — `Candidate`, `ObservationFrontier`, proposal provenance, validation, terminal failures |
| `Evidence` | Aggregated observations — `Observation`, `ObservationContext`, `Estimate`, `TaskCountFidelity` |
| `Observations` | Observation construction and the comparability guard over epoch/phase/prefix/effort |
| `Selection` | Deterministic finalist selection and the cycle-aware validation shortlist |
| `Proposal` | Model-guided proposal — `ProposalSource`, `ModelObservation`, `ProposalRequest`, `ModelProposer`, source schedules |
| `Tasks` | Weighted-fair task corpora and cumulative tuning blocks |
| `Shadow` | Evidence-only shadow race decisions — paired bootstrap and successive-halving evidence, frozen method versions |
| `Elimination` | Active-elimination types — typed decision margins, `ApplyElimination`, `SuspendActiveElimination`, audited boundary reversals |
| `ActiveElimination` | Deterministic enforced-elimination sampling and audited-reversal recovery |
| `Diagnostic` | The direct candidate-vs-candidate matchup graph and cycle detection |
| `DiagnosticMatchmaking` | Deterministic choice of the next diagnostic pair to allocate |
| `Racing` | Iterated racing — `Cohort`/`CohortRecord`, `ReplayState`, `ResourceAllocation`, `AllocationDecision`, compute ledger |
| `Allocation` | The single authoritative allocator — decide, translate, and query ready work |
| `Ranking` | Production output — `RankedSet`/`RankedEntry`, `MatchupMatrix`, `TuningResult` |
| `Artifacts` | The frozen manifest and all policy specifications |
| `EventPayloads` | The closed tagged union of evidence event payloads and the append-only envelope |
| `ShadowAudit` | Same-run maximum-prefix counterfactual audit of recorded shadow decisions |
| `OpponentInteractions` | Candidate-by-opponent response matrix and ranking-reversal detection |
| `Target` | The game-binary boundary and bounded pair execution outcomes |
| `Bakeoff` | Proposer and elimination bake-off specifications and child facts |
| `Mechanism` | Druid-realism calibration and the shadow-race mechanism sweep |

## Core invariants (from the north-star)

1. **No score comparison crosses objective epochs, search budgets, or unequal task prefixes.** `ObservationContext` and `Observations.comparable` enforce this structurally.
2. **The atomic unit of evidence is a seat-swapped pair.** `PairResult` contains exactly two `GameResult`s, candidate plays first then second; `pairUtility` is win=1, draw=0.5, loss=0 per seat, averaged.
3. **Races use common nested task blocks.** All active candidates complete the same `TaskPrefix` before any elimination or deepening decision.
4. **One allocator controls all resource axes.** `AllocationDecision` covers introduce/deepen/refine/validate and the same `Manifest`-aware allocator produces every `ResourceAllocation`.
5. **Candidates are immutable and canonicalized.** `mkCandidate`/`candidateFromConfig` canonicalize before creating a `Candidate`; the fingerprint never changes.
6. **The ranked output carries uncertainty and may declare ties.** `RankedEntry` has an `Estimate`, a top-K probability, and a `reTiedWith` list.
7. **Family-level quotas do not exist.** The proposal policy assigns sources per slot without per-family allocation; only named-family exclusions are supported.
8. **Pruning is validated against deeper counterfactuals.** Shadow races record evidence-only decisions; `ShadowAudit` and `ActiveElimination` measure false elimination and audited boundary reversals.

## Relationship to the Python implementation

| Haskell module | Python modules |
|---|---|
| `Json` | `codec.py` |
| `Identity` | `identity.py` |
| `Effort` | `effort.py` |
| `ConfigSpace` | `space.py` (abstract conditional space) |
| `Schema` | `schema.py`, `space.py` |
| `FamilyExclusions` | `family_exclusions.py` |
| `Deployment` | `objective.py`, `target.py` (opponent/panel parsing) |
| `Statistics` | `statistics.py` |
| `Evaluation` | `domain.py` (`PairTask`, `GameResult`, `PairResult`), `target.py` (wire decode) |
| `Candidate` | `domain.py` (`Candidate`), `identity.py`, `family_exclusions.py` |
| `Evidence` | `domain.py` (`Observation`, `ObservationContext`), `evidence.py` |
| `Observations` | `observations.py` |
| `Selection` | `selection.py` |
| `Proposal` | `proposer.py`, `cohort.py` (proposal creation), `smac_proposer.py` |
| `Tasks` | `tasks.py` |
| `Shadow` | `shadow.py`, `successive_halving.py`, `race_policy.py` |
| `Elimination` | `elimination.py` (types), `domain.py` (margins) |
| `ActiveElimination` | `elimination.py`, `active_audit.py` |
| `Diagnostic` | `diagnostic_graph.py` |
| `DiagnosticMatchmaking` | `diagnostic_matchmaking.py` |
| `Racing` | `cohort.py`, `continuation.py`, `domain.py` (replay state) |
| `Allocation` | `allocator.py`, `cohort.py` |
| `Ranking` | `report.py`, `domain.py` (artifact types) |
| `Artifacts` | `artifacts.py` |
| `EventPayloads` | `event_payloads.py`, `evidence.py` |
| `ShadowAudit` | `shadow_audit.py` |
| `OpponentInteractions` | `opponent_interactions.py` |
| `Target` | `target.py`, `executor.py` |
| `Bakeoff` | `bakeoff_artifacts.py`, `bakeoff_metrics.py`, `elimination_bakeoff.py`, `elimination_bakeoff_metrics.py` |
| `Mechanism` | `mechanism_calibration.py`, `mechanism_sim.py`, `mechanism_sweep.py` |

## Building and exploring

```sh
# Build the library (type-checks all modules)
cabal build

# Load into GHCi to explore types
cabal repl
```

All function bodies are `= undefined` — this is intentionally a type-level model.
