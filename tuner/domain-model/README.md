# tuner-domain-model

A Haskell domain model for the model-guided paired-racing game-strategy tuner.
Types and function signatures only — no implementation.

## Module dependency graph

```
ConfigSpace   Deployment
     \           /   \
      \         /     \
    Candidate  /   Evaluation
        \     /    /    \
      Proposal  /  Evidence
           \   /   /
          Racing  /
              \  /
           Allocation
                |
            Ranking
```


## Module summaries

| Module | Purpose |
|---|---|
| `ConfigSpace` | Typed, conditional parameter spaces — categorical, integer, numeric, Boolean; conjunctions/disjunctions of activation conditions; forbidden combinations and relational constraints |
| `Deployment` | The optimization target — `DeploymentCase` (game config, opening, opponent, seed), `DeploymentDistribution`, `Opponent`/`OpponentPanel`, `ObjectiveEpoch`, `SearchEffort` |
| `Evaluation` | The atomic evidence unit — `TaskCase`, `TaskCorpus`, `TaskPrefix` (common task blocks), `PairTask` (seat-swapped pair), `GameResult`/`PairResult`, `pairUtility` (seat-balanced score) |
| `Evidence` | Aggregated observations — `Observation` (comparable only within same epoch/phase/prefix/effort), `ObservationContext`, `Estimate` (mean with Hoeffding interval), `ObservationFrontier` (visible to proposer) |
| `Candidate` | Immutable canonical candidates — `Candidate` (id + fingerprint + config), `CandidateLineage` (provenance), `CandidateValidation`, `CandidateFailure` |
| `Proposal` | Model-guided proposal — `ProposalSource` (schema default, bootstrap random, SMAC model, random reserve, QMC), `ProposalProvenance`, `Proposal`, `ModelProposer m` (ask/tell interface in monad `m`), `proposalPolicy` |
| `Racing` | Iterated racing — `Cohort`/`CohortRecord`, `RacingState` (cohorts, active candidates, elites, finalists), `deepen`, `eliminate` (evidence-based with practical-effect margin), `completeCohort`, `startNextCohort`, `selectFinalists`, `promotionBoundary` |
| `Allocation` | The single authoritative allocator — `AllocationDecision` (introduce/deepen/refine), `ResourceAllocation`, `ComputeBudget`/`ComputeLedger`, `allocate`, `nextReadyPair` |
| `Ranking` | Production output — `RankedSet`/`RankedEntry` with score, uncertainty, top-K probability, ties; `MatchupMatrix`, `detectCycles`, `TuningResult` (the complete artifact) |

## Core invariants (from the north-star)

1. **No score comparison crosses objective epochs, search budgets, or unequal task prefixes.** `ObservationContext` enforces this structurally — `comparable` checks all fields.
2. **The atomic unit of evidence is a seat-swapped pair.** `PairResult` contains exactly two `GameResult`s, candidate plays first then second. `pairUtility` is win=1, draw=0.5, loss=0 per seat, averaged.
3. **Races use common nested task blocks.** All active candidates complete the same `TaskPrefix` before any elimination or deepening decision.
4. **One allocator controls all resource axes.** `AllocationDecision` covers exactly three uses: introduce, deepen, refine ranking.
5. **Candidates are immutable and canonicalized.** `mkCandidate` validates and canonicalizes before creating a `Candidate`; the fingerprint never changes.
6. **The ranked output carries uncertainty and may declare ties.** `RankedEntry` has an `Estimate`, a top-K probability, and a `reTiedWith` list.
7. **Family-level quotas do not exist.** The proposal policy assigns sources per slot without per-family allocation.

## Relationship to the Python implementation

| Haskell module | Python modules |
|---|---|
| `ConfigSpace` | `space.py`, `schema.py` |
| `Deployment` | `objective.py`, `target.py` |
| `Evaluation` | `domain.py` (`PairTask`, `GameResult`, `PairResult`), `target.py` (parse) |
| `Evidence` | `domain.py` (`Observation`, `Estimate`), `observations.py`, `statistics.py` |
| `Candidate` | `domain.py` (`Candidate`), `identity.py`, `family_exclusions.py` |
| `Proposal` | `proposer.py`, `cohort.py` (proposal creation), `smac_proposer.py` |
| `Racing` | `cohort.py`, `allocator.py`, `continuation.py`, `selection.py` |
| `Allocation` | `allocator.py`, `domain.py` (`AllocationDecision`, `ResourceAllocation`) |
| `Ranking` | `report.py`, `domain.py` (artifact types) |

## Building and exploring

```sh
# Build the library (type-checks all modules)
cabal build

# Load into GHCi to explore types
cabal repl
```

All function bodies are `= undefined` — this is intentionally a type-level model.
