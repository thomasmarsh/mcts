# tuner-domain-model: a tutorial

This document is for someone who has never seen the tuner before. It explains,
in plain language, what the tuner *is*, why it is shaped the way it is, and how
the types in this package fit together. It assumes you can read a Haskell `data`
declaration, but not that you know anything about game tuning, racing, or the
rest of this repository.

The `README.md` next to this file is a reference: a table of every module and a
mapping to the Python implementation. This document is the story. Read this
first; reach for the README when you need to find a specific type.

---

## 1. What the tuner does

Imagine you have a game-playing program — a search-based AI whose behavior is
controlled by a few dozen knobs: an exploration constant, an iteration budget, a
threshold for when to widen the search, which heuristic to use, and so on. Some
knobs only exist when other knobs are set a certain way (there is no "width" to
tune if you've disabled widening entirely). Some combinations are forbidden, and
several different-looking settings can mean the same thing.

You want to find the configuration that *wins the most games* — not against one
fixed opponent you happened to test with, but against the realistic mix of
opponents, starting positions, and rule sets you actually care about. And you
have a finite budget of compute to spend finding it.

The tuner's job, in one sentence:

> Given a description of the configuration space and a frozen definition of
> "winning," spend a compute budget to produce a **ranked set of
> configurations**, each with an estimated win rate, an uncertainty interval,
> and the full trail of evidence behind it.

### Why this is hard

- **The space is huge and conditional.** Brute force is impossible, and many
  entries in the space aren't even legal most of the time.
- **Win rate is noisy.** A configuration that is truly better may lose any
  particular game, so you need many games to tell two candidates apart.
- **First-player advantage exists.** Whoever moves first in many games has a
  systematic edge, so a single game "A beat B" is not a fair measurement.
- **Opponents matter.** A configuration can farm one weak opponent and fail
  against everyone else.
- **Non-transitivity is real.** In strategy games, "A beats B, B beats C" does
  not always imply "A beats C." A single number can hide rock-paper-scissors.
- **Cheap measurements mislead.** Evaluating candidates with fewer games or
  shallower search is faster, but the cheap answer doesn't always agree with
  the expensive answer.

The design of the tuner — and therefore of this domain model — is an explicit
response to each of these problems.

## 2. What this repository actually is

`tuner-domain-model` is a Haskell package containing **types and function
signatures only**. Every function body is `= undefined`. It is a *model*: the
skeleton of the real (Python) tuner, expressed as Haskell data types so that
the important invariants can be pinned down structurally rather than described
in prose.

Why write a package whose functions don't do anything?

- **It is the source of truth for the vocabulary.** When the Python code and a
  design discussion use words like "observation," "prefix," or "cohort," the
  Haskell type is what those words are formally referring to.
- **It makes the invariants checkable.** "No score comparison crosses objective
  epochs or unequal task prefixes" is not just a sentence; it is the fact that
  `Observations.comparable` returns an error unless two observations share a
  context. A type signature can't prove the body (there is no body), but it
  forces the *shape* of the guarantee to be explicit.
- **It compiles.** `cabal build` type-checks all thirty modules, so the model
  can't drift into incoherence the way an untyped diagram can.

A tiny handful of pure helpers (e.g. `pairUtility`, `mkSearchEffort`,
`isScalar`) are actually implemented because they are arithmetic facts, not
behavior. Everything else is a signature.

## 3. The big picture

The tuner is a pipeline:

```text
conditional configuration space
            │
            ▼
global model-guided proposals
            │
            ▼
paired, instance-matched races with adaptive resource allocation
            │
            ▼
held-out production-budget validation
            │
            ▼
ranked set with uncertainty and complete evidence
```

Read top to bottom:

1. **Configuration space.** The game binary tells you what knobs exist, which
   depend on which, and what's forbidden (`ConfigSpace`, `Schema`).
2. **Proposals.** A model suggests new candidate configurations worth trying
   (`Candidate`, `Proposal`).
3. **Racing.** Candidates are evaluated against the same sequence of opponents
   and positions ("tasks"), and the weak ones are progressively eliminated
   while the survivors get more evidence (`Evaluation`, `Racing`, `Allocation`).
4. **Validation.** The few finalists are re-measured on a fresh, held-out set of
   games at production quality (`Selection`, `Ranking`).
5. **Ranked output.** You get a list, with uncertainty, and ties are allowed.

The rest of this document unpacks each stage.

## 4. A running example

To make the rest concrete, keep this tiny scenario in mind. It is deliberately
smaller than any real run.

- The game is **Tak** (pick any game you like; the tuner doesn't care).
- The configuration space has three knobs: a `family` choice
  (`meta_mcts` or `mcts`), an `exploration` float, and an `iteration_budget`
  integer that only exists when `family = mcts`.
- The frozen objective says "win rate against two opponents: `weak-baseline`
  (weight 1) and `strong-baseline` (weight 2)."
- A tuning prefix is **3 tasks** (one weighted cycle: `weak`, `strong`,
  `strong`). Full tuning depth is **6 tasks** (two cycles).

Throughout, we'll follow three candidates, **A**, **B**, and **C**, from
proposal to the final ranking.

## 5. The concepts

### 5.1 Configuration → candidate

A **configuration** is a set of `parameter = value` assignments, e.g.
`{ family: mcts, exploration: 0.7, iteration_budget: 200 }`.

A **candidate** (`Candidate.Candidate`) is what a configuration becomes once it
has been:

- **validated** — it satisfies every activation condition and violates no
  forbidden combination or relational constraint (`ConfigSpace.validate`);
- **canonicalized** — normalized to one unambiguous form so that two settings
  that mean the same thing produce the same candidate
  (`ConfigSpace.canonicalize`);
- **fingerprinted** — given a stable ID and a SHA-256 digest of its canonical
  form (`Identity.candidateFromConfig`).

```haskell
data Candidate = Candidate
  { candId             :: String
  , candFingerprint    :: String
  , candCanonicalConfig :: CanonicalConfig
  }
```

The crucial property is **immutability and identity**. Re-evaluating candidate A
at a new seed, a new opponent, or a new fidelity *adds evidence to the same
candidate*; it never creates a new, unrelated trial. Every later record (a
pair, an observation, an elimination) points at A's ID and fingerprint, so the
whole history of A is joinable.

> In our example, "A" is one immutable candidate. Whether we play it against
> `weak-baseline` or `strong-baseline`, it is still A.

### 5.2 The seat-swapped pair: the atom of evidence

The smallest unit of evidence is not a game. It is a **pair**: two games against
the *same opponent at the same position and seed*, with the candidate playing
**first in one game and second in the other**.

This is `Evaluation.PairResult`:

```haskell
data PairResult = PairResult
  { prTask  :: PairTask
  , prGames :: (GameResult, GameResult)  -- candidate first, then second
  }
```

Why two games with swapped seats? Because of first-player advantage. If the
candidate always played first, a genuinely-strong candidate and a merely-lucky
first-mover would be indistinguishable. Scoring each seat separately and
averaging cancels the positional bias.

Each individual game is scored as win = 1, draw = 0.5, loss = 0
(`Statistics.gameUtility`), and the pair utility is the average of the two seats
(`Statistics.pairUtility`):

```haskell
pairUtility first second = (first + second) / 2.0
```

So the candidate's score for a pair is a number in `[0, 1]` that is already
seat-balanced. The raw outcomes, seeds, positions, and configurations are kept
alongside (`GameResult`), so nothing is lost in the averaging.

> A beats `weak-baseline` 2–0 (pair utility 1.0), draws `strong-baseline` 1–1
> (pair utility 0.5). These two pair utilities are A's raw evidence so far.

**Nothing may be decided between the two halves of a pair.** No elimination, no
allocation, no "we'll skip the second seat because the first was a blowout." The
pair is indivisible.

### 5.3 Task cases, corpora, prefixes: everyone runs the same course

A **task case** (`Evaluation.TaskCase`) is one concrete paired comparison to
play: it names the phase, a seed, the opponent, the opponent's fingerprint, the
panel fingerprint, and a start position.

```haskell
data TaskCase = TaskCase
  { tcTaskId, tcStratumId, tcOpponentId, ... :: ...
  , tcSeed, tcOrdinal :: ...
  }
```

A **task corpus** (`Evaluation.TaskCorpus`) is an *ordered* list of task cases
for one phase, with its own fingerprint. A **task prefix** (`TaskPrefix`) is
just the first N cases of a corpus.

The design rule that matters most:

> **Within a race, every active candidate plays the same tasks in the same
> order.**

If A played only `weak-baseline` while B played only `strong-baseline`, their
scores would be incomparable — you'd be comparing easy games to hard games.
Instead, all candidates complete the *same prefix* before anyone is compared.
Reaching "depth 3" means "completed the same first 3 tasks as everyone else."
The deeper prefixes are cumulative: depth 6 includes the depth-3 tasks plus
three more.

This is the "common nested task blocks" idea. It is what makes paired
*differences* between candidates meaningful: the only thing that varies across
candidates is the candidate itself.

The block order is also **stratified** so every prefix is representative of the
target distribution, and **weighted-fair** (`Tasks.weightedSchedule`) so that a
panel with weights `{weak: 1, strong: 2}` is sampled in that proportion rather
than round-robin.

> In our example, one full weighted cycle is 3 tasks: `weak, strong, strong`.
> A prefix is 3 tasks: `weak, strong, strong`; full depth 6 is
> `weak, strong, strong, weak, strong, strong`. A, B, and C all play exactly this
> sequence.

### 5.4 What "winning" means: opponents, panels, epochs, and effort

"Win rate" is only meaningful relative to a frozen definition of *against whom,
from what positions, under which rules*. The tuner makes this explicit.

- **Opponent** (`Deployment.Opponent`): one external player — its canonical
  config, fingerprint, role (`Default` or `HistoricalReference`), and a weight.
- **Opponent panel** (`Deployment.OpponentPanel`): a frozen, weighted set of
  opponents. This is the reference field of competition during tuning.
- **Deployment distribution** (`Deployment.DeploymentDistribution`): a versioned
  distribution over full *deployment cases* — game config, opening, opponent,
  seed, rules, adjudication. This is the true target; the panel is a practical
  stand-in for it.
- **Objective epoch** (`Deployment.ObjectiveEpoch`): a frozen reference frame.
  When the panel changes (say a new frontier opponent is added), a new epoch
  begins, and old scores are *not* treated as measurements of the new
  objective.

Finally, **search effort** (`Effort.SearchEffort`) is how hard the candidate
thinks per move — either a number of iterations or a time budget:

```haskell
data SearchEffort = SearchEffort
  { effortKind  :: EffortKind   -- Iterations | TimeMs
  , effortValue :: Int
  }
```

Effort is part of the measurement context. A candidate evaluated at 16
iterations and one evaluated at 64 iterations are not the same measurement,
even if both are "A."

> Our example's objective is the panel `{weak: 1, strong: 2}`, frozen as epoch
> `v1`, at some fixed effort. That is the definition of "winning" for the whole
> run.

### 5.5 Observations and the comparability rule

An **observation** (`Evidence.Observation`) is aggregated evidence for one
candidate at one context:

```haskell
data Observation = Observation
  { obsId            :: String
  , obsCandidateId   :: String
  , obsContext       :: ObservationContext
  , obsPairUtilities :: [Double]      -- one number per completed pair
  , obsEstimate      :: Estimate      -- mean + Hoeffding bounds
  }
```

The **context** (`Evidence.ObservationContext`) is the four-way key that pins
down *what* was measured:

```haskell
data ObservationContext = ObservationContext
  { ocObjectiveEpochId :: String
  , ocPhase            :: Phase          -- Tuning | Validation
  , ocTaskPrefix       :: TaskPrefix
  , ocSearchEffort     :: SearchEffort
  }
```

The single most important rule in the whole model:

> **Two observations may be compared only if they share the same epoch, phase,
> task prefix, and search effort.**

This is `Observations.comparable`, and it returns `Either String ()` — an
enforced error, not a convention:

```haskell
comparable :: Observation -> Observation -> Either String ()
```

You cannot accidentally rank a candidate measured on an old panel against one
measured on the current panel, or a 2-task observation against a 6-task
observation. The types make the mistake a type error (or at least an explicit
runtime failure) instead of a silent wrong answer.

The **estimate** (`Statistics.Estimate`) carries a mean plus a conservative
confidence interval (Hoeffding bounds at `alpha = 0.05`):

```haskell
data Estimate = Estimate { estMean, estLower, estUpper :: Double }
```

Two comparable observations can be turned into a **paired difference** estimate
(`Observations.pairedDifference`), and an estimate can be classified as
`Better | Worse | Tie` (`Statistics.tieRelation`). The tuner never invents a
precise ordering when the evidence supports only a tie.

### 5.6 The race: cohorts, elimination, deepening, elites

A **race** is an elimination tournament run on evidence, not on a bracket.

A **cohort** (`Racing.CohortRecord`) is the set of candidates competing in one
round. The race loop is:

1. **Form a cohort** — a mix of new model-guided proposals, a random/diversity
   reserve, and **retained elites** (the strongest survivors of previous
   cohorts).
2. **Run the common initial block** — every member completes the same minimum
   prefix.
3. **Eliminate only with evidence** — candidates with negligible probability of
   reaching the promotion boundary are pruned, accounting for a
   practical-effect margin (a difference too small to care about is not grounds
   for elimination).
4. **Deepen the survivors** — the smaller field gets larger common prefixes,
   increasing confidence as the field narrows.
5. **Repeat** — the survivors become elites, a new challenger cohort forms, and
   the cycle continues until the budget is exhausted.

The two anti-patterns this is designed against:

- **Quota-based elimination.** "Cut the bottom 50% every round" is *not* the
  rule. A geometric ratio is a schedule, not a discard quota. If the cohort is
  genuinely ambiguous, nobody is forced out.
- **Wasting elite evidence.** If an elite has already completed a pair at the
  exact same epoch, task, and effort, that result is reused — the tuner never
  pays to replay an identical pair. But elites *do* race every fresh shared
  block their challengers see.

> In our example, cohort 1 is `{A, B, C}`. After the 3-task prefix, suppose
> A and B are indistinguishable and C is clearly worse. C is pruned; A and
> B deepen to 6 tasks. A and B become the elites for cohort 2, which also admits
> two new proposals D and E.

### 5.7 One allocator to rule them all

At every scheduling boundary, exactly **one component** decides what to do next
(`Allocation`). Its decisions (`Racing.AllocationDecision`) are a small closed
set:

```haskell
data AllocationDecision
  = ResolveProposal      -- validate/canonicalize a new proposal
  | ExecutePair          -- run one pair task
  | ChooseDiagnosticPair -- resolve a matchup-graph ambiguity
  | EmitObservation      -- a candidate finished a prefix; record it
  | EmitShadowRace       -- record evidence-only pruning decisions
  | EnforceElimination   -- actively prune, if the policy allows
  | CompleteCohort / StartNextCohort
  | DeepenCohort         -- move survivors to the next common block
  | SelectFinalists
  | CompleteRun
  | ...
```

The point is the *single source of authority*: two racing policies never fight
over the same resource. The budget is **actual compute** (pair attempts,
completed pairs, search iterations, wall time — `Racing.ComputeLedger`), not a
count of configuration objects.

Each decision is translated into a concrete **resource allocation**
(`Racing.ResourceAllocation`) — the thing recorded in the evidence log — such
as `IntroduceCandidate`, `DeepenCohortAllocation`, or `RetainElites`.

The allocator's four big choices are: **introduce** a new candidate, **deepen**
an existing one, **diagnose** a non-transitivity ambiguity, or **validate** a
finalist. Those four uses of compute are the whole strategy space of the tuner.

### 5.8 Proposals and the global model

Where do new candidates come from? A **proposer** (`Proposal`).

The north-star proposer is a *global model* — a surrogate that predicts
production-quality performance (and its uncertainty) from every compatible
observation, while treating fidelity and task context explicitly. It balances
exploiting strong known regions, exploring uncertain ones, and maintaining a
random/low-discrepancy reserve so no region is permanently abandoned.

The model's interface is a pure ask:

```haskell
data ModelProposer = ModelProposer
  { mpAdapterVersion :: String
  , mpAsk            :: ProposalRequest -> ProposedConfiguration
  }
```

`ProposalRequest` carries the observations at a common frontier, the excluded
fingerprints, the attempt ordinal/seed, and the ranked parents. The
`ProposedConfiguration` it returns is wrapped in full provenance
(`Candidate.ProposalProvenance`): which source proposed it, the model version,
the acquisition value and predicted score at proposal time, and the parent
candidate if it was derived from one.

Every proposal also records the **observation frontier** visible when it was
made (`Candidate.ObservationFrontier`) — the exact epoch/prefix/effort/observations
the proposer could see. This is what makes a proposal *replayable*: you can
reconstruct what the model knew.

The fixed set of sources (`Candidate.ProposalSource`):

```haskell
data ProposalSource
  = SchemaDefault | BootstrapRandom | SmacModel | RandomReserve
  | RandomSearch | QmcSearch | IraceModel
```

There are no per-family quotas and no per-family elimination. `family` is just
one categorical variable in the schema; the model may learn `family × c`
interactions, but no mechanism reserves slots or budget per family. The only
family-level policy is **exclusion**: a frozen list of named families that may
not be proposed at all (`FamilyExclusions`).

> In our example, `meta_mcts` is excluded, so only `family = mcts` candidates
> are ever proposed. The proposer might learn that `exploration ≈ 0.7` does
> well and propose around it, while a random reserve keeps `exploration ≈ 0.3`
> on the table.

### 5.9 Pruning that can be trusted: shadow decisions and audits

Pruning (eliminating candidates early to save compute) is the most dangerous
part of the loop, because a cheap decision can throw away a strong candidate.
The design treats pruning as a *hypothesis to validate*, with three layers of
protection.

1. **Shadow decisions** (`Shadow`). The policy is asked what it *would* do from
   every prefix, but the decision is **recorded, not enforced**. Every candidate
   still runs to full depth. This produces the counterfactual ground truth:
   did the candidate the policy would have pruned turn out to be strong? A
   `ShadowRaceDecision` carries per-candidate dispositions
   (`Continue | Eliminate | Protected`) and the evidence behind each.

2. **Audited counterfactuals** (`ShadowAudit`). Recorded shadow decisions are
   labeled against the maximum-prefix evidence: false eliminations of eventual
   top-k candidates, boundary reversals (a pruned candidate that would have
   reached the promotion boundary), calibration bins, and per-stratum reversal
   summaries.

3. **Active elimination with a randomized audit** (`ActiveElimination`,
   `Elimination`). Only a policy that has *passed a preregistered gate* may
   enforce pruning. When it does, a predeclared random sample of prune decisions
   is overridden and continued to full depth, so the policy's mistakes can never
   become permanently invisible. If an audited continuation *reverses* a
   boundary decision, the active elimination can be suspended
   (`SuspendActiveElimination`).

Every enforced elimination carries a **typed decision margin**
(`Elimination.EliminationDecisionMargin`) — how far below the boundary the
candidate fell — so the decision is inspectable.

The practical upshot: in a default run, the pruning policy is usually running
*shadow-only*, and the run still evaluates every candidate to full tuning depth.
The shadow evidence is what would later justify (or refuse to justify) turning
pruning on.

### 5.10 Non-transitivity and diagnostics

A single skill number cannot represent rock-paper-scissors. The model keeps two
related but distinct views:

- **Opponent response matrix** (`OpponentInteractions`): for each candidate and
  each *external* opponent, the stationary estimate and pair count. Two
  candidates can be compared per opponent, exposing ranking *reversals* ("A
  beats weak but B beats strong").
- **Direct matchup graph** (`Diagnostic`): candidate-vs-candidate pairs,
  labeled separately from objective evidence, used to test for *material
  cycles* (`A > B > C > A`).

The key distinction: a response matrix against external opponents can show
interaction and reversal, but it **cannot by itself establish a cycle among the
candidates**. A cycle claim requires direct candidate-vs-candidate edges, which
is exactly what the diagnostic graph collects.

Diagnostics have their **own budget** (`Racing.ComputeBudget`'s
`cbDiagnosticPairAttempts`) and never leak into race observations, proposer
costs, or the deployment score. `DiagnosticMatchmaking.nextDiagnosticAllocation`
deterministically picks the next candidate-vs-candidate pair to play, driven by
reasons like `GraphConnectivity`, `PotentialCycleClosure`, or `RankingBoundary`.

### 5.11 The ranked output: uncertainty and ties

The final artifact (`Ranking`) is a **ranked set**, not a single winner:

```haskell
data RankedEntry = RankedEntry
  { reCandidate         :: Candidate
  , reDeploymentScore   :: Estimate
  , reTopKProbability   :: Double
  , rePairCount         :: Int
  , reOpponentCount     :: Int
  , reIsDistinguishable :: Bool
  , reTiedWith          :: [String]
  }
```

Each entry carries:

- the candidate itself;
- the deployment-score **estimate** with bounds;
- the probability it belongs to the top `k`;
- its evidence counts (pairs, opponents);
- whether it is **distinguishable** from its neighbors, and the list of
  candidates it is **practically tied** with.

The whole `TuningResult` adds the matchup matrix, detected cycles, the objective
epoch, the production budget, and total compute/wall time. The tuner **may
return ties**; it never fabricates a precise order the evidence can't support.

Crucially, the *ranking* comes from **held-out validation** — fresh games at
production budget that the proposer and allocator never saw. Tuning evidence
selects the shortlist; validation evidence produces the published claim. The
shortlist is drawn broadly enough to survive tuning noise
(`Selection.selectValidationShortlist`), including candidates with material
posterior probability of reaching the top `k` and structurally diverse
candidates when non-transitivity is suspected.

### 5.12 Replayability: the manifest and the evidence log

Two structures make an entire run reproducible from scratch.

- **The manifest** (`Artifacts.Manifest`) freezes *everything that configures a
  run*: the game spec, the objective, the panel, the corpora and prefixes, the
  epoch, the proposer spec, the shadow/elimination/diagnostic/failure policy
  specs, the effort values, and the compute budget. It carries a fingerprint.
- **The evidence log** (`EventPayloads`) is the append-only record of *what
  happened*: a sequence of `EvidenceEvent`s, each with a sequence number and a
  payload from a **closed tagged union** — proposals created/accepted/rejected,
  pairs started/completed/failed, cohorts completed, observations completed,
  allocations decided, shadow races decided, run interrupted/completed.

The two together are the difference between "we got these numbers" and "here is
every decision, every game, and the exact configuration that produced them."

`Target.Target` sits at the very bottom as the **game-binary boundary**: the
four operations the tuner can ask of a real executable (`describe`, `validate`,
`evaluate`, `cancel`). Everything above it is pure bookkeeping; everything below
it is the world.

## 6. The life of a candidate

Here is the same material as a narrative — the path of candidate A through the
modules.

1. **Proposal.** The allocator (`Allocation.decideAllocation`) decides it's time
   to introduce a candidate. The proposer (`Proposal.ModelProposer`) receives a
   `ProposalRequest` describing everything it may see, and returns a
   `ProposedConfiguration`. The tuner validates and canonicalizes it into a
   `Candidate` and records a `ProposalCreated`/`ProposalAccepted` event
   (`Candidate`, `ConfigSpace`, `EventPayloads`).

2. **Admission to a cohort.** A lands in a `CohortRecord` along with the other
   cohort members (`Racing`).

3. **Common block.** The allocator emits `ExecutePair` decisions; the corpus
   supplies the ordered `TaskCase`s, and A plays a seat-swapped `PairTask`
   against each opponent in the current `TaskPrefix` (`Evaluation`,
   `Target.Target`). Each pair yields a `PairResult` with two `GameResult`s, raw
   and timestamped.

4. **Observation.** When A finishes the prefix, its pair utilities are
   aggregated into an `Observation` tagged with the full `ObservationContext`
   (`Evidence`, `Observations.observation`), and an `ObservationCompleted` event
   is appended.

5. **Shadow look (evidence-only).** At an eligible prefix, the shadow policy
   computes what it *would* have done — `Eliminate` A, `Continue` A, or
   `Protected` — and records a `ShadowRaceDecision` without acting on it
   (`Shadow`). Later, `ShadowAudit` labels that decision against A's eventual
   maximum-depth evidence.

6. **Elimination or deepening.** If active elimination is enabled and A is
   pruned, an `ApplyElimination` is recorded (possibly with an audit
   continuation). Otherwise A is retained (`RetainElites`) and deepened to the
   next common block (`DeepenCohort`). This repeats until A is eliminated or
   the tuning budget runs out.

7. **Shortlist.** If A survives to the end (or is an elite with material top-k
   probability), `Selection` puts it on the validation shortlist.

8. **Held-out validation.** A is re-measured on the *fresh* production
   validation corpus at production effort — games the tuner's search never
   touched (`Selection`, `Allocation`).

9. **Ranking.** `Ranking.deploymentScores` turns A's validation observations
   into a `RankedEntry` with an `Estimate`, a top-k probability, and possibly a
   `reTiedWith` list. The final `TuningResult` carries A's entry alongside the
   matchup matrix, any detected cycles, and the total compute spent.

At every step, the decisions are recorded as `EvidenceEvent`s and the state is
checkpointed in `ReplayState` — so A's whole journey, from proposal to ranked
entry, can be replayed exactly.

## 7. A map of the modules

Grouped by concept (the README has the alphabetical-by-module reference table).

**Foundations**

| Module | What it is |
|---|---|
| `Json` | The canonical JSON value (`JsonValue`), with ints and floats kept distinct so fingerprints match the Python implementation byte-for-byte. |
| `Identity` | Deterministic IDs and fingerprints. Every `candidate`, `task`, `prefix`, `epoch`, `observation`, and game ID is a stable hash of a canonical payload. |
| `Effort` | `SearchEffort` (iterations or time per move), strictly constructed so zero/negative is impossible. |

**The configuration space**

| Module | What it is |
|---|---|
| `ConfigSpace` | The abstract typed conditional space: parameter kinds, activation conditions (`And`/`Or`/`Not`/`Equals`/`In`), relational constraints, log/linear scales, and `validate`/`canonicalize`. |
| `Schema` | The *concrete* game-host view: `GameSpec`, `TuningSchema`, `ParameterSpec`, `ActivationCondition`, and the bridge into `ConfigSpace`. |
| `FamilyExclusions` | The one family-level policy: a frozen list of named families that may not be proposed. |

**What "winning" means**

| Module | What it is |
|---|---|
| `Deployment` | `Opponent`, `OpponentPanel`, `ObjectiveEpoch`, `DeploymentCase`/`DeploymentDistribution`, and the `ResolvedObjective` that freezes them. |

**Measurement**

| Module | What it is |
|---|---|
| `Statistics` | `Utility`, `Estimate` (Hoeffding bounds at alpha 0.05), paired differences, percentile bootstrap, and `TieRelation`. |
| `Evaluation` | `TaskCase`/`TaskCorpus`/`TaskPrefix`, `PairTask`/`PairResult`, `GameResult`, and the diagnostic pair analogues. |
| `Evidence` | `Observation` + `ObservationContext` (epoch, phase, prefix, effort). |
| `Observations` | Construction, the `comparable` guard, and paired differences. |

**Candidates and proposals**

| Module | What it is |
|---|---|
| `Candidate` | The immutable `Candidate`, `ProposalSource`, `ProposalProvenance`, `ObservationFrontier`, `Proposal`, and terminal `CandidateFailure`. |
| `Proposal` | `ModelProposer`, `ProposalRequest`, `ProposedConfiguration`, and the source schedules. |

**The race machinery**

| Module | What it is |
|---|---|
| `Tasks` | Weighted-fair task schedules, corpus construction, and cumulative tuning blocks. |
| `Racing` | `CohortRecord`, `ComputeBudget`/`ComputeLedger`, `ResourceAllocation`, `AllocationDecision`, and the immutable `ReplayState`. |
| `Allocation` | The single allocator: `decideAllocation`, `resourceAllocation`, `readyPairs`, `candidateFailureDue`, and friends. |
| `Selection` | Finalist selection and the cycle-aware validation shortlist. |

**Pruning, and validating pruning**

| Module | What it is |
|---|---|
| `Shadow` | Evidence-only pruning decisions (paired bootstrap, successive halving), frozen method versions. |
| `Elimination` | Typed decision margins and `ApplyElimination`/`SuspendActiveElimination`. |
| `ActiveElimination` | Enforced pruning with randomized audit continuation and boundary-reversal recovery. |
| `ShadowAudit` | Labels recorded shadow decisions against maximum-prefix truth: false eliminations, reversals, calibration. |

**Non-transitivity**

| Module | What it is |
|---|---|
| `Diagnostic` | The direct candidate-vs-candidate matchup graph and cycle detection. |
| `DiagnosticMatchmaking` | Deterministic choice of the next diagnostic pair. |
| `OpponentInteractions` | Candidate-by-opponent response matrix and ranking-reversal detection. |

**Output**

| Module | What it is |
|---|---|
| `Ranking` | `RankedEntry`, `RankedSet`, `MatchupMatrix`, and the full `TuningResult`. |

**Freeze and replay**

| Module | What it is |
|---|---|
| `Artifacts` | The frozen `Manifest` and all policy specifications. |
| `EventPayloads` | The closed tagged union of evidence events and the append-only envelope. |
| `Target` | The game-binary boundary and bounded pair execution. |

**Experiments**

| Module | What it is |
|---|---|
| `Bakeoff` | Equal-compute proposer and elimination bake-off specifications and child facts. |
| `Mechanism` | Druid-realism calibration and the shadow-race mechanism sweep. |

## 8. How the pieces connect

A few signatures show the wiring explicitly.

The allocator is a pure function from state to decision:

```haskell
decideAllocation :: Manifest -> ReplayState -> AllocationDecision
```

A decision becomes a concrete, loggable allocation:

```haskell
resourceAllocation :: AllocationDecision -> Manifest -> ReplayState
                   -> Maybe ResourceAllocation
```

The pair executor only ever runs tasks the allocator chose:

```haskell
readyPairs :: Manifest -> ReplayState -> Maybe Int -> [PairTask]
```

Evidence flows from pairs to observations to estimates:

```haskell
contextualObservation :: Candidate -> ObservationContext -> [PairResult] -> Observation
pairedDifference      :: Observation -> Observation -> Estimate
```

Comparability is a typed gate, not a convention:

```haskell
comparable :: Observation -> Observation -> Either String ()
```

Proposals are an ask with full context:

```haskell
mpAsk :: ProposalRequest -> ProposedConfiguration
```

And the ranked output is assembled from validation estimates:

```haskell
deploymentScores :: [Candidate] -> [(Candidate, Estimate)] -> [RankedEntry]
rank             :: [RankedEntry] -> Double -> RankedSet
```

Reading these together, the whole system is a loop: `decideAllocation` inspects
`ReplayState`, picks an `AllocationDecision`, `resourceAllocation` records it,
`readyPairs` feeds the executor, `contextualObservation` turns results into
`Observation`s, `comparable`/`pairedDifference` gate and summarize comparisons,
and eventually `rank` turns the held-out validation into the `RankedSet`.

## 9. Exploring it yourself

```sh
# Type-check every module
cabal build

# Load the model into GHCi
cabal repl
```

Useful GHCi commands:

```haskell
:module + MyLib                     -- bring everything into scope
:browse Observations                -- every exported type and signature
:info ObservationContext            -- the four-way comparability key
:t comparable                       -- the type of the comparability gate
:t decideAllocation                 -- allocator: state -> decision
:t mpAsk                            -- proposer: request -> proposal
```

What you can and cannot do:

- **You can** inspect every type and signature, follow which fields a
  `ReplayState` carries, and confirm the model compiles as one coherent whole.
- **You cannot** run the tuner here: nearly every function body is
  `= undefined`, by design. Calling one will throw. The actual behavior lives in
  the Python implementation under `tuner/src/tuner_cli/`.

A few pure arithmetic helpers *are* implemented and callable (for example
`Statistics.pairUtility`, `Effort.mkSearchEffort`, `Json.isScalar`) — these are
facts, not behavior.

To see the model's vocabulary reflected in the running system, open
`tuner/src/tuner_cli/domain.py` and notice that its dataclasses mirror the
Haskell types here module-for-module (the README has the full mapping).

## 10. Glossary

- **Candidate** — an immutable, canonicalized, fingerprinted configuration.
- **Canonicalization** — normalizing a configuration to one unambiguous form so
  equivalent settings share an identity.
- **Fingerprint** — a SHA-256 digest of a canonical payload; the basis of
  identity.
- **Pair / seat-swapped pair** — two games against the same opponent at the
  same position, candidate first then second; the atomic unit of evidence.
- **Task case** — one concrete paired comparison (opponent, seed, position).
- **Task corpus / prefix** — an ordered list of task cases / its first N cases.
- **Common block** — a prefix every active candidate in a race completes before
  comparison.
- **Panel** — a frozen, weighted set of opponents.
- **Objective epoch** — a frozen reference frame; scores from different epochs
  are not compared.
- **Search effort** — iterations or time per move; part of measurement context.
- **Observation** — aggregated pair utilities for one candidate at one context.
- **Comparability** — the rule that two observations share epoch, phase,
  prefix, and effort before they may be compared.
- **Cohort** — the set of candidates racing together in one round.
- **Elite** — a survivor retained across cohorts.
- **Elimination** — evidence-based pruning of candidates that cannot reach the
  promotion boundary.
- **Shadow decision** — a pruning decision recorded but not enforced.
- **Active elimination** — enforced pruning, subject to randomized audit.
- **Audit continuation** — running a pruned candidate to full depth to check
  the decision.
- **Diagnostic** — candidate-vs-candidate evidence for non-transitivity.
- **Deployment distribution** — the versioned distribution of cases that
  defines the true objective.
- **Validation** — held-out measurement at production budget that produces the
  final ranking.
- **Manifest** — the frozen record of everything that configures a run.
- **Evidence log** — the append-only, replayable record of everything that
  happened.

## 11. Where to go next

- `README.md` — the per-module reference and the Haskell↔Python mapping.
- `../README.md` — how to actually *run* the tuner and what each CLI flag means
  (the behavior this model describes).
- `tuner/objectives/druid-reference-v1.json` — a real, checked-in deployment
  objective, to see what "winning" concretely looks like.
- The Python sources under `tuner/src/tuner_cli/` — the implementation whose
  shape this package pins down.

The conceptual *why* behind every choice here — the argument for paired racing
over naive tournaments, for shadow validation of pruning, for explicit epochs —
is the tuner's north-star design. This tutorial is its translation into the
concrete types of `tuner-domain-model`.
