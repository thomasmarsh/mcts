#set document(title: "Paired-Comparison Hyperparameter Tuning for MCTS", author: "MCTS project")
#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(size: 10.5pt, font: "New Computer Modern")
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 15pt, weight: "bold")[
    Paired-Comparison Hyperparameter Tuning for MCTS:
    Where Successive Halving Is Unsafe, and a Cross-Game Calibration Failure
  ]

  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Internal technical note -- MCTS workspace
  ]
]

#v(0.5em)

#block(inset: (x: 1.5em))[
  *Abstract.* We tune MCTS hyperparameters by playing a candidate against a
  fixed panel of reference opponents and scoring it on _paired_ game
  outcomes -- the same task seed drives a candidate game and its mirror, and
  the objective is the bootstrap distribution of the per-pair utility
  difference. This is a different statistical object from the scalar noisy
  $f(theta)$ that irace, SMAC, and Hyperband assume, and it changes where a
  racing/elimination policy goes wrong. With a deterministic simulation of
  the elimination rule, calibrated once from real game data, we find an
  $eta = 2$ successive-halving rank cut has *intact top-set integrity* --
  it essentially never eliminates a genuine finalist -- but *reverses the
  survivor boundary* on #sym.approx 7% of its eliminations: the cut candidate is at or
  above the last retained one at full depth. The cause is structural:
  resolving the 4th-vs-5th boundary of an 8-cohort on a 12-pair estimate is
  a coin flip, and a rank cut always resolves it. A "spare-near-tie"
  softening removes #sym.approx 95% of the boundary reversals in simulation while
  keeping #sym.approx 3#sym.times the compute saving of the conservative incumbent --
  but enabled on a _different_ game behind a randomized-audit net, it
  tripped the audit's suspension once in six seeds, showed no held-out
  gain, and produced a boundary-reversal rate two orders of magnitude above
  the first game's calibrated prediction. The conservative paired-bootstrap
  pruner remains authoritative. Separately: SMAC's `ask()` is not
  reproducible on byte-identical inputs, so reproducibility rests on replay
  of a recorded evidence log, never on re-running a seed.
]

= The tuner and its objective

The tuner searches a conditional mixed hyperparameter space for one MCTS
game: a family choice among #sym.approx 30 tree policies, plus continuous and integer
knobs conditional on family (exploration constant, RAVE/AMAF weights,
progressive-history weight, NST backoff, rollout schedule, ...). A
_candidate_ is evaluated by a _pair_: two games at a fixed task seed against
one reference opponent, candidate on each seat once, so opening-and-color
luck cancels within the pair. The opponent panel is frozen per objective
epoch. The per-pair observation is a bounded utility difference in
$[-1, 1]$; a candidate's score is the mean over its pairs, and all inference
is nonparametric bootstrap over the recorded pair differences, stratified by
opponent.

Search proceeds in _cohorts_: propose a batch (SMAC-guided, with
random-reserve and bootstrap slots), evaluate against a shared common
prefix of task seeds, then advance the whole cohort or eliminate part of it
before deepening. Elimination runs through one tagged interface: the
authoritative *all-strata paired bootstrap* (eliminate only when a candidate
is behind the survivor boundary with high confidence in _every_ opponent
stratum), and a *shadow* mode that records what an alternative policy would
have decided without acting. A randomized *continuation audit* keeps a
deterministic 25% sample of eliminated candidates running to full depth; one
audited candidate that reaches its boundary suspends active elimination for
the rest of that run.

Under a paired objective the unit of evidence is already a difference
$d_i = u_i^"cand" - u_i^"mirror"$ against a fixed panel, so the _population_
question ("is $A$ worse than $B$?") is sharper than under the scalar
instance-sampled draw irace and Hyperband assume -- but the _boundary_
question is not. Two close configurations differ by a mean paired
difference near zero, and the 12-pair minimum common prefix gives a
standard error large enough that the sign is near a coin flip. Any policy
that must produce a total order at the cut -- a rank cut does -- resolves
that near-tie, and resolves it wrong about half the time it matters.

= The successive-halving shadow gate

We asked whether the frozen $eta = 2$ common-prefix rank cut (keep
$ceil(n/2)$ of an $n$-cohort in one look, on the 12-pair prefix mean, ties
broken by a deterministic fingerprint) is safe enough to enable actively.

Two earlier approaches failed instructively. A matched-pair design -- one
paired-bootstrap run and one halving run per seed, required to share
byte-identical evidence -- was unreachable because *the tuner is not
reproducible at a fixed seed*: SMAC's `ask()` proposes different candidates
from byte-identical told observations even with `deterministic=True`, one
worker, and a fixed scenario seed. A single-run replay design -- recompute
_both_ policies over one run's recorded evidence -- works for
realistic-instance checks, but four real runs give only eight cohort draws
and cannot deliberately probe the near-tie regime.

The gate that settled it is a *deterministic simulation sweep* of the cut
rule. We calibrate a synthetic pair-outcome model once from five recorded
reduced-fidelity runs of the deployment game (Druid): an empirical
single-pair utility CDF in five candidate-strength bins ($gt.eq 30$ samples
each), the within-cohort strength spread (mean 0.28, sd 0.20), the observed
true 4th-vs-5th boundary-gap distribution (mean 0.083, sd 0.070), and a
cross-stratum deviation correlation $rho = 0.74$ from 210 stratum records.
The sweep calls the _shipped_ decision function on synthetic cohorts across
21 $("boundary gap", "spread")$ cells $times$ 3000 trials, sweeping the
latent boundary gap from $-0.04$ (5th candidate secretly stronger) through
0 to $+0.20$. Two metrics, computed against full-prefix truth: *top-set
false eviction* (eliminated candidate whose full-prefix estimate lands
within the finalist count) and *boundary reversal* (eliminated candidate
whose full-prefix mean is strictly above the last retained candidate's;
exact fingerprint ties counted separately). Preregistered: top-set false
95% upper $lt.eq$ 3%; boundary-reversal 95% upper $lt.eq$ 3% overall,
$lt.eq$ 6% worst cell, $lt.eq 2 times$ the paired baseline per cell; mean
pairs saved $gt.eq$ the paired baseline.

= Result: top-set safe, boundary unsafe

#figure(
  table(
    columns: (1.7fr, 0.9fr, 0.9fr, 1.5fr, 0.9fr),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    [*Policy*], [*evict/run*], [*pairs saved*],
    [*boundary rev. (95% u.)*], [*top-set false/run*],
    [paired bootstrap], [0.9], [5.3], [0.00% (0.01%)], [0.000],
    [$eta=2$ rank cut (shipped)], [4.0], [24.0], [6.81% (6.91%)], [0.011],
    [keep $ceil(0.625 n)$], [3.0], [18.0], [9.67% (9.81%)], [0.002],
    [keep $ceil(0.75 n)$], [2.0], [12.0], [13.71% (13.90%)], [0.001],
    [spare-near-tie, margin 0.05], [3.3], [19.8], [2.11% (2.17%)], [0.002],
    [spare-near-tie, margin 0.10], [2.7], [16.2], [0.62% (0.65%)], [0.000],
  ),
  caption: [Deterministic sweep of the elimination rule, 21 cells
    $times$ 3000 trials, calibrated from recorded Druid pairs. "pairs saved"
    is projected suffix work not done. The $eta=2$ cut and both
    keep-more-than-half variants fail the preregistered boundary-reversal
    thresholds; both spare-near-tie margins pass every clause in every
    cell.],
)

The shipped $eta = 2$ cut *fails*. Top-set integrity is intact -- its
top-set false-eviction rate is #sym.approx 0.3% (0.011 per run, inside the
preregistered 3% bound; it does not promote a back-half candidate over a
genuine finalist) -- and it saves 4--8#sym.times the suffix compute of the
incumbent. But 6.8% of its evictions are boundary reversals,
against a paired-bootstrap baseline that reverses at rate zero in every
cell. The rate is roughly _flat_ across the latent-gap range rather than
spiking in the forced near-tie cells: it is a property of resolving any
survivor boundary on a 12-pair estimate. The cut keeps 4 of 8; candidates
ranked 6--8 are cleanly behind and never reverse; the 4th-vs-5th pair is
the coin flip, it is one of the four evictions per cohort, and it lands
wrong about a quarter of the time $arrow.r$ #sym.approx 6--7% across all evictions.
The paired-bootstrap pruner scores zero only because it demands confidence
the boundary pair never has -- it "evicts almost nobody" (0.9/run vs 4.0).

*Keeping more does not help.* The keep-$0.625n$ and keep-$0.75n$ variants
have the same boundary reversals _per run_ as the $eta=2$ cut (0.27--0.29)
-- the same wrong cuts, fewer and larger, at half the compute saving.

*Not cutting a near-tie does.* Spare-near-tie keeps the rank cut but
re-admits any would-be-eliminated candidate whose 12-pair paired mean is
within margin $delta$ of the last survivor. At $delta = 0.10$, boundary
reversals drop from 0.27 to 0.017 per run -- the incumbent's cleanliness --
while still saving #sym.approx 16 pairs/run, #sym.approx 3#sym.times the incumbent's 5.3. A sweep
re-run at $delta = 0.10$ over a larger cohort profile put the 95% upper
bound at 0.06% overall / 0.35% worst cell, top-set false at 0.08%. The
softening also passed replay-rescoring of four recorded runs (0 reversals)
and two fresh live runs whose recorded decisions matched replay
recomputation byte-for-byte.

= The active bake-off, and a calibration that did not transfer

Passing the shadow gate authorised a three-arm equal-compute bake-off --
_no elimination_, _paired-bootstrap_, _spare-near-tie_ -- each a full
tuning campaign under matched compute with the continuation audit live. The
frozen decision rule keeps the incumbent unless the challenger is _both_
audit-clean _and_ measurably better on held-out score. Because the
mechanism question is game-independent by construction, the bake-off was run
on 8#sym.times 8 Breakthrough (#sym.approx 5#sym.times faster per pair than the Druid
deployment target) at 1200 iterations, $3 times 6 times 2 = 36$ runs, #sym.approx 5 h.

#figure(
  table(
    columns: (1.5fr, 1fr, 1fr, 1.3fr, 1.7fr),
    align: (left, right, right, right, right),
    stroke: 0.5pt,
    inset: 5pt,
    [*Arm*], [*nominal elims*], [*pruned*],
    [*audited bdy. rev.*], [*held-out score vs. incumbent*],
    [no elimination], [0], [0], [--], [$plus.minus 0$ (identical)],
    [paired bootstrap], [6], [4], [0 / 2 audited], [$plus.minus 0$ (identical)],
    [spare-near-tie], [49], [35], [1 / 14 audited], [$+0.007$ (CI $-0.09, +0.10$)],
  ),
  caption: [Breakthrough bake-off, aggregated over 6 seeds $times$ 2
    budgets. "audited" is the 25% continuation sample. The one audited
    boundary reversal tripped the suspend-after-first-reversal rule for
    that run.],
)

Outcome: *keep paired-bootstrap elimination*. Spare-near-tie is not
audit-clean -- one audited continuation reached its boundary; extrapolated
by the 25% audit rate that is #sym.approx 4 true reversals over 49 eliminations, #sym.approx 8%
-- and it buys nothing on quality (held-out score and top-set recall
against both other arms indistinguishable, score delta CI spanning zero).
The compute _was_ saved as designed (spare-near-tie funded one extra
completed cohort in five of six seeds), but the rule requires safe _and_
better.

The number worth carrying forward is the mismatch. The Druid calibration
put the $delta = 0.10$ boundary-reversal rate at a 95% upper bound of
0.06%; the live Breakthrough audit implies #sym.approx 8%, over two orders of
magnitude higher. Breakthrough is a sharper tactical game -- small
hyperparameter differences swing more pairs, so the 4th-vs-5th boundary is
noisier relative to its true gap -- and a calibration built from one game's
pair-utility CDF and strength spread does not describe another. *The noise
model is not portable across games.* Future use of successive-halving
elimination needs a per-game calibration or a deliberately wide noise prior.

= Relation to irace

The finding refines rather than contradicts irace's design.

- *Elimination risk is not where the rank test spends its power.* irace's
  Friedman test is conservative about the _top_ and, like our simulation,
  has intact top-set behaviour. The reversal risk we measure is _inside the
  retained set_, at the survivor boundary -- exactly where a rank statistic
  has least power because the configs are close. irace inherits the same
  boundary problem; its two-candidate Wilcoxon shortcut (stop when all
  paired signs agree) _is_ a boundary decision on a small paired sample.

- *Our incumbent is strictly more conservative than irace's racing.* The
  all-strata paired-bootstrap pruner eliminates only on high confidence in
  every stratum, so it cuts far less than per-iteration Friedman
  elimination would (0.9 vs. 4.0 per run) and never reverses. That is the
  trade: irace accepts some mis-elimination to race faster; we currently do
  not, and pay full suffix compute on the retained lower ranks. The
  bake-off showed that conservatism is _currently free_ (identical held-out
  score to no elimination) but also *not yet paying off*.

- *irace mechanisms that stay open comparators*, each to be measured
  through the same shadow / equal-compute interface first: the
  per-parameter generational sampling model as an alternative proposer
  (implemented, not yet raced to a decision); soft restart on measured
  stagnation; elite-relative time capping (only if runtime becomes a
  declared objective -- it changes the censoring contract). irace's
  `dom_elim` is execution-cost capping, not a statistical-dominance proof.

= Reproducibility

Because SMAC `ask()` is nondeterministic on byte-identical told
observations, a fresh run at a fixed seed does not reproduce a prior run.
What _is_ deterministic is *resume and replay of a recorded evidence log*:
the append-only per-run event log is the scientific authority and every
derived quantity is a pure function of it. All gates above are therefore
evaluated over one recorded log at a time -- both elimination policies
recomputed over the same observations -- never by comparing independent
runs. (Engine-level determinism was still worth fixing: two MCTS families,
`flat_mc` and `random`, were reseeding from OS entropy because the
direct-search builder never received the run seed. Fixed; those families
are now seed-stable even though the proposer above them is not.)

= What is authoritative

Elimination: *all-strata paired bootstrap*, shadow-only by default, active
mode gated behind the randomized continuation audit and the
suspend-after-first-audited-reversal rule. No successive-halving variant is
authoritative. Proposer: SMAC-guided; the generational irace-style proposer
exists as a measured alternative, not a default. The deterministic
elimination-rule sweep and its Druid calibration are kept as reusable
tooling; the calibration is explicitly single-game and must be regenerated
per deployment game.

// TODO: regenerate the pair-outcome calibration from recorded Breakthrough
// (and one more game) and re-run the sweep, to see whether the
// boundary-reversal rate is predictable per-game or the simulation needs a
// wide-prior robustness band instead of a point estimate.
// TODO: run the proposer bake-off (11a tooling) to an actual
// keep/change/reject decision for the generational irace-style proposer vs
// SMAC-mixed, under matched compute on a fast game.
// TODO: a spare-near-tie variant whose margin comes from a per-run noise
// estimate rather than a fixed 0.10, preregistered and gated separately,
// if per-game calibration turns out to be tractable.
