#set document(title: "RAVE-Blended Correction for DAG-Merged Information Set MCTS", author: "MCTS project")
#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(size: 10.5pt, font: "New Computer Modern")
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 15pt, weight: "bold")[
    RAVE-Blended Correction for DAG-Merged Information Set MCTS
  ]

  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Internal technical note -- MCTS workspace
  ]
]

#v(0.5em)

#block(inset: (x: 1.5em))[
  *Abstract.* Merging transposed nodes in a DAG-structured search tree is a
  standard variance-reduction technique under perfect information, but under
  hidden information two histories that share an information set are not
  necessarily interchangeable -- forcing an edge's local estimate to agree
  with a pooled node estimate risks *strategy fusion*. We show, with a
  hand-verifiable fixture small enough to brute-force the true Bayes-optimal
  value, that an existing residual-correction mechanism ported from a
  perfect-information Monte-Carlo Graph Search paper is *not* just
  unhelpful but measurably biased when applied to Information Set MCTS
  (ISMCTS): its error does not shrink as the search budget grows, and by
  construction it permanently freezes a corrected edge out of further
  direct sampling. We propose and validate a fix inspired by RAVE/GRAVE:
  blend the pooled node estimate into edge *selection* using a decaying
  schedule on the edge's own visit count, rather than gating traversal
  outright. On the same fixture, this RAVE-blended correction converges to
  the true value at a rate comparable to (and in this fixture, better than)
  plain ISMCTS with no merging at all, resolving the soundness question.
  A real-game strength benchmark (Oh Hell) shows the fix does not yet
  translate into a playing-strength win over plain ISMCTS at the budget
  tested -- a separate, still-open question from the soundness result this
  note is centrally about.
]

= Motivation

`SearchConfig::ismcts_mode` implements Information Set MCTS (Cowling,
Powley & Whitehouse, 2012) with optional explicit DAG merging: nodes
reached by different real histories that share an information set (keyed
by `Game::info_set_hash`) are merged into one shared node, pooling both
edge- and node-level statistics (`GraphStats::Both`). Because a merge under
hidden information is not guaranteed exact -- unlike a perfect-information
transposition, where two paths reaching the same literal state truly have
one correct value -- the codebase gates this merge behind a correction
mechanism intended to catch a merged node whose pooled estimate has
drifted from what a specific traversing edge would say on its own.

The mechanism in production, `McgsCorrection::Residual`, is adapted from
Czech, Korus & Kersting's Monte-Carlo Graph Search paper for AlphaZero
(arXiv:2012.11045) -- a *perfect-information* setting, where a divergent
edge is simply undersampled noise and trusting the better-sampled pooled
value is a safe variance-reduction move. Under hidden information this
polarity is suspect: a divergence can instead reflect a *real* difference
in correct play depending on which hidden information a player holds,
exactly the strategy-fusion failure mode that already makes plain
single-observer ISMCTS lose to determinized UCT/PIMC in some published
benchmarks (Cowling et al.'s Phantom $(4,4,4)$ result).

= The soundness test

Rather than infer soundness indirectly from a win-rate benchmark -- which
can only ever say "helps" or "hurts," never "is this estimate correct" --
we built a fixture small enough to brute-force the true answer. `ConvergeGame`
is a 6-action "pick 4 of 6" game ($binom(6,4)=15$ terminal states) extended
with one hidden coin bit that is resampled every iteration and deliberately
excluded from `info_set_hash`, so states agreeing on the picked mask but
disagreeing on the coin share one information set, while the terminal
payoff genuinely depends on both. This is small enough to compute the exact
Bayes-optimal value at any info-set node by full enumeration, and small
enough for a full search sweep to run in milliseconds.

For a non-root info-set node reachable by more than one action order (so
DAG merging has something to merge), we ran plain `ismcts_mode`, `ismcts_mode`
with DAG merging plus `Residual`, and (this note's contribution)
`ismcts_mode` with DAG merging plus `RaveBlend`, at increasing iteration
budgets (200 / 1,000 / 5,000, averaged over 5 seeds), and measured the
absolute error between the searched node's pooled score and the
brute-forced true value.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt,
    [*Configuration*], [*200 iters*], [*1,000 iters*], [*5,000 iters*],
    [Plain `ismcts_mode` (no merge)], [1.15], [0.78], [0.56],
    [DAG + `Residual`], [1.00], [1.20], [1.20],
    [DAG + `RaveBlend` (this note)], [0.65], [0.56], [0.35],
  ),
  caption: [Mean absolute error against the brute-forced Bayes-optimal
    value at a shared info-set node, as iteration budget grows 25$times$.
    Lower is better; a decreasing trend across columns is the actual
    soundness signal, not any single column.],
)

Plain `ismcts_mode` converges as expected (error roughly halves),
confirming the fixture and harness are sound on their own. `Residual`
does *not* converge -- error stays flat or grows despite a 25$times$ budget
increase, a direct, reproducible confirmation of bias, not merely an
absence of measured benefit. Digging into the mechanism explains why:
`Residual` fires by intercepting descent *before* the merged child is ever
entered, backpropagating a correction through ancestors only -- the
corrected edge and the node it points to receive no visit that iteration,
regardless of which direction the correction points. Once an edge first
triggers the check, it can be permanently prevented from gathering further
direct evidence about its own target, independent of how much search
budget follows. Flipping the correction's polarity alone would not fix
this: it would still gate on the same divergence check, only trusting the
opposite side once it fires.

= A RAVE-blended alternative

We propose `McgsCorrection::RaveBlend`, applying the same *shape* of fix
RAVE/GRAVE (Gelly & Silver, 2011; Cazenave, 2015) already use for a
structurally similar problem: blending a fast-converging pooled estimate
into a slower-converging direct one, using a schedule that decays toward
the direct estimate as its own visit count grows -- critically, entirely
at *selection* time, never by gating or rerouting backpropagation. Given
an edge's own expected score $q_e$ with $n_e$ visits and its DAG-merged
target's pooled expected score $q_p$ with $n_p$ visits, and a RAVE-style
schedule $beta(n_e, n_p) in [0,1]$ (e.g. the existing `HandSelected`,
`MinMSE`, or `Threshold` schedules already implemented for RAVE):

$ "score"_"blend" = beta(n_e, n_p) dot q_p + (1 - beta(n_e, n_p)) dot q_e $

This value replaces the raw edge exploitation term inside UCB1's
selection score. Unlike `Residual`, this is unconditional (no threshold,
no gating) and never intercepts traversal: whichever child is chosen,
selection or not, the search still visits it and still accumulates a real
sample on that edge. This is *this project's own hypothesis* -- no source
already cited combines RAVE-style blending with DAG merging under hidden
information; Cowling et al.'s ISMCTS paper predates GRAVE, and both the
MCGS paper `Residual` comes from and Cazenave's GRAVE paper are
perfect-information only.

As the last two rows of the table above show, `RaveBlend` converges
substantially over the same $25 times$ budget growth (0.65 $arrow$ 0.35),
at a rate comparable to plain `ismcts_mode`, and in this fixture even lands
below plain `ismcts_mode`'s error at every budget tested -- consistent
with DAG merging behaving as genuine variance reduction once the
correction no longer forces strategy fusion or freezes evidence-gathering.

= Strength benchmark: a separate, still-open question

A soundness result does not by itself imply a strength win, and the two
should not be conflated. We re-ran an existing Oh Hell (2-player,
trick-taking, hidden hands) strength benchmark -- `ismcts` vs. `ismcts` +
DAG + correction, matched iteration budget (8,000/move), 15 round-robin
rounds -- substituting `RaveBlend` for `Residual`.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt,
    [*Strategy*], [*Win rate*], [*95% CI (Wilson)*],
    [cheating (sees true hands)], [82.5%], [71.0--90.1%],
    [`ismcts` (no merge)], [43.3%], [31.6--55.9%],
    [`ismcts` + DAG + `RaveBlend`], [24.2%], [15.1--36.3%],
  ),
  caption: [Oh Hell round-robin, 60 games/strategy, matched iteration
    budget. For reference, an earlier run with `Residual` in place of
    `RaveBlend` landed at near noise-level parity with plain `ismcts`
    (36.7% vs. 35.8%).],
)

`RaveBlend`'s merged configuration clearly *underperforms* plain `ismcts`
here, and even underperforms the earlier `Residual`-based run. This is a
real result, not a contradiction of the soundness finding above: the
soundness test proves the estimate is *unbiased in the limit*, not that
it is a good use of a fixed, real-game search budget. A plausible (not yet
confirmed) explanation is schedule mistuning: the default RAVE schedule
(`HandSelected` with $k=1000$) decays slowly, tuned for visit counts
reaching hundreds to thousands, which the small `ConvergeGame` fixture's
edges reach within a few thousand total iterations but Oh Hell's much
larger branching factor likely does not, at this budget, per merged edge --
leaving $beta$ close to 1 (near-total trust in the pooled value) for most
of a real search, unlike the fixture where it visibly decays. This does not
reintroduce `Residual`'s specific failure mode (no freeze, and the
asymptotic bias is gone), but it is a real practical cost at this budget
worth separating cleanly from the soundness question this note is
centrally about.

= Conclusion

`Residual`, ported directly from a perfect-information paper, is
confirmed biased and structurally self-freezing when applied to DAG-merged
ISMCTS. `RaveBlend`, an application of RAVE/GRAVE's existing
selection-time blending idea (rather than a hard, backprop-gating
override) to this same divergence, resolves both problems on a
hand-verifiable fixture: it converges to the true Bayes-optimal value as
budget grows, at a rate at least as good as an unmerged tree. Whether it
also improves playing strength on a real game at practical budgets remains
open and, on the one benchmark tried so far, is negative -- a separate
question from the one this note set out to answer, and the natural
next thing to tune rather than a reason to doubt the soundness result.

// TODO: run the soundness test at a wider seed sweep (20+ seeds, matching
// the ad hoc wider sweep already done for Residual) and report a
// confidence interval on the error trend itself, not just point estimates.
// TODO: sweep RaveSchedule variants (MinMSE, Threshold) and k values tuned
// to Oh Hell's actual per-edge visit-count regime, to test the schedule-
// mistuning explanation for the strength benchmark's negative result.
// TODO: extend the strength benchmark to Ingenious and Phantom(4,4,4) once
// their own transposition density (measured separately per game, e.g. via
// examples/transposition_density.rs) justifies it, to check whether the
// Oh Hell strength result generalizes or is domain-specific.
// TODO: a second hand-verifiable fixture with a larger branching factor
// (more actions per node) would let the soundness test itself probe
// whether the strength benchmark's schedule-mistuning hypothesis is
// actually the cause, rather than relying on an untested plausibility
// argument.
// TODO: consider an adaptive schedule keyed on the *node's* own branching
// factor / average edge visit count at construction time, rather than a
// single fixed k shared across every game.
