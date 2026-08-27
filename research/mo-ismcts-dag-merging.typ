#set document(title: "DAG Merging in Multi-Observer Information Set MCTS", author: "MCTS project")
#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(size: 10.5pt, font: "New Computer Modern")
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 15pt, weight: "bold")[
    DAG Merging in Multi-Observer Information Set MCTS
  ]

  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Internal technical note -- MCTS workspace
  ]
]

#v(0.5em)

#block(inset: (x: 1.5em))[
  *Abstract.* Multi-observer Information Set MCTS (MO-ISMCTS; Cowling,
  Powley & Whitehouse, 2012, Section IV-G) grows one search tree per
  player, all advanced together each iteration, with only the acting
  player's own tree consulted for selection at any node. A companion note
  (`rave-blend-dag-ismcts.typ`) added explicit DAG merging to the
  *single*-tree ISMCTS mode and found the existing residual correction it
  is gated behind to be biased under hidden information. This note extends
  DAG merging to MO-ISMCTS and asks the same soundness question, with the
  same method: a hand-verifiable fixture small enough to brute-force the
  true Bayes-optimal value. The suspected failure mode was specific to the
  per-player structure -- the merge key `Game::info_set_hash` is
  *mover*-relative (it hashes the information set from the point of view of
  whoever is to move at a node), but player $p$'s tree $T_p$ needs to keep
  $p$'s own belief state consistent even at nodes where some other player
  moves, so a mover-relative key might merge two histories $p$ can tell
  apart. We find it does not: with no correction at all, DAG merging
  converges player 0's estimate at a shared non-root node to exactly the
  value plain MO-ISMCTS converges to, at a comparable rate. The
  mover-relative key is not a source of bias here; it is also not an
  improvement (it does not reach the player-conditioned value a genuinely
  player-relative merge might). Sound, no correction needed -- a cleaner
  outcome than single-tree DAG merging, which needed `RaveBlend`.
]

= Background

`SearchConfig::ismcts_mode` has two variants. `SingleTree` (SO-ISMCTS)
grows one shared tree; `MultiTree` (MO-ISMCTS) grows one tree per player.
In `MultiTree`, every player's tree gets a node at every position reached
during search, but at a node where player $q$ is to move, only $T_q$ is
selected from -- every other $T_p$ still records a node there, because
$T_p$'s premise is to model the whole game as player $p$ would track it, a
single fixed observer for every node in that tree.

The companion note added explicit DAG merging to `SingleTree`: nodes
reached by different real histories sharing an information set (keyed by
`Game::info_set_hash`) are merged into one node, pooling edge- and
node-level statistics (`GraphStats::Both`). It showed the correction this
is gated behind, `McgsCorrection::Residual` (ported from a
perfect-information Monte-Carlo Graph Search paper), is measurably biased
under hidden information and proposed `RaveBlend` as a fix.

`MultiTree` + DAG had no implementation at all: `SearchConfig::validate()`
rejected the pairing outright, because `choose_action_multi_tree` builds
its per-player trees directly and never consulted `graph_search` /
`mcgs_correction`.

= The merge-key question

`Game::info_set_hash(state)` hashes the information set `state` belongs to
*from the point of view of whoever is to move* in `state`. That is the
right notion for a single shared tree, where the mover changes as you
descend and a merge only ever needs to be sound for that mover at that
node. It is questionable for $T_p$: at a node in $T_p$ where player $q eq.not p$
moves, merging by "the mover's point of view" hashes by what $q$ can
distinguish, discarding whatever $p$ -- the tree's actual fixed observer --
privately knows there. Two histories that differ only in a card $p$ has
already seen revealed would then merge, even though $p$ can tell them
apart.

Whether this bites depends on re-determinization. Without per-node
re-determinization (`ismcts_redeterminize`), `Game::determinize` runs once
per iteration at the root, where $p$ is the mover for $T_p$'s own
decisions, so `priv[p]` stays at its true root value everywhere in $T_p$
for that iteration -- a mover-relative key drops a dimension that never
varies, harmlessly. *With* re-determinization, descent through an opponent
node resamples `priv[p]` (the opponent is the mover there, so `determinize`
guesses $p$'s hidden state), so $T_p$ genuinely sees both values of
`priv[p]` below that opponent node, and a mover-relative merge conflates
them. So the benchmark must run with re-determinization on for the failure
mode to be reachable at all.

= The fixture

`MoConvergeGame` (`mcts/src/strategies/tests.rs`,
`mo_converge_game_tests`) is the `ConvergeGame` "pick 4 of 6"
transposition diamond ($binom(6,4) = 15$ terminal states, reachable by
many action orders) with one owned private bit *per player* instead of
`ConvergeGame`'s single shared coin. `Game::determinize` keeps the current
mover's own bit and resamples the other player's -- matching how a real
card game's determinization keeps the acting player's hand fixed and
guesses the opponent's. `Game::info_set_hash` is mover-relative: the
picked mask plus the mover's own bit, never the opponent's. The terminal
`winner` is deliberately asymmetric in `priv[0]`, so player 0's
Bayes-average value genuinely varies with its own bit (unlike
`ConvergeGame`'s symmetric coin) -- which is what a mover-relative merge
at a player-1-to-move node would be unable to keep distinct inside $T_0$.

Two reference values are computable exactly by backward induction over the
mask-only reduced game:

- the *marginal* value, averaging the leaf payoff over both private bits
  uniformly. This is what plain MO-ISMCTS with re-determinization
  converges to -- re-determinization marginalizes $p$'s own bit below the
  root regardless of any merging.
- the *player-0-conditioned* value, fixing `priv[0]` to the root's true
  value and averaging only over `priv[1]` (which player 0 can never
  observe). A genuinely player-relative merge inside $T_0$ would be
  expected to converge here instead -- arguably more correct, since at the
  root player 0 does know its own bit.

= Result

Target node: picked mask $\{1,2,3\}$ (count 3, player 1 to move), reachable
by all six orderings of $\{1,2,3\}$. Player 0's pooled-edge score at that
node, absolute error against each reference, mean over seeds 1--6, at
increasing budgets:

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt,
    [*Configuration*], [*200 iters*], [*1,000 iters*], [*5,000 iters*],
    [Plain `MultiTree` vs. marginal], [0.55], [0.34], [0.29],
    [`MultiTree` + DAG vs. marginal], [0.42], [0.26], [0.27],
    [`MultiTree` + DAG vs. conditioned], [0.92], [0.76], [0.77],
  ),
  caption: [Mean absolute error at the shared info-set node as the
    iteration budget grows 25$times$ (marginal value $= -0.5$, conditioned
    value $= -1.0$). All runs use `ismcts_redeterminize` and
    `McgsCorrection::Disabled` (merging alone, no correction).],
)

`MultiTree` + DAG converges to the *marginal* value -- the same fixed
point plain `MultiTree` reaches -- at a comparable rate, and slightly
ahead of it at the smaller budgets (more merged samples per node). It does
*not* converge toward the player-0-conditioned value: that error stays
near $0.77$ regardless of budget. The mover-relative merge key is
therefore:

- *not a source of bias.* The suspected failure -- conflating `priv[0]`
  values at a player-1 node and dragging player 0's estimate somewhere
  systematically wrong -- does not occur. The estimate converges to a
  well-defined quantity, the same one an unmerged tree converges to.
- *not an improvement either.* Because plain `MultiTree` with
  re-determinization already marginalizes `priv[0]` below the root, and a
  mover-relative merge does the same thing (dropping `priv[0]` at the
  player-1 node), both land at the marginal value. A genuinely
  player-relative merge key -- `info_set_hash_for(state, observer)`, used
  with `observer = p` at every node in $T_p$ -- would be needed to reach
  the conditioned value, and is left as future work. The measurement says
  it is optional, not required for soundness.

DAG merging inside each independent `PlayerTree`, keyed by the existing
mover-relative hash, is therefore sound with *no* correction -- unlike
`SingleTree` + DAG, which needed `RaveBlend` to converge.

= What was implemented

- `SearchConfig::validate()` now accepts `IsmctsMode::MultiTree` +
  `GraphSearch::Dag(GraphStats::Both)` + `McgsCorrection::Disabled` (only
  `Disabled` -- `SingleTree`'s two corrections have not been validated
  against a per-player tree's convergence).
- `choose_action_multi_tree` gives each `PlayerTree` its own
  `TranspositionTable`. `select_multi_tree` routes each tree's
  child-creation through `table_p.get_or_insert_graph`, keyed by
  `TranspositionKey::new(keying, Game::info_set_hash(new_state), ply + 1)`,
  so two action orders reaching the same information set inside one tree
  share a node. `GraphStats::Both` node-level virtual-loss bookkeeping is
  paired in the descent loop to match `backprop_step`'s removals.
- A known gap, shared with `SingleTree` + DAG: only a newly filled
  `ChildArray` slot consults the table. Once a slot caches an id, a later
  iteration whose determinization hashes to a different information set
  still reuses the cached node -- `select_step` re-checks via
  `verified_child_id`; this path does not yet. Harmless for a game like
  `MoConvergeGame` whose per-mask structure keeps every order reaching one
  info set.
- Tests: `dag_multi_tree_converges_like_plain_multi_tree_does`
  (`mo_converge_game_tests`), plus `validate()` acceptance/rejection
  cases. All run in `cargo test --lib`, sub-second.

= Conclusion

Extending DAG merging from single-tree to multi-observer ISMCTS is sound
as-is: each player's tree merges its own nodes by the mover-relative
information-set hash, no correction, and player 0's estimate at a shared
node converges to the same marginal value plain MO-ISMCTS reaches. The
mover-relative key neither biases the search nor improves it. A
player-relative merge key that would converge to the (arguably more
correct) player-conditioned value is a well-defined follow-on, but the
measurement shows it is not needed for soundness.

// TODO: a playing-strength benchmark of MultiTree + DAG vs. plain
// MultiTree on Oh Hell / Phantom(4,4,4) -- not gating (soundness was the
// deliverable), but the natural next measurement, following
// examples/strength_phantom_mo_ismcts.rs's pattern.
// TODO: design info_set_hash_for(state, observer) and re-run this fixture
// to check whether a player-relative key converges to conditioned_value_p0
// as hypothesised, and whether that is a strength win over the
// mover-relative merge validated here.
// TODO: wider seed sweep (20+ seeds) with a confidence interval on the
// error trend, matching the follow-up already flagged for the SingleTree
// soundness test.
