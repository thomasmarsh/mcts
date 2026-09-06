# Implementation Notes

These notes are mostly for my own consumption but written out so that others may
potentially benefit or point out errors in understanding.

## General Principles

* _Ease of Adoption_: Conforming to `Game` should not place undue burden on the game implementor.

* _Aheuristic_: Aside from basic configuration, the MCTS algorithm used should prefer approaches which do not require the user provide complex domain dependent knowledge, such as online tuning techniques. (This is corollary to the _Ease of Adoption_ principle.)

* _Single Implementation_: Rather than many different tree search implementations, prefer a single one which is parameterized, at the potential cost of memory or efficiency.

## Instrumentation

A general idea that I come back to is that it would be nice to instrument the tree
search to capture statistics about operations like `gen_moves` to inform the
cost/benefit of any such changes. The hope is that we can 1) avoid doing error-prone
checks of the validity operations (e.g., calling `G::apply` on terminal states), 2)
eliminate redunant checks, and 3) clarify the intent of the code.

```rust
struct Stats {
  is_terminal_count: u32,
  gen_moves_count: u32,
  max_depth: u32,
  avg_branch_factor: u32,
  transposition_found_count: u32,
  ...
}
```

Also consider whether these stats can parameterically define whether certain values
should be loaded lazily, such as whether or not a node is terminal.

## Performance Profiles

MCTS can be used in many potential applications. In real time video processing
it is common to expand all nodes during simulation. In book generation, we
only generate a tree from a single, very long running iteration. Test cases or
benchmarks should be introduced to establish that these or other use cases are
not negatively impacted by future design decisions.

## Transpositions

Transpositions increase memory demand but have two advantages in that they
can: 1) cache potentially  expensive operations like teminality checks and move
generation, and 2) transform the game into a DAG with potentially more efficient
evaluation strategies.

A basic version that uses average statistics for all transpositions during
selection. This provides a modest bump in performance. However, a generalized
approach would be better so we can parameterize the evaluation and backprop.
This bumps into the problem of symmetries. (See below.)

More sophisticated use of the transpositions can, for some games, require solving
the graph history interaction.

## Symmetries

Many games exhibit various symmetries. For example, tic-tac-toe exhibits 8-way
symmetry. Leveraging this reduces the state complexity from hundreds of thousands
of states to a more manageable 765.

These symmetries can be exploited when identifying transpositions to establish an
average value for a node. This depends on canonicalization of the state to a single
symmetry which is used as the key for the evaluation. (In tic-tac-toe, the board is
represented as a u32; the canonical value is the symmetry with the lowest value.)

This approach fails for RAVE and other techniques which memorize actions, rather than
states. This can be resolved by canonicalizing the actions as well.

## Implementation Guide

MCTS is often approached with enthusiasm at the prospect of an aheuristic
approach, especially as compared to, for example, the difficulty of constructing
a robust minimax evaluation function. There are, however, some very critical
decisions that depend on the domain in question. For example, RAVE is typically
more useful for games that involve piece placement but few or no movement
actions. An additional complication is that certain strategies and algorithms
may not be cleanly combined.

It would be nice to have a guide for game developers who wish to add an AI with clear
decisions and specific recommendations for certain types of games. Implementing an
AI depends on good benchmarks. Random and flat Monte Carlo strategies are also provided.
It would be nice to have automated round-robin testing, benchmarking, and parameter
tuning.

## Players

Monte Carlo Tree search in its basic form does not distinguish between players and
can treat the game from perspective of the root node. However, many algorithms that
improve MCTS depend on isolating traces or other statistics per-player. Many
implementations assume a 2-player game at this point. The Game API should strive as
much as possible to maintain independence from assumptions about the number of players.

Some examples of games with specific numbers of players:

- _1 player_: puzzles, such as sudoku or games like Threes/2048
- _2 players_: most classical abstract strategy board games like chess or go
- _N players_: most modern Euro board games like Carcassone
- _N players, 1 perspective_: cooperative board games like Pandemic

Where possible, player specific values are maintained in an array, with a value per
player. For the moment, it makes sense to use a paranoid strategy by default.

### Move Order

The move order may not be necessarily alternating. A simple example of such a
game is Bidding Tic Tac Toe, where players bid for the who moves next. Another
example is when move splitting is employed. A use may split moves to reduce the
branching factor, to the benefit of certain algorithms like MCTS-Solver.

Although various move orders can be emulated by introducing a null move (for
example, if a player has not won the bid and should not play), such as solution
is complex to generalize and would presumably introduce bias into the the tree
search. Rather than require all games to emulate player alternation, we should
strive to simply rely on knowing the current player for a given state.

The first move of a game employing the "pie rule" or Swap2 in Gomoku also allow
the players to swap colors. Here the order of the order of players is not necessarily
changed, but the color they are playing is. The swap rule is implemented in gonnect.

## End Game Performance

Vanilla MCTS is characterized by excessively pondering in the face of obvious wins.
Decisive move and anti-decisive moves have general application, but would also help here.
Additionally, MCTS-Solver is a great technique for the end game that can make the
final moves more efficient. One technique is to apply MCST-Solver when available moves
drops below a threshold.

## AMAF / RAVE

As a preliminary implementation of RAVE, I do not isolate points by player.
I understand that this is usual for early implementations of RAVE and for
descriptions of the algorithm. Changing this depends on settling on how to
represent the player in the `Game` interface. The same issue arises from MAST
and many other algorithms.

For RAVE, I have implemented the simplest version without a skip heuristic.
Additionally, GRAVE and HRAVE enhancements look very straightforward. In
particular, GRAVE is supposed to provide state of the art results at the time of
its writing, with a significant advantage over RAVE for some games.

## MAST

As a first step in guiding the simulation step of random rollouts, I added an incorrect
implementation of MAST. Surprisingly this actually seems to help performance for at
least one game. MAST seems very similar to PoolRAVE

## Low Level Performance

We use [rand_xorshift](https://crates.io/crates/rand_xorshift) and
[rustc_hash](https://crates.io/crates/rustc_hash) out of performance consideration.
It is important to note that although rand_xorshift claims to pass many randomness
metrics, it may introduce some artifacts.

## Parallelization

Supporting all common models of paralellization is a goal, but a late one. Establishing
more of the shape of the problem takes priority before more intrusive structural changes
paralellization would necessate.

Leaf paralellization, in particular, seems trivial to implement. Tree
paralellization with virtual loss is also well explored and is an obvious
target.

A downside of paralellization is that we lose the ability to target certain
runtimes such as wasm. A decision would need to be made whether to 1) not
support those platforms, 2) maintain two versions of the code, 3) support a
single implementation, perhaps with the help of macros.
