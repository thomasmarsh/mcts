# Implementation Notes

These notes are mostly for my own consumption but written out so that others may
potentially benefit or point out errors in understanding.

## General Principles

* _Ease of Adoption_: Conforming to `Game` should not place undue burden on the game implementor.

* _Aheuristic_: Aside from basic configuration, the MCTS algorithm used should prefer approaches which do not require the user provide complex domain dependent knowledge, such as online tuning techniques. (This is corollary to the _Ease of Adoption_ principle.)

* _Single Implementation_: Rather than many different tree search implementations, prefer a single one which is parameterized, at the potential cost of memory or efficiency.


## Type Safety

In general, there are some opportunies for [parsing vs. validation](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/).

For example, consider something like:

```rust
  struct ActiveState<S>(S);
  struct TerminalState<S>(S);
  enum State<S> {
    Active(ActiveState<S>),
    Teriminal(Teriminal<S>),
  }

  trait Game {
    type S;
    type M;

    fn apply(state: &ActiveState<S>) -> State<S>;
    fn gen_moves(state: &ActiveState<S>) -> NonEmpty<M>
    fn get_reward(state: &TerminalState<S>) -> i32;

    ...
  }
```

We do a lot of `is_terminal` checks, many more than `gen_moves`. However,
it violates the  principle that the `Game` interface should not place undue
burden on the implementor. The interface for the implementor of the trait could
maintain simpler types, but a wrapper could be used internally by the libary
which mediates the "parsing". Note that some things like terminality might be
better determined lazily.

Similarly, more in keeping with the current approach, is to to check (to parse) if
the state is terminal at the time we receive it. Additionally, a similar strategy can
be applied for the `Node<M>` type, which becomes an enum:

```rust
enum Node<M> {
  Root(...),
  Unexpanded(...),
  Expanded(...),
  Terminal(...),
}
  
```

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

## Transpositions

Transpositions increase memory demand but have two advantages in that they
can: 1) cache potentially  expensive operations like teminality checks and move
generation, and 2) transform the game into a DAG with potentially more efficient
evaluation strategies.


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

## Terminology

I'm using "move" and "action" synonymously. I prefer "action", and may unify the
terminology around the more generic term. 

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
changed, but the color they are playing is.

### Game Interface

Tentatively, we could consider the following interface:

```rust
trait Game {
  type S : ...;              // The game state
  type P : Copy + Hash + Eq; // The player type
  
  // Get the current active player
  fn player_to_move(state: &Self::S) -> P;

  fn all_players() -> Vec<P>;

  ...
  }
```

The new method `all_players()` would give us the ability to iterate over the set of P.

By making `P` conform to `Hash`, we can use `P` values as hash indices for lookups. It
is also worth considering whether to require the game provide conversions to `usize`
so that we may more efficiently use the player as an array index. A less type safe
convention used in other engines is simply to establish that `P` is an integral type.

## Traces

Many algorithms depend on traces. For example, AMAF/RAVE depends on the history of
simulated moves during rollout. MAST requires the same history as well as the history
of nodes from the selection process. (Is that correct?)

Currently a `stack: Vec<M>` is maintained for the selection and backprop, and I have
code which maintains a `history: Vec<M>` for preliminary MAST support during. It would
be nice to simply write `self.trace(m, phase)` where `phase` is
`enum Phase { Select, Simulate }`, or similar.

Rather than tracing as needed, we could also consider supporting a diff
operation, where the state at the root is compared to the state select, which
can be compared to the state at the end of simulation. Generating a diff of
moves can be more efficient but requires support from the `Game` interface. Such
a diffing strategy, might discard move ordering, which would eliminate the later
possibility of NST or similar techniques.

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


## Secure Child

Max, Robust, and other child selection techniques. Another one listed was "Secure
Child". This is what I came up with, but didn't explore fully. Dropping it here so
it is not lost:

```rust
use statrs::distribution::{Normal, Univariate};

fn secure_child(confidence_level: f64, child: &Node<M>) -> f64 {
    let mean = child.q as f64 / child.n as f64;
    let std_dev = (child.q_squared as f64 / child.n as f64 - mean.powi(2)).sqrt();
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.inverse_cdf(confidence_level)
}
```
