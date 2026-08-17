# Research Plan

This is a plan of what to research, not necessarily an implementation plan. Some 
things implemented (perhaps only partially) are checked off on the list.

### Foundational
- [x] Random (baseline)
- [x] Flat mc (baseline)
- [x] Vanilla mcts
- [x] Max time
- [ ] Replace `Compose` and `Strategy` implementation with composable algebra

### Benchmarking / Tuning
- [x] Battle royale
- [x] Round robin
- [x] SMAC3 ad hoc integration
- [x] Automatic SMAC3 tuning
- [ ] Generalize benchmarking (WIP)

### Selection
- [x] Max Child
- [x] Robust Child
- [x] Max-Robust Child
- [x] Secure Child
- [x] UCT
- [x] UCB1-tuned
- [ ] UCB-V
- [ ] Bayesian UCT
- [ ] EXP3 (probabilistic, partial observable games, simultaneous moves)
- [ ] Hierarchical optimistic optimization for trees
- [ ] Move groups
- [ ] Decisive moves / anti-decisive moves
- [ ] Progressive bias
- [x] MTCS-Solver
- [ ] PUCT
- [ ] Monte Carlo paraphrase generation (MCPG)
- [ ] Regulated Policy Optimization Selection
- [ ] Semisplit Moves / Turn Linearization

### Simulation << MORE ADVANTAGEOUS THAN SELECTION
- [ ] Rule based simulation policy
- [ ] Contextual Monte Carlo search
- [ ] Fill the board
- [x] Move Average Sampling Technique (MAST)
- [x] N-gram selection technique (NST)
- [ ] Predicate-Average Sampling Technique (PAST)
- [ ] Feature Average Sampling Technique (FAST)
- [ ] Use History Heuristics
- [ ] Use of evaluation functions
- [ ] Simulation balancing 
- [ ] Last good reply (LGR)
- [ ] Patterns
- [ ] Dynamic Backoff NAST (N-gram Average Sampling)
- [ ] LGRF-2 (Last Good Reply with Forgetting)
- [ ] UCB1-Driven Playouts
- [ ] Shallow Cutoffs with Shallow Material Evaluation
- [ ] PoolRollout
- [ ] Decisive / Anti-Decisive Playout Truncation

### Tuning
- [x] Opening books (Quasi-Best-First self-play, Gonnect; `game_host::GameAdapter::book_build` is the
      generic per-game hook other games can adopt the same way `tune_eval` was adopted)
- [ ] Online Tuning
- [ ] Search seeding (seed nodes with artificial runs)

### Move pruning
- [ ] Progressive unpruning / widing
- [ ] Absolute and Relative pruning
- [ ] Pruning with domain knowledge
- [ ] Elastic Tree Nodes / Dynamic State Abstraction
- [ ] Novelty-Based Pruning

### Others
- [ ] History heuristic
- [x] Progressive History

### AMAF Variants
- [x] AMAF
- [x] RAVE
- [x] GRAVE
- [ ] HRAVE
- [ ] Permuation AMAF
- [ ] Alpha AMAF
- [ ] Same-first AMAF
- [ ] Cutoff AMAF
- [ ] Killer RAVE
- [ ] PoolRAVE 
- [ ] Rapid Action Value Correction (RAVC)

### Structural
- [ ] Iterative widening
- [x] Meta-MCTS / Nested MCTS
- [ ] Infrastructure to easily expose game to MuZero
- [ ] N-players (n > 2)
- [x] Tree reuse across moves (re-rooting)

### DAG
- [ ] Persistent transposition table / exact position cache (hash -> NodeStats | (hash,action) -> NodeStats)
- [x] UCB for DAGs
- [ ] Structural - configurablue Node|Edge|Node+Edge stat storage
- [ ] Graph re-rooting (parent-specific stats,  bounded reachability pruning—not the existing tree promotion path)
- [ ] Symmetry canonicalization

```
graph_reuse = none | exact_node | exact_edge | both
history_reuse = off | progressive_history | prior
```


### Paralellization
- [x] Virtual loss
- [x] Leaf paralellization
- [x] Root paralellization
- [x] Root-tree parallelization
- [ ] Hybrid parallelizaiton

### Backprop
- [ ] Weighing Simulation Results (higher weight for shorter simulations, later in game sims)
- [ ] Score bonus
- [ ] Decaying reward
- [ ] Transposition table updates

See: https://ics.uci.edu/~dechter/courses/ics-295/fall-2019/presentations/Pezeshki.pdf

## Composable Algebra

We want to avoid lots of policy variant branching in the hot path or storage waste

First revision was:

```rust
pub trait Strategy<G: Game>: Clone + Sync + Send + Default {
    type Select: select::SelectStrategy<G>;
    type Simulate: simulate::SimulateStrategy<G>;
    type Backprop: backprop::BackpropStrategy;
    type FinalAction: select::SelectStrategy<G>;

    // etc.
}
```

Implementations suffer from combinatorial explosion and ignore important compositional
properties like interactions between the methods. To help minimize the explicit struct/impl
burden, added:

```rust
#[derive(Clone, Copy, Default)]
pub struct Compose<Sel, Sim, Bp = backprop::Classic, FA = select::RobustChild>(
    PhantomData<(Sel, Sim, Bp, FA)>,
);

impl<G, Sel, Sim, Bp, FA> Strategy<G> for Compose<Sel, Sim, Bp, FA>
where
    G: Game,
    Sel: select::SelectStrategy<G>,
    Sim: simulate::SimulateStrategy<G>,
    Bp: backprop::BackpropStrategy,
    FA: select::SelectStrategy<G>,
{
    type Select = Sel;
    type Simulate = Sim;
    type Backprop = Bp;
    type FinalAction = FA;
}
```

But still not ideal. Model in Idris and lower to Haskell then Rust?

Conceptually:

```
    mkStrategy
        :: {Game g}
        => Config       -- A complex ADT / feature mask
        -> Strategy g   -- A reified strategy
```

But if a config is something like "use RAVE and MAST and PN", we need to account for
all interactions, which structural elements to use, and which to abandon. This seems to
form a lattice of sorts.

- Storage:
    - Anything global tree storage related: doesn't need to be optimized (a transposition
      table, e.g.); can be present but unused

    - Anything Node/Edge specific _should_ be conditionally present

    - Consider having several template tree/node types rather than try to construct
      this dynamically. Expand set as necessary

- Computation / branching:
    - Anything in coarse steps, initialiation, between search/select/expand/rollout/backprop
      steps _might_ be permissible to include in any combination of features

    - Anything in rollout is the hottest path
