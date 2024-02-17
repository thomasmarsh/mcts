# Research Plan

This is a plan of what to research, not necessarily an implementation plan. Some 
things implemented (perhaps only partially) are checked off on the list.

### Foundational
- [x] Random (baseline)
- [x] Flat mc (baseline)
- [x] Vanilla mcts
- [x] Max time

### Benchmarking / Tuning
- [x] Battle royale
- [x] Round robin
- [x] SMAC3 ad hoc integration
- [ ] Automatic SMAC3 tuning
- [ ] Generalize benchmarking (TBD)

### Selection
- [x] Max Child
- [x] Robust Child
- [ ] Max-Robust Child
- [ ] Secure Child
- [x] UCT
- [x] UCB1-tuned
- [ ] Bayesian UCT
- [ ] EXP3 (probabilistic, partial observable games, simultaneous moves)
- [ ] Hierarchical optimistic optimization for trees
- [ ] First play urgency
- [ ] Move groups
- [ ] Decisive moves / anti-decisive moves
- [ ] Progressive bias
- [ ] MTCS-Solver
- [ ] PUCT
- [ ] Monte Carlo paraphrase generation (MCPG)

### Simulation << MORE ADVANTAGEOUS THAN SELECTION
- [ ] Rule based simulation policy
- [ ] Contextual Monte Carlo search
- [ ] Fill the board
- [x] Move Average Sampling Technique (MAST)
- [ ] N-gram selection technique (NST)
- [ ] Predicate-Average Sampling Technique (PAST)
- [ ] Feature Average Sampling Technique (FAST)
- [ ] Use History Heuristics
- [ ] Use of evaluation functions
- [ ] Simulation balancing 
- [ ] Last good reply (LGR)
- [ ] Patterns

### Tuning
- [ ] Opening books
- [ ] Online Tuning
- [ ] Search seeding (seed nodes with artificial runs)

### Move pruning
- [ ] Progressive unpruning / widing
- [ ] Absolute and Relative pruning
- [ ] Pruning with domain knowledge

### Others
- [ ] History heuristic
- [ ] Progressive History

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

### Structural
- [ ] Iterative widening
- [ ] Meta-MCTS (rollout should just be a function that takes a strategy as an arg)
- [ ] Infrastructure to easily expose game to MuZero
- [ ] N-players (n > 2)

### DAG
- [ ] Support for transposition tables
- [ ] UCB for DAGs

### Paralellization
- [ ] Virtual loss
- [ ] Leaf paralellization
- [ ] Root paralellization
- [ ] Root-tree parallelization

### Backprop
- [ ] Weighing Simulation Results (higher weight for shorter simulations, later in game sims)
- [ ] Score bonus
- [ ] Decaying reward
- [ ] Transposition table updates

See: https://ics.uci.edu/~dechter/courses/ics-295/fall-2019/presentations/Pezeshki.pdf
