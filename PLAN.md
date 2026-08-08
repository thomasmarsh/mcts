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
- [x] Secure Child
- [x] UCT
- [x] UCB1-tuned
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
- [ ] Opening books
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
- [ ] Rapid Action Value Correction (RAVC)

### Structural
- [ ] Iterative widening
- [ ] Meta-MCTS (rollout should just be a function that takes a strategy as an arg)
- [ ] Infrastructure to easily expose game to MuZero
- [ ] N-players (n > 2)
- [x] Tree reuse across moves (re-rooting)

### DAG
- [ ] Support for transposition tables
- [ ] UCB for DAGs

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
