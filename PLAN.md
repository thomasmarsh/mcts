
* Foundational
[x] random (baseline)
[x] flat mc (baseline)
[x] vanilla mcts
[x] max time

* Benchmarking
[x] battle royale
[ ] round robin
[ ] generalize benchmarking (TBD)

* Selection
[x] UCT
[ ] UCB1-tuned
[ ] Bayesian UCT
[ ] EXP3 (probabilistic, partial observable games, simultaneous moves)
[ ] hierarchical optimistic optimization for trees
[ ] first play urgency
[ ] move groups
[ ] decisive moves / anti-decisive moves
[ ] progressive bias (add domain specific knowledge as a heuristic - a kind of prior)
[ ] MTCS-Solver (used for end game solvers)
[ ] PUCT
[ ] Monte Carlo paraphrase generation (MCPG)

* Simulation << MORE ADVANTAGEOUS THAN SELECTION
[ ] Rule based simulation policy
[ ] contextual monte carlo search
[ ] Fill the board
[ ] Move Average Sampling Technique (MAST)
[ ] Predicate-Average Sampling Technique (PAST)
[ ] Feature Average Sampling Technique (FAST)
[ ] Use History Heuristics
[ ] Use of evaluation functions
[ ] Simulation balancing 
[ ] last good reply (LGR)
[ ] Patterns

* Tuning
[ ] Opening books
[ ] Online Tuning
[ ] Search seeding (seed nodes with artificial runs)


* Move pruning
[ ] Progressive unpruning / widing
[ ] Absolute and Relative pruning
[ ] Pruning with domain knowledge

* Others
[ ] history heuristic
[ ] Progressive History
[ ] N-gram selection technique (NST) -- generalization of MAST

* AMAF Variants
[ ] AMAF
[ ] RAVE
[ ] GRAVE
[ ] Permuation AMAF
[ ] alpha AMAF
[ ] Same-first AMAF
[ ] Cutoff AMAF
[ ] Killer rave
[ ] Pool rave

* Structural
[ ] Iterative widening
[ ] Meta-MCTS (rollout should just be a function that takes a strategy as an arg)
[ ] Infrastructure to easily expose game to MuZero
[ ] n-players (n > 2)

* DAG
[ ] Support for transposition tables
[ ] UCB for DAGs
[ ] Alternative approaches (TBD)


* Paralellization
[ ] Virtual loss
[ ] Leaf paralellization
[ ] Root paralellization
[ ] Root-tree parallelization

* Backprop
[ ] Weighing Simulation Results (higher weight for shorter simulations, later in game sims)
[ ] Score bonus
[ ] Decaying reward
[ ] Transposition table updates

See: https://ics.uci.edu/~dechter/courses/ics-295/fall-2019/presentations/Pezeshki.pdf
