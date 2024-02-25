# Monte Carlo Tree Search
[![Rust](https://github.com/thomasmarsh/mcts/actions/workflows/rust.yml/badge.svg)](https://github.com/thomasmarsh/mcts/actions/workflows/rust.yml)

Learning project for MCTS. This code aims for some efficiency and is strong
enough to play well or better against other libraries. I use it in my
[Nego](https://github.com/thomasmarsh/nego) project.

Current features:

* UCT / UCB1Tuned
* RAVE/GRAVE
* MAST
* Hyperparameter tuning with [SMAC3](https://automl.github.io/SMAC3/main/)
* Arena allocation (just a `Vec`, inspired by [indextree](https://github.com/saschagrunert/indextree))
* Preliminary benchmarking tools
* A growing number of [game implementations](src/games)

Some things I would like to explore:

- Better testability, ergonomics, safety
- More simulation strategies, selection improvements, etc.
- DAGs / transposition tables
- Online tuning

Other alternatives in Rust:
- [minimax-rs](https://github.com/edre/minimax-rs): lock-free tree parallel implementation. Better for low branching factor tactical games due to use of full node expansion strategy and reliance on MCTS-Solver.
- [ggpf](https://github.com/TheLortex/rust-mcts): implements a lot of stuff, including
  AlphaZero and MuZero TF integration. Supports RAVE, PUCT, etc.
- [zxqfl/mcts](https://github.com/zxqfl/mcts): some pretty clean looking code
  with lots of atomics and support for transposition tables. From the author
  of TabNine. I think it has a lot of good ideas on how to parameterize and mix
  different strategies.
- [recon_mcts](https://github.com/trtsl/recon_mcts): mostly focused on parallelism,
  with some clever strategies to combine tree results.
- [arbor](https://github.com/prestonmlangford/arbor/): vanilla MCTS, but with a focus
  on single threaded efficiency, uses hand maintained arena. Has some transposition
  support, but says it is experimental. I haven't looked deeply because the project
  [doesn't provide a LICENSE](https://github.com/prestonmlangford/arbor/issues/2).
- [OxyMcts](https://github.com/Sagebati/OxyMcts): seems to be a vanilla UCT client

Some code and general approaches come from
[minimax-rs](https://github.com/edre/minimax-rs) which has an MCTS strategy.

Not sure if this will become a published library, but it is improving and PRs are welcome.
