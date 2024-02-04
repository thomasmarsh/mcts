# MCTS
[![Rust](https://github.com/thomasmarsh/mcts/actions/workflows/rust.yml/badge.svg)](https://github.com/thomasmarsh/mcts/actions/workflows/rust.yml)

Learning project for MCTS. This code is not entirely focused on efficiency, but is
strong enough to play well or better against other libraries. I use it in my [Nego](https://github.com/thomasmarsh/nego) project.

Current features:

* UCT
* RAVE
* Single/full node expansion strategies
* Arena allocation (just a `Vec`, inspired by [indextree](https://github.com/saschagrunert/indextree))

This current implementation is not efficient, but here are some things I would like to
explore:

- better testability, ergonomics, safety
- more simulation strategies, selection improvements, etc.
- DAGs / transposition tables
- online tuning

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
[minimax-rs](https://github.com/edre/minimax-rs) which has an MCTS strategy. Not
sure if this will become a published library, just a plaything, or maybe rolled
back into minimax-rs. I started from scratch and then added some utilies from
minimax-rs back in.

