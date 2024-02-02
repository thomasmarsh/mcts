# MCTS
[![Rust](https://github.com/thomasmarsh/mcts/actions/workflows/rust.yml/badge.svg)](https://github.com/thomasmarsh/mcts/actions/workflows/rust.yml)

Playing around with some MCTS stuff. Some code and general approaches come from
[minimax-rs](https://github.com/edre/minimax-rs) which has an MCTS strategy. Not
sure if this will become a published library, just a plaything, or maybe rolled
back into minimax-rs.

I started this because I wanted to understand MCTS a bit better. I was originally
planning to extend minimax-rs, but ended up starting from scratch and then adding
back some elements of minimax-rs's threading and utility libraries.

Although minimax-rs has good lock-free root parallelism, it uses MCTS-Solver, which
is more effective at end game than for general use. Additionally, it uses full
expansion of each node rather than single expansion, which can reduce performance
in games with large branching factors.

This current implementation is not efficient, but here are some things I would like to
explore:

- AMAF / RAVE and other selection improvements
- more simulation strategies
- DAGs / transposition tables
- using an arena for storage
- online tuning
- better testability, ergonomics, safety

Other alternatives:
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
  support, but says it is experimental.
- [OxyMcts](https://github.com/Sagebati/OxyMcts): seems to be a vanilla UCT client
