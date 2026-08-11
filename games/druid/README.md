# Druid

Druid is a connection game designed by Cameron Browne:
<http://cambolbro.com/games/druid/>.

The game is hard for MCTS, and so probably a good benchmark.

## Implementation issues

- No tuning has been done yet.
- MCTS-Solver might help in the more tactical situations.
- Board size is stored as a global const, but should be some game context.
- `G::gen_moves` can fail by producing an empty set when it has hit the
  ceiling.
- `G::gen_moves` and `G::is_terminal` are expensive.
- `max_depth` is helpful but I think reduces the quality of playouts.

You can see the live grid rendered by this crate's test-binary / web front end;
the rules and the designer's own guidance on search follow.

## Designer guidance (Cameron Browne, 2013)

When asked about MCTS issues he said the following. [Email correspondence,
January 2013]

> One approach is to use RAVE or other enhancements to improve the efficiency
> of UCT, but as the paper shows even RAVE does not always work, and this could
> take a lot of trial and error. Generally the better approach is to add some
> heuristics to the playouts, to make each playout more realistic, i.e. more like
> moves that people would actually make during a game. For example, adding forced
> moves due to bridge intrusions solved the problem with Hex.
>
> Suitable heuristics for Druid might include:
> 1. If the opponent's last move threatens to build on one of your pieces, make a
>    blocking move with high probability.
> 2. If the opponent's last move intrudes into one part of a fork virtually
>    connecting two of your pieces, then make the corresponding fork move to save
>    the connection with high probability.
> 3. Make moves that threaten the opponent's best connection with high probability.
> 4. Higher is better!
>
> Note that I say "with high probability" rather than applying that same move
> every time, so there is still a bit of randomness in the playouts, otherwise
> you could trick the AI into choosing the wrong move every time. Monte Carlo
> search is all about playing the odds over large numbers of simulations, so
> probabilistic approaches are generally best.

When asked about an evaluation function for minimax, and difficultied on modeling
connectedness, he said:

> Do you mean the problem is that connections aren't permanent, i.e. they
> can't be relied upon because they can be built over? If so, then a probabilistic
> model might help: assign each adjacency a probability between 0 and 1 based
> on how likely it is to survive. So if the opponent has no immediate chance of
> breaking that connection in the next few moves its probability will be high (say
> 0.95), but if the opponent can bridge over it next move then the probability
> might be say 0.25, and if the opponent has a fork that guarantees them cutting
> a connection regardless of what you do then its probability will be almost
> 0 (maybe 0.05 to indicate that there still is a connection there, however
> tenuous). Some connections might be guaranteed (probability 1) but proving this
> could be a tricky problem in itself.
>
> Then when you have the probability for each adjacent step, the strength
> of a connection from one side to the other is the product of the associated
> probabilities for the steps along that path. This is the main difference between
> Hex and Druid, apart from the hex/square topology: connections are permanent
> (probability 1) in Hex but not in Druid.
>
> Another way to improve connection tests might be to identify virtual connections
> (two nearby pieces that are not physically connected but which the opponent
> can't block) and give then a high adjacency value, much like the good Hex
> players count bridge connections and edge templates as "connected" for the sake
> of their connectivity tests.
>
> [...]
>
> I'd start with the path probability mentioned above for an evaluation
> function, i.e. fitness = your_best_path_prob / opponent's_best_path_prob.
>
> Then you could look at all of your best paths to connection and all of your
> opponent's best paths to connection, and look for key cells that most of these
> paths flow through.
>
> You could also incorporate some of the heuristics I mention above.
>
> As for UCT vs AB search, that's hard to say -- Druid is a difficult game!
> But I've found that humans can't plan ahead reliably more than a few moves
> due to the confusing 3D element, so perhaps a simple AB search could be quite
> effective, assuming that your evaluation function is realistic.

## Representation

Druid is defined here once (`State`, connectivity, Zobrist hashing, move
cache, playout heuristics) and exposed through two *move encodings* selected
by the `Druid<M>` type parameter (see `src/moves.rs`):

- `Split` (the default / shipped): a whole-turn placement is offered as the
  linearized `Piece`/`Orientation`/`Cell` sub-action sequence, tracked by
  `State::pending`. This is what the server binary and AI presets play.
- `Flat` (pre-move-splitting snapshot): a `PlacedPiece` is the whole action,
  with `State::pending` always left at `Pending::None`. Kept solely for
  `examples/strength_move_splitting.rs` to pit the two representations
  against each other in one binary.