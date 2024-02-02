What about something like:

```
  enum State<A> {
    Active(ActiveState<A>),
    Terminal(TerminalState<A>),
  }

  fn gen_moves(state: &ActiveState<S>) -> State<S>;

  fn get_reward(state: &TerminalState<S>) -> i32;
```

It makes move generation more expensive. Would need to benchmark. We do a lot of
is_terminal checks, many more than gen_moves.
