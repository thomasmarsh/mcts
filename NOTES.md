# Implementation Notes

## Type Safety

In general, there are some opportunies for [parsing vs. validation](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/).

What about something like:

```rust
  enum State<A> {
    Active(ActiveState<A>),
    Terminal(TerminalState<A>),
  }

  fn gen_moves(state: &ActiveState<S>) -> State<S>;

  fn get_reward(state: &TerminalState<S>) -> i32;
```

It makes move generation more expensive. Would need to benchmark. We do a lot of
is_terminal checks, many more than gen_moves.

Alternatively, more in keeping with the current approach, is to to check if
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

## Terminology

I'm using "move" and "action" synonymously. I prefer "action", and may unify the
terminology around the more generic term. 
