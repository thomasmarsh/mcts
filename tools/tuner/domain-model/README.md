# tuner-domain-model

This is a proof-of-concept Haskell model of the game-strategy tuner. It is a
place to make the vocabulary, boundaries, and invariants of the real tuner
explicit. Most functions intentionally have no implementation; `cabal build`
checks that the types still describe one coherent system.

Start with [TUTORIAL.md](TUTORIAL.md). It follows one candidate through a run
and explains why the model insists on paired games, common task prefixes,
held-out validation, and an auditable decision trail.

## What the model commits us to

- A candidate is validated, canonicalized, fingerprinted, and immutable. New
  games add evidence to that candidate rather than creating a new trial.
- A seat-swapped pair is the atomic measurement. It contains two games of the
  same case, with the candidate playing each seat once.
- Candidates may be compared only in the same objective epoch, phase, task
  prefix, and search effort. An old panel, a shorter prefix, or a different
  effort is a different measurement.
- Active candidates finish the same nested block of tasks before a race makes
  an elimination or deepening decision.
- One allocator spends all compute. Introducing candidates, deepening races,
  diagnostics, and validation do not each have a competing scheduler.
- The final result is a ranked set, not necessarily a single winner. It carries
  uncertainty and may preserve practical ties.
- Early elimination is a claim to be tested. Shadow decisions and randomized
  audit continuations make its mistakes visible.
- A frozen manifest and append-only evidence log make a run inspectable and
  replayable.

## Relationship to the implementation

The running tuner is under [`../src/tuner_cli/`](../src/tuner_cli/). This model
is deliberately not a second implementation of it. The Python code performs
the work; these types make the shape of that work and its contracts easier to
read, discuss, and keep aligned.

The model uses a small canonical JSON representation so identities and
fingerprints can agree across the Haskell and Python boundaries. Integers and
floating-point values remain distinct for that reason.

## Exploring the model

```sh
cabal build
cabal repl
```

In GHCi, `:browse MyLib` gives the public vocabulary, and `:info
ObservationContext` is a useful first stop. Function bodies which are
`undefined` are intentional. A few small helpers are implemented because they
state arithmetic or representation facts, such as pair utility and search
effort construction.

The package does not run games. The `Target` type marks that boundary: it
describes the questions the tuner may ask of a game executable, while leaving
execution to the real system.
