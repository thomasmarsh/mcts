# Repository layout

The workspace keeps durable source and documentation at the repository root,
and keeps machine-local output under `local/`, which is ignored by Git.

- `crates/` contains reusable Rust libraries.
- `games/` contains game crates and their shared game support.
- `apps/` contains runnable products: the server, browser UI, and game-host
  protocol executable.
- `tools/` contains operational commands, including the tuner and GDL compiler.
- `examples/` contains reusable Cargo examples owned by the root manifest.
- `research/` contains durable research notes, the project roadmap, and
  standalone evaluation experiments.
- `docs/` contains project documentation.
- `local/runs/` contains run databases and experiment output.
- `local/experiments/` contains local analysis artifacts such as bakeoffs.
- `local/generated/` contains reproducible generated assets such as opening
  books.
- `local/work/` contains personal working material.

New generated or machine-specific output belongs in `local/`; checked-in source,
documentation, and reusable tools should use one of the durable directories
above.
