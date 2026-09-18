#!/usr/bin/env bash
# Unattended-launch wrapper for coordinator_othello_cnn.sh: backgrounds it
# (nohup + disown) so a multi-hour Gumbel self-play run survives the
# launching terminal closing or the launching Claude Code session ending --
# no LLM session needs to stay open for this to keep running. Everything
# this script does, `coordinator_othello_cnn.sh` already does in the
# foreground; this only adds the backgrounding/PID-file/timestamped-RUN_DIR
# convenience on top for a genuinely walk-away launch.
#
#   bash research/az-train/launch_othello_cnn_run.sh
#
# All of coordinator_othello_cnn.sh's own env knobs (GAMES, GENS, SIMS,
# EPOCHS, GATE_GAMES, EDAX_BINARY, REPLAY_WINDOW, START, ...) pass through
# unchanged and override the defaults below; only RUN_DIR gets a default
# that includes a timestamp so repeated launches don't collide.
#
# The defaults below (GAMES/SIMS/GENS/EPOCHS/REPLAY_WINDOW/GATE_GAMES) favor
# many small generations over a few heavy ones: more self-play iterations
# with frequent, cheap updates beat a bigger inner training loop. Sized from
# measurements at the production C128/B6 geometry on the 8GB development M1:
#   - batched self-play: ~6.5 s/game (64 games measured in 7 min through
#     this coordinator; an 800-game pass measured the same) -> GAMES=200 is
#     ~22 min.
#   - training at BATCH_SIZE=32: ~34 ms/step, ~1.05 ms per position per
#     epoch; a generation holds ~61 positions per game, so at the
#     REPLAY_WINDOW=8 cap (~98k positions) 30 epochs is ~50 min per fit
#     attempt. A stalled (dead-seed) attempt is cut off after 20 epochs
#     (~35 min at the cap); at C128/B6/batch 32 roughly half of seeds
#     die, so expect one or two of those per generation.
#   - gates use the per-node engine: ~24 s per game (head-to-head, rollout
#     anchor and Edax each play GATE_GAMES) -> GATE_GAMES=10 is ~12 min for
#     the vs-gen0 gate and ~8 min for the vs-previous one.
# That is roughly 1.5h per generation at the window cap before retries,
# ~30h for GENS=20. Check `$RUN_DIR/log.jsonl` after generation 0 -- its
# `wall_seconds` is the real number -- and adjust.
#
# CHUNK_SIZE (MLX batch cap per forward call) is deliberately not defaulted
# here: coordinator_othello_cnn.sh's own default (64) needs ~2.3GB of
# headroom at C128/B6; lower it (e.g. CHUNK_SIZE=16, ~0.6GB) if the machine
# has less free memory. Run large jobs under research/az-train/
# watch_and_run.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

RUN_DIR=${RUN_DIR:-local/output/az/othello-cnn/run-$(date +%Y%m%d-%H%M%S)}
export RUN_DIR
export DEVICE=${DEVICE:-mps}
export EDAX_BINARY=${EDAX_BINARY:-games/othello/edax/vendor/bin/mEdax-native}
export EDAX_DATA_DIR=${EDAX_DATA_DIR:-games/othello/edax/vendor/data}
export GAMES=${GAMES:-200}
export SIMS=${SIMS:-32}
export GATE_GAMES=${GATE_GAMES:-10}
export GENS=${GENS:-20}
export EPOCHS=${EPOCHS:-30}
export REPLAY_WINDOW=${REPLAY_WINDOW:-8}

mkdir -p "$RUN_DIR"

nohup bash research/az-train/coordinator_othello_cnn.sh > "$RUN_DIR/launch.log" 2>&1 &
pid=$!
disown

echo "$pid" > "$RUN_DIR/launch.pid"

cat <<EOF
Launched (PID $pid, backgrounded and disowned -- safe to close this
terminal or end this session).

  RUN_DIR: $RUN_DIR

Progress, while it runs or after you come back:
  tail -f $RUN_DIR/launch.log            # raw stdout/stderr of the whole run
  tail -f $RUN_DIR/log.jsonl             # one line per completed generation
                                          # (wall clock, positions, value/policy
                                          # metrics, both gate results)
  tail -f $RUN_DIR/gen<N>.epochs.jsonl   # live per-epoch val trace for
                                          # whichever generation is fitting now

Is it still running:
  kill -0 $pid && echo running || echo done

Stop it:
  kill $pid

Resume after an interrupt: check $RUN_DIR/log.jsonl for the last completed
generation N, then re-run with the same RUN_DIR and START=N (the completed
gen<N>.cnn.bin checkpoints are reused, not refit):
  RUN_DIR=$RUN_DIR START=<N> bash research/az-train/launch_othello_cnn_run.sh
EOF
