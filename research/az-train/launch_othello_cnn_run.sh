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
# GAMES/SIMS/GATE_GAMES default lower here than coordinator_othello_cnn.sh's
# own bare defaults (800/32/100): self-play is single-threaded, one board
# position evaluated per `CnnValueNet` call with no batching, and a live
# measurement on this machine at the current (4-residual-block) geometry
# put self-play throughput around 24s/game at 32 sims -- 800 games alone
# would run past 5 hours before training or gating even start. The values
# below (measurement-scaled, not independently re-measured at these exact
# settings) target a single generation finishing in about an hour, so a
# several-hour unattended run covers a handful of generations rather than
# stalling on generation 0. Check `$RUN_DIR/log.jsonl` after generation 0
# completes -- its `wall_seconds` is the real number -- and adjust GENS
# (via a resumed relaunch, see below) if it differs a lot from this
# estimate.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

RUN_DIR=${RUN_DIR:-local/output/az/othello-cnn/run-$(date +%Y%m%d-%H%M%S)}
export RUN_DIR
export DEVICE=${DEVICE:-mps}
export EDAX_BINARY=${EDAX_BINARY:-games/othello/edax/vendor/bin/mEdax-native}
export EDAX_DATA_DIR=${EDAX_DATA_DIR:-games/othello/edax/vendor/data}
export GAMES=${GAMES:-200}
export SIMS=${SIMS:-16}
export GATE_GAMES=${GATE_GAMES:-40}
export GENS=${GENS:-6}

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
