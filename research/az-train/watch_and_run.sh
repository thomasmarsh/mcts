#!/usr/bin/env bash
# Memory watchdog for anything that can exhaust this machine's RAM (C128/B6-
# scale self-play, training, gating). Runs a command in its own process
# group, polls vm_stat once per interval, and SIGKILLs the whole group the
# moment system-available memory falls under a floor -- before macOS starts
# thrashing, which on this 8GB machine has twice ended in a hard reboot.
#
#   research/az-train/watch_and_run.sh [--floor-mb 900] [--interval 1] \
#       [--log watch.log] -- <command> [args...]
#
# "Available" is (free + inactive) pages, the reclaimable-cache-aware number
# Activity Monitor's own reading agrees with; raw free pages alone understate
# real headroom on macOS. Each sample (time, available MB, the group's summed
# RSS MB) goes to --log (default: stderr) so a run's memory shape can be
# read afterwards. Exit status is the command's own, or 137 if this script
# killed it.
set -uo pipefail

floor_mb=900
interval=1
log=/dev/stderr
while [ $# -gt 0 ]; do
  case "$1" in
    --floor-mb) floor_mb=$2; shift 2 ;;
    --interval) interval=$2; shift 2 ;;
    --log) log=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown flag $1 (put the command after --)" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [--floor-mb N] [--interval S] [--log PATH] -- command [args...]" >&2; exit 2; }

available_mb() {
  vm_stat | awk '
    /page size of/ { gsub(/[^0-9]/, "", $8); page = $8 }
    /^Pages free/ { gsub(/\./, "", $3); free = $3 }
    /^Pages inactive/ { gsub(/\./, "", $3); inactive = $3 }
    END { printf "%d\n", (free + inactive) * page / 1048576 }'
}

group_rss_mb() {
  ps -axo pgid=,rss= | awk -v g="$1" '$1 == g { kb += $2 } END { printf "%d\n", kb / 1024 }'
}

set -m
"$@" &
pid=$!
set +m

killed=0
while kill -0 "$pid" 2>/dev/null; do
  avail=$(available_mb)
  echo "$(date +%H:%M:%S) available_mb=$avail group_rss_mb=$(group_rss_mb "$pid")" >>"$log"
  if [ "$avail" -lt "$floor_mb" ] && kill -0 "$pid" 2>/dev/null; then
    echo "$(date +%H:%M:%S) WATCHDOG: available ${avail}MB < floor ${floor_mb}MB -- killing process group $pid" | tee -a "$log" >&2
    kill -KILL -- "-$pid" 2>/dev/null
    killed=1
    break
  fi
  sleep "$interval"
done

wait "$pid" 2>/dev/null
status=$?
if [ "$killed" -eq 1 ]; then exit 137; fi
exit "$status"
