#!/usr/bin/env bash
#
# Build Edax (abulmo/edax-reversi) locally for use as an external Othello
# strength yardstick, and fetch its evaluation weights.
#
# Everything lands under games/othello/edax/vendor/ (gitignored). Re-running
# is cheap: the clone/build/fetch steps each skip if their output already
# exists. Pass --force to rebuild from scratch.
#
# Outputs:
#   vendor/bin/mEdax-native   the engine binary (host arch, -O3 -flto)
#   vendor/data/eval.dat      pattern-evaluation weights (~13.3 MB)
#
# The match harness (games/othello/examples/edax_match.rs) drives this binary
# over its native line protocol -- see README.md in this directory for the
# exact request/response strings and the one non-obvious gotcha (Edax aborts
# an in-progress `go` search the moment another line is queued on stdin, so
# the driver must write one command and read its reply before writing the
# next; piping a whole script at once silently yields no move).

set -euo pipefail

# Pinned upstream commit: tag v4.6.
EDAX_SHA=14f048c05ddfa385b6bf954a9c2905bbe677e9d3
EDAX_REPO=https://github.com/abulmo/edax-reversi
# v4.6 release tarball; we only extract data/eval.dat from it (the weights
# are platform-independent, so the linux archive is fine on macOS and avoids
# needing a 7-Zip extractor for the separate eval.7z asset).
EVAL_TARBALL_URL="https://github.com/abulmo/edax-reversi/releases/download/v4.6/edax-4.6-linux-x86.tar.gz"
EVAL_DAT_SHA256=f8b2299612d9fa4414157e70e932636e33111c2602d0c2fc382a7d90ef21b792

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vendor="$here/vendor"
src="$vendor/src"
bin="$vendor/bin"
data="$vendor/data"

force=0
[[ "${1:-}" == "--force" ]] && force=1

if [[ $force == 1 ]]; then
  rm -rf "$vendor"
fi

# --- 1. clone at the pinned commit --------------------------------------------
if [[ ! -d "$vendor/.git" ]]; then
  echo "cloning $EDAX_REPO @ $EDAX_SHA"
  git clone --filter=blob:none "$EDAX_REPO" "$vendor"
  git -C "$vendor" checkout --quiet "$EDAX_SHA"
else
  have="$(git -C "$vendor" rev-parse HEAD)"
  if [[ "$have" != "$EDAX_SHA" ]]; then
    echo "vendor checkout is $have, expected $EDAX_SHA -- run with --force to reset" >&2
    exit 1
  fi
fi

# --- 2. build ----------------------------------------------------------------
os=linux
exe=lEdax-native
if [[ "$(uname -s)" == "Darwin" ]]; then
  os=osx
  exe=mEdax-native
fi

if [[ $force == 1 || ! -x "$bin/$exe" ]]; then
  echo "building edax ($os, ARCH=native)"
  mkdir -p "$bin"
  # The Makefile's `build` target is a single clang invocation over all.c.
  make -C "$src" build ARCH=native OS="$os"
fi
[[ -x "$bin/$exe" ]] || { echo "build produced no $bin/$exe" >&2; exit 1; }

# --- 3. fetch eval.dat -----------------------------------------------------
mkdir -p "$data"
if [[ $force == 1 || ! -f "$data/eval.dat" ]]; then
  echo "fetching eval weights"
  tmp="$(mktemp)"
  curl -fsSL -o "$tmp" "$EVAL_TARBALL_URL"
  tar xzf "$tmp" -O data/eval.dat > "$data/eval.dat"
  rm -f "$tmp"
fi
got="$(shasum -a 256 "$data/eval.dat" | cut -d' ' -f1)"
if [[ "$got" != "$EVAL_DAT_SHA256" ]]; then
  echo "eval.dat sha256 mismatch: got $got, expected $EVAL_DAT_SHA256" >&2
  exit 1
fi

# --- 4. self-test ----------------------------------------------------------
# Ask Edax for its first move from the standard opening. Note the spaced
# writes: a single `printf 'go\nquit\n'` would let `quit` land on stdin
# before the search finishes, and Edax aborts the search (returning no move).
echo
echo "self-test: level 3 move from the opening position"
reply="$(
  { printf 'mode 3\n';   sleep 0.3
    printf 'setboard ---------------------------O*------*O---------------------------- *\n'; sleep 0.3
    printf 'go\n';        sleep 2
    printf 'quit\n';      sleep 0.3
  } | "$bin/$exe" -q -eval-file "$data/eval.dat" -book-usage off -level 3 2>&1 \
    | grep -a 'Edax plays' || true
)"
if [[ -z "$reply" ]]; then
  echo "self-test FAILED: no 'Edax plays' line" >&2
  exit 1
fi
echo "  $reply"
echo
echo "edax binary: $bin/$exe"
echo "eval weights: $data/eval.dat"
echo "ok"
