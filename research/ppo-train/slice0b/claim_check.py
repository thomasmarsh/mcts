"""Independent check of the `../selfplay` claim in our own yardstick (omnireset-ppo.md slice 0b).

Plays raw-policy (no search) nets against Edax levels and against each other, on Pgx's Othello rules,
from random openings, both seats. Three kinds of agents:

  selfplay:<ckpt>[:<config>]   the released Flax AZNet (128x6, BatchNorm) from ../selfplay, greedy
  ours:<bin>[:d4|literal]      one of our OTCNN001 .bin nets, greedy on masked logits (D4-averaged by default)
  edax:<level>                 the Edax build our other gates use
  random                       uniform over legal moves

Run with the ../selfplay uv environment, e.g.

  UV_PROJECT_ENVIRONMENT=<venv> uv run --project ../selfplay python research/ppo-train/slice0b/claim_check.py \\
      selfplay:../selfplay/results/checkpoints/oth-aznet-s0-final.msgpack edax:3 --games 100
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pgx

REPO = Path(__file__).resolve().parents[3]
SELFPLAY = REPO.parent / "selfplay"
sys.path.insert(0, str(SELFPLAY))
sys.path.insert(0, str(REPO / "research" / "othello-eval" / "src"))

from flax import serialization  # noqa: E402
from othello_eval.convnet import _unpack_k, predict_k  # noqa: E402
from othello_eval.ntuple import D4  # noqa: E402
from othello_eval.policy import INV  # noqa: E402
from pgx4.train import ActorCritic  # noqa: E402

NEG = -1e9
PASS = 64
ENV = pgx.make("othello")
V_STEP = jax.jit(jax.vmap(ENV.step))
V_INIT = jax.jit(jax.vmap(ENV.init))
EDAX_BIN = REPO / "games/othello/edax/vendor/bin/mEdax-native"
EDAX_DATA = REPO / "games/othello/edax/vendor/data"


def bucket(n):
    return 1 << max(0, (n - 1).bit_length())


def pad_to(a, n):
    return np.concatenate([a, np.zeros((n - len(a),) + a.shape[1:], a.dtype)]) if len(a) < n else a


class Agent:
    name = "?"

    def act(self, obs, mask):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed=0):
        self.name = "random"
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        return np.array([self.rng.choice(np.flatnonzero(m)) for m in mask])


def greedy(logits, mask):
    return np.argmax(np.where(mask, logits, NEG), axis=-1)


class SelfplayAgent(Agent):
    def __init__(self, ckpt, config=None):
        ckpt = Path(ckpt)
        cfg = json.loads(Path(config or ckpt.parent / (ckpt.name.split("-final")[0].split("-iter")[0] + "-config.json")).read_text())
        self.name = f"selfplay:{ckpt.stem}"
        self.net = ActorCritic(n_actions=65, width=cfg["width"], depth=cfg["depth"], arch=cfg["arch"])
        tmpl = self.net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8, 8, 2)))
        self.params = serialization.from_bytes(tmpl, ckpt.read_bytes())
        self.fwd = jax.jit(lambda o: self.net.apply(self.params, o)[0])

    def logits(self, obs):
        n = bucket(len(obs))
        return np.asarray(self.fwd(jnp.asarray(pad_to(obs.astype(np.float32), n))))[: len(obs)]

    def act(self, obs, mask):
        return greedy(self.logits(obs), mask)


def load_otcnn(path):
    b = Path(path).read_bytes()
    assert b[:8] == b"OTCNN001", "not an OTCNN001 file"
    h = struct.unpack("<9I", b[8:44])
    _version, _r, _c, _inp, channels, blocks, value_hidden, _pol, n = h
    w = np.frombuffer(b[44:], dtype="<f4").copy()
    assert len(w) == n
    return w, channels, blocks, value_hidden


class OursAgent(Agent):
    """OTCNN001 forward in JAX (unpacked with the numpy reference's own layout), cross-checked
    against `othello_eval.convnet.predict_k` by `--selftest`."""

    def __init__(self, path, mode="d4"):
        self.name = f"ours:{Path(path).parent.name}/{Path(path).stem}:{mode}"
        self.w, self.channels, self.blocks, self.vh = load_otcnn(path)
        self.mode = mode
        p = [jnp.asarray(t) for t in _unpack_k(self.w, self.blocks, False, self.channels, self.vh)]
        self.p = p
        self.d4 = jnp.asarray(np.stack([np.asarray(D4[s]) for s in range(8)]))
        self.inv = jnp.asarray(np.stack([np.asarray(INV[s]) for s in range(8)]))
        self.fwd = jax.jit(self._logits)

    @staticmethod
    def _conv(x, w, b, pad):
        y = jax.lax.conv_general_dilated(x, w, (1, 1), [(pad, pad)] * 2, dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return y + b[None, :, None, None]

    def _literal(self, me, opp):
        p = self.p
        x = jax.nn.relu(self._conv(jnp.stack([me, opp], 1).reshape(-1, 2, 8, 8), p[0], p[1], 1))
        at = 2
        for _ in range(self.blocks):
            r = x
            x = jax.nn.relu(self._conv(x, p[at], p[at + 1], 1))
            x = jax.nn.relu(self._conv(x, p[at + 2], p[at + 3], 1) + r)
            at += 4
        at += 6
        f = jax.nn.relu(self._conv(x, p[at], p[at + 1], 0)).reshape(-1, 64)
        return f @ p[at + 2] + p[at + 3]

    def _logits(self, me, opp):
        if self.mode == "literal":
            return self._literal(me, opp)
        n = me.shape[0]
        total = 0.0
        for s in range(8):
            cols = self.d4[s]
            total = total + self._literal(me[:, cols], opp[:, cols])[:, self.inv[s]]
        return total / 8.0

    def logits(self, obs):
        me = obs[..., 0].reshape(len(obs), 64).astype(np.float32)
        opp = obs[..., 1].reshape(len(obs), 64).astype(np.float32)
        n = bucket(len(obs))
        return np.asarray(self.fwd(jnp.asarray(pad_to(me, n)), jnp.asarray(pad_to(opp, n))))[: len(obs)]

    def act(self, obs, mask):
        lg = self.logits(obs)
        full = np.concatenate([lg, lg.mean(axis=1, keepdims=True)], axis=1)
        return greedy(full, mask)


class EdaxAgent(Agent):
    def __init__(self, level):
        self.name = f"edax-L{level}"
        self.proc = subprocess.Popen(
            [str(EDAX_BIN), "-q", "-book-usage", "off", "-eval-file", str(EDAX_DATA / "eval.dat"), "-level", str(level)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.proc.stdin.write("mode 3\n")
        self.proc.stdin.flush()

    def ask(self, me, opp):
        board = "".join("X" if a else "O" if b else "-" for a, b in zip(me, opp)) + " X"
        self.proc.stdin.write(f"setboard {board}\ngo\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            assert line, "edax closed its output"
            if line.strip().startswith("Edax plays "):
                tok = line.strip()[len("Edax plays "):].strip().lower()
                if tok in ("pa", "pass"):
                    return PASS
                return (int(tok[1]) - 1) * 8 + (ord(tok[0]) - ord("a"))

    def act(self, obs, mask):
        out = []
        for o, m in zip(obs, mask):
            if not m[:64].any():
                out.append(PASS)
                continue
            a = self.ask(o[..., 0].reshape(64) > 0, o[..., 1].reshape(64) > 0)
            assert m[a], f"{self.name} played {a}, not legal per pgx: rules/board mapping mismatch"
            out.append(a)
        return np.array(out)

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def parse_agent(spec):
    kind, _, rest = spec.partition(":")
    if kind == "random":
        return RandomAgent()
    if kind == "edax":
        return EdaxAgent(int(rest))
    if kind == "selfplay":
        ckpt, _, cfg = rest.partition("::")
        return SelfplayAgent(ckpt, cfg or None)
    if kind == "ours":
        path, _, mode = rest.partition("::")
        return OursAgent(path, mode or "d4")
    raise SystemExit(f"unknown agent {spec}")


def openings(n, plies, seed):
    """`n` distinct-ish random `plies`-ply openings as a batched pgx state."""
    rng = np.random.default_rng(seed)
    st = V_INIT(jax.random.split(jax.random.PRNGKey(seed), n))
    for _ in range(plies):
        mask = np.asarray(st.legal_action_mask)
        act = np.array([rng.choice(np.flatnonzero(m[:64])) if m[:64].any() else PASS for m in mask])
        st = V_STEP(st, jnp.asarray(act))
    assert not np.asarray(st.terminated).any(), "opening reached terminal state"
    return st


def play(a, b, games, plies, seed):
    """Agent `a` vs `b`, `games` games (pairs share an opening, seats swapped). Returns (W, D, L) for a."""
    assert games % 2 == 0
    op = openings(games // 2, plies, seed)
    st = jax.tree.map(lambda x: jnp.concatenate([x, x], 0), op)
    half = games // 2
    a_seat = np.concatenate([np.zeros(half, int), np.ones(half, int)])
    while True:
        term = np.asarray(st.terminated) | np.asarray(st.truncated)
        if term.all():
            break
        cur = np.asarray(st.current_player)
        obs = np.asarray(st.observation)
        mask = np.asarray(st.legal_action_mask)
        act = np.full(games, PASS)
        for agent, who in ((a, cur == a_seat), (b, cur != a_seat)):
            idx = np.flatnonzero(who & ~term)
            if len(idx):
                act[idx] = agent.act(obs[idx], mask[idx])
        assert all(mask[i, act[i]] for i in np.flatnonzero(~term)), "illegal action"
        new = V_STEP(st, jnp.asarray(act))
        keep = jnp.asarray(term)
        st = jax.tree.map(lambda n, o: jnp.where(keep.reshape((-1,) + (1,) * (n.ndim - 1)), o, n), new, st)
    r = np.asarray(st.rewards)
    ra = r[np.arange(games), a_seat]
    return int((ra > 0).sum()), int((ra == 0).sum()), int((ra < 0).sum())


def wilson(w, d, l, z=1.96):
    n = w + d + l
    p = (w + 0.5 * d) / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def selftest():
    """The JAX OTCNN001 forward agrees with the numpy reference `predict_k` (D4-averaged and literal)."""
    rng = np.random.default_rng(0)
    for path in [REPO / "local/output/az/othello-cnn/graded-cnn-800/gen5.cnn.bin",
                 REPO / "local/output/az/othello-cnn/run-s4-20260918-185212/gen2.cnn.bin"]:
        ag = OursAgent(path)
        n = 3
        st = openings(n, 12, 5)
        obs = np.asarray(st.observation)
        me = obs[..., 0].reshape(n, 64).astype(np.float32)
        opp = obs[..., 1].reshape(n, 64).astype(np.float32)
        _, ref = predict_k(ag.w, me, opp, ag.blocks, False, ag.channels, ag.vh)
        got = ag.logits(obs)
        print(f"{path.parent.name}/{path.stem}: max|jax-numpy| = {np.abs(got - ref).max():.2e} "
              f"(logit scale {np.abs(ref).max():.2f})")
        assert np.abs(got - ref).max() < 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="*")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None, help="append one JSON line per result here")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    a_spec, foes = args.agents[0], args.agents[1:]
    a = parse_agent(a_spec)
    for spec in foes:
        b = parse_agent(spec)
        t0 = time.time()
        w, d, l = play(a, b, args.games, args.opening_plies, args.seed)
        p, lo, hi = wilson(w, d, l)
        row = dict(a=a.name, b=b.name, games=args.games, opening_plies=args.opening_plies, seed=args.seed,
                   w=w, d=d, l=l, score=p, lo=lo, hi=hi, secs=round(time.time() - t0, 1))
        print(f"{a.name} vs {b.name}: W-D-L {w}-{d}-{l}  score {p:.3f} [{lo:.3f}, {hi:.3f}]  ({row['secs']}s)", flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if isinstance(b, EdaxAgent):
            b.close()
    if isinstance(a, EdaxAgent):
        a.close()


if __name__ == "__main__":
    main()
