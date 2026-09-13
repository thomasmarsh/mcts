"""Merge one generation's value/policy fit metadata and its
``gumbel_gate``/Edax-yardstick text output into a single JSON metrics line
for the graded Othello coordinator (``research/az-train/coordinator_othello.sh``).

Mirrors ``az_train.coordinator_metrics_c4``'s role (turn gate stdout into a
reconstructable ``log.jsonl``), parsing ``games/othello/examples/
gumbel_gate.rs``'s own output format instead of the Connect Four gate's.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_H2H_RE = re.compile(
    r"^(?P<label>head to head vs baseline|rollout anchor): candidate "
    r"(?P<wins>\d+)-(?P<draws>\d+)-(?P<losses>\d+) \(W-D-L\) over (?P<games>\d+), "
    r"score share (?P<share>[0-9.]+), Wilson LB (?P<lb>[0-9.]+)\s*-> (?P<verdict>PASS|FAIL)",
    re.MULTILINE,
)
_EDAX_RE = re.compile(
    r"^edax yardstick \(level (?P<level>\d+)\)\s*.*?games=\s*(?P<games>\d+)\s+"
    r"W-D-L (?P<wins>\d+)-(?P<draws>\d+)-(?P<losses>\d+)\s+win_rate=(?P<rate>[0-9.]+)\s+"
    r"ci=\[(?P<lo>[0-9.]+), (?P<hi>[0-9.]+)\]",
    re.MULTILINE,
)


def parse_gate_output(text: str) -> dict[str, object]:
    """Both checks (``head to head vs baseline``, ``rollout anchor``) plus
    an optional Edax yardstick line, from one ``gumbel_gate`` run's stdout."""
    checks: dict[str, object] = {}
    for m in _H2H_RE.finditer(text):
        key = "h2h" if "baseline" in m["label"] else "rollout_anchor"
        checks[key] = {
            "wins": int(m["wins"]),
            "draws": int(m["draws"]),
            "losses": int(m["losses"]),
            "games": int(m["games"]),
            "score_share": float(m["share"]),
            "wilson_lower_bound": float(m["lb"]),
            "verdict": m["verdict"],
        }
    if not checks:
        raise ValueError(f"no head-to-head/rollout-anchor result found in:\n{text}")
    edax = _EDAX_RE.search(text)
    if edax is not None:
        checks["edax_yardstick"] = {
            "level": int(edax["level"]),
            "games": int(edax["games"]),
            "wins": int(edax["wins"]),
            "draws": int(edax["draws"]),
            "losses": int(edax["losses"]),
            "win_rate": float(edax["rate"]),
            "ci_low": float(edax["lo"]),
            "ci_high": float(edax["hi"]),
        }
    return checks


def merge_generation_metrics(
    *,
    generation: int,
    wall_seconds: float,
    gate_vs_gen0_text: str,
    value_meta: dict[str, object],
    policy_meta: dict[str, object],
    gate_vs_prev_text: str | None = None,
) -> dict[str, object]:
    return {
        "generation": generation,
        "wall_seconds": round(float(wall_seconds), 1),
        "value_metrics": value_meta.get("metrics"),
        "policy_metrics": policy_meta.get("metrics"),
        "gate_vs_gen0": parse_gate_output(gate_vs_gen0_text),
        "gate_vs_prev": parse_gate_output(gate_vs_prev_text) if gate_vs_prev_text else None,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m az_train.coordinator_metrics_othello")
    ap.add_argument("--weights-meta", required=True, help="gen<N>/weights.meta.json")
    ap.add_argument("--policy-meta", required=True, help="gen<N>/policy.meta.json")
    ap.add_argument("--gate-vs-gen0", required=True, help="gumbel_gate stdout, gen vs gen0")
    ap.add_argument("--gate-vs-prev", default=None, help="gumbel_gate stdout, gen vs gen(N-1)")
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--wall-seconds", type=float, required=True)
    args = ap.parse_args(argv)

    line = merge_generation_metrics(
        generation=args.generation,
        wall_seconds=args.wall_seconds,
        gate_vs_gen0_text=Path(args.gate_vs_gen0).read_text(),
        value_meta=json.loads(Path(args.weights_meta).read_text()),
        policy_meta=json.loads(Path(args.policy_meta).read_text()),
        gate_vs_prev_text=Path(args.gate_vs_prev).read_text() if args.gate_vs_prev else None,
    )
    print(json.dumps(line, sort_keys=True))


if __name__ == "__main__":
    main()
