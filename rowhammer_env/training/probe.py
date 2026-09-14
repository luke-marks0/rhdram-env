"""Training-only reward shaping for the discovery families, kept strictly below a
real success so probing can never substitute for a flip.

The bonus rewards *making a decisive bank-conflict measurement*: alternately
reading exactly the victim and one candidate, repeated, whose real ``timing_digest``
lands cleanly in one regime (``acts_delta == 0`` -> different bank; ``acts_delta >=
2`` -> same bank). It is outcome-neutral (same-bank and different-bank pay the same),
derived only from the trusted digest the policy already sees at ``full_trace``, and
each distinct candidate is credited at most once. Evaluation/benchmark scoring never
adds it (see the trainer/eval callers). This mirrors the removed ``llm/shaping.py``.

Torch-free: operates on the plain ``TurnRecord`` list a rollout produces.
"""
from __future__ import annotations

import json
from typing import Any

from .metrics import TurnRecord

PROBE_ROW_COUNT = 2       # a bank-conflict probe compares exactly two rows
MIN_READS_PER_ROW = 2     # a decisive probe alternates (repeats), not a one-shot warm
PER_PROBE_BONUS = 0.25    # four clean unique probes saturate the unweighted term at 1.0


def _timing_digest(turn: TurnRecord) -> dict[str, Any]:
    digest = turn.feedback.get("timing_digest") if isinstance(turn.feedback, dict) else None
    return digest if isinstance(digest, dict) else {}


def is_decisive_probe(turn: TurnRecord) -> bool:
    """A ``dram.issue`` whose real digest is a pairwise, repeated, cleanly-separated
    bank-conflict measurement. Ambiguous small ``acts_delta`` does not count."""
    if turn.tool != "dram.issue":
        return False
    digest = _timing_digest(turn)
    per_addr = digest.get("per_addr_hits") or {}
    if not isinstance(per_addr, dict) or len(per_addr) != PROBE_ROW_COUNT:
        return False
    reads = [int(b.get("hits", 0) or 0) for b in per_addr.values() if isinstance(b, dict)]
    if len(reads) != PROBE_ROW_COUNT or min(reads, default=0) < MIN_READS_PER_ROW:
        return False
    acts_delta = int(digest.get("acts_delta", 0) or 0)
    return acts_delta == 0 or acts_delta >= PROBE_ROW_COUNT


def count_decisive_probes(turns: list[TurnRecord]) -> int:
    return sum(1 for t in turns if is_decisive_probe(t))


def _addr_token(addr: Any) -> str:
    if type(addr) is int:
        addr = {"kind": "logical", "addr": addr}
    if not isinstance(addr, dict):
        return ""
    if addr.get("kind") == "logical":
        return f"logical:{addr.get('addr')}"
    if addr.get("kind") == "handle":
        return f"handle:{addr.get('id')}"
    return json.dumps(addr, sort_keys=True)


def unique_probe_candidates(turns: list[TurnRecord], metadata: dict[str, Any]) -> set[str]:
    """Distinct candidates the policy measured against the victim with a decisive probe.

    Keyed by the disclosed candidate token the policy itself supplied — never a
    resolved address or a server-private role — so re-probing the same candidate,
    a wide hammer, or an errored turn cannot farm the bonus.
    """
    objective = metadata.get("objective") or {}
    victim = _addr_token(metadata.get("target") or objective.get("target"))
    candidates = {_addr_token(c) for c in metadata.get("candidates", [])}
    measured: set[str] = set()
    for turn in turns:
        if not is_decisive_probe(turn):
            continue
        rows = _probe_rows(turn.args)
        tokens = {_addr_token(r) for r in rows}
        if victim not in tokens:
            continue
        other = tokens - {victim}
        measured |= other & candidates
    return measured


def _probe_rows(args: dict[str, Any]) -> list[Any]:
    """The two distinct rows a probe alternated, from the requested command list."""
    commands = args.get("commands") if isinstance(args, dict) else None
    if not isinstance(commands, list):
        return []
    rows: list[Any] = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        if cmd.get("op") == "HAMMER" and isinstance(cmd.get("rows"), list):
            rows.extend(cmd["rows"])
        elif cmd.get("op") == "RD" and "addr" in cmd:
            rows.append(cmd["addr"])
    # Deduplicate while preserving order.
    seen: list[Any] = []
    for r in rows:
        if r not in seen:
            seen.append(r)
    return seen


def probe_shaping(turns: list[TurnRecord], metadata: dict[str, Any], *, per_probe: float = PER_PROBE_BONUS) -> float:
    """Unweighted shaping reward in ``[0, 1]``: ``min(1, per_probe * unique_probes)``.

    The trainer multiplies this by ``grpo.probe_shaping`` (validated strictly below 1,
    a real success) before adding it to the sparse reward.
    """
    n = len(unique_probe_candidates(turns, metadata))
    return min(1.0, float(per_probe) * n)
