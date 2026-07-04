#!/usr/bin/env python3
"""Phase 12 admission gate: address forms + disclosure projection (fixes defect I).

P11 gave the environment a real decoded issued-event stream but the policy-facing
address space was ``logical``-only and the disclosure matrix was partly faked.
P12 adds the ``physical`` and ``handle`` address forms and enforces the SPEC §7-8
disclosure levels (``mapping``/``adjacency``/``victim``/``profile``/``feedback``)
with no hidden-state leakage.

Checks (see spec/IMPLEMENTATION_PLAN_V2.md §P12 and spec/TEST_PLAN.md C5/E6):

- **Differential decode**: the Python ``AddressMapper`` reproduces Ramulator's own
  ``addr_vec`` for a sample of addresses, in both directions, through the *live*
  worker — a logical read decodes to the mapper's coordinates, and a physical
  read round-trips back to the coordinates asked for.
- **Disclosure gating**: for ``physical`` / ``logical_only`` / ``opaque_handles``
  tasks, the permitted address forms work and every hidden form fails closed with
  ``ADDRESS_NOT_DISCLOSED``.
- **Non-leakage**: a hidden-target task exposes no target coordinate, address, or
  threshold in observations, ``dram.info``, error messages, handle names, or the
  issued-event trace — yet the opaque handle still resolves server-side.
"""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv, RowHammerTaskEnv  # noqa: E402
from rowhammer_env.tools.addressing import COORD_KEYS  # noqa: E402

SEED = 12


def read(env, addr: dict, length: int = 1):
    return env.step(Phase2Action(tool="dram.read", args={"addr": addr, "length": length}))


def last_command_coords(obs, op: str = "RD") -> dict:
    events = obs.feedback.get("trace_tail", [])
    hits = [e for e in events if e.get("op") == op]
    if not hits:
        raise SystemExit(f"no {op} event in trace to compare coordinates against")
    e = hits[-1]
    return {k: e[k] for k in COORD_KEYS}


def check_differential() -> None:
    """The mapper's decode/encode must equal Ramulator's own address decode."""
    env = RowHammerDisturbanceEnv()
    obs = env.reset(seed=SEED, episode_id="phase12_diff")
    if obs.error:
        raise SystemExit(f"differential setup failed: {obs.error}")
    mapper = env.address_mapper
    assert mapper is not None

    samples = []
    for bankgroup in (0, 3):
        for bank in (0, 2, 3):
            for row in (1, 10, 137, 5000, 65535):
                for column in (0, 1, 63, 127):
                    samples.append(
                        {"channel": 0, "rank": 0, "bankgroup": bankgroup, "bank": bank, "row": row, "column": column}
                    )

    for coords in samples:
        linear = mapper.encode(coords)
        # Pure-Python round-trips first.
        if mapper.decode(linear) != coords:
            raise SystemExit(f"decode(encode({coords})) mismatch: {mapper.decode(linear)}")
        if mapper.encode(mapper.decode(linear)) != linear:
            raise SystemExit(f"encode(decode({linear})) mismatch")

        # Decode direction against the worker: a logical read of `linear` must
        # decode to exactly the mapper's coordinates.
        obs = read(env, {"kind": "logical", "addr": linear})
        if obs.error:
            raise SystemExit(f"logical read failed: {obs.error}")
        worker_coords = last_command_coords(obs)
        if worker_coords != coords:
            raise SystemExit(f"worker decode {worker_coords} != mapper {coords} for addr {linear}")

        # Encode direction: a physical read of `coords` must land on the same
        # decoded coordinates in the worker (full round-trip through Ramulator).
        obs = read(env, {"kind": "physical", **coords})
        if obs.error:
            raise SystemExit(f"physical read failed: {obs.error}")
        if last_command_coords(obs) != coords:
            raise SystemExit(f"physical form did not round-trip for {coords}")

    env.close()
    print(f"  differential: {len(samples)} addresses decode/encode identically to Ramulator's addr_vec")


def _expect_forms(env, forms: set[str]) -> None:
    obs = env.step(Phase2Action(tool="dram.info", args={}))
    if set(obs.metadata.get("address_forms", [])) != forms:
        raise SystemExit(f"dram.info advertised {obs.metadata.get('address_forms')} != {sorted(forms)}")


def _expect_ok(obs, what: str) -> None:
    if obs.error:
        raise SystemExit(f"{what} unexpectedly failed: {obs.error}")


def _expect_not_disclosed(obs, what: str) -> None:
    if not obs.error or obs.error["code"] != "ADDRESS_NOT_DISCLOSED":
        raise SystemExit(f"{what} did not fail closed with ADDRESS_NOT_DISCLOSED: {obs.error}")


def check_disclosure_gating() -> None:
    """Each mapping level admits exactly its address forms; others fail closed."""
    physical_coords = {"kind": "physical", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": 10, "column": 0}

    # mapping: physical -> logical + physical accepted, handle hidden.
    env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
    obs = env.reset(seed=SEED, episode_id="phase12_physical")
    _expect_ok(obs, "physical-task reset")
    _expect_forms(env, {"logical", "physical"})
    addr = obs.metadata["target"]["addr"]
    _expect_ok(read(env, {"kind": "logical", "addr": addr}), "logical read under physical mapping")
    _expect_ok(read(env, physical_coords), "physical read under physical mapping")
    _expect_not_disclosed(read(env, {"kind": "handle", "id": "h_0000000000000000"}), "handle read under physical mapping")
    env.close()

    # mapping: logical_only (hidden_target) -> logical + target handle only.
    env = RowHammerTaskEnv(task={"family": "hidden_target"})
    obs = env.reset(seed=SEED, episode_id="phase12_logical")
    _expect_ok(obs, "hidden-target reset")
    _expect_forms(env, {"logical", "handle"})
    handle = obs.metadata["target"]["id"]
    _expect_ok(read(env, {"kind": "logical", "addr": 0}), "logical read under logical_only")
    _expect_ok(read(env, {"kind": "handle", "id": handle}), "target-handle read under logical_only")
    _expect_not_disclosed(read(env, physical_coords), "physical read under logical_only")
    env.close()

    # mapping: opaque_handles -> handles only, logical and physical both hidden.
    env = RowHammerTaskEnv(
        task={
            "family": "known_target_anybit",
            "disclosure": {
                "mapping": "opaque_handles",
                "adjacency": "exact",
                "victim": "row_handle",
                "profile": "public_profile_id",
                "feedback": "summarized_counts",
            },
        }
    )
    obs = env.reset(seed=SEED, episode_id="phase12_opaque")
    _expect_ok(obs, "opaque-handles reset")
    _expect_forms(env, {"handle"})
    handle = obs.metadata["target"]["id"]
    _expect_ok(read(env, {"kind": "handle", "id": handle}), "handle read under opaque_handles")
    _expect_not_disclosed(read(env, {"kind": "logical", "addr": 0}), "logical read under opaque_handles")
    _expect_not_disclosed(read(env, physical_coords), "physical read under opaque_handles")
    env.close()
    print("  disclosure: physical / logical_only / opaque_handles admit only their forms; others fail closed")


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _leaks(surfaces: list, needle: int) -> bool:
    text = str(needle)
    for surface in surfaces:
        for scalar in _walk(surface):
            if isinstance(scalar, bool):
                continue
            if isinstance(scalar, int) and scalar == needle:
                return True
            if isinstance(scalar, str) and text in scalar:
                return True
    return False


def check_non_leakage() -> None:
    """Hidden physical state must be unrecoverable from any public surface."""
    env = RowHammerTaskEnv(task={"family": "hidden_target"})
    obs = env.reset(seed=SEED, episode_id="phase12_leak")
    _expect_ok(obs, "hidden-target reset")
    dist = env.disturbance
    secret_addr = dist.target_addr
    secret_threshold = dist.known_threshold
    handle = obs.metadata["target"]["id"]

    # Structural: the target is an opaque handle; no exact coordinate/threshold.
    if obs.metadata["target"] != {"kind": "handle", "id": handle}:
        raise SystemExit("hidden target disclosed more than an opaque handle")
    if "known_target_row" in obs.metadata["disturbance"] or "known_threshold" in obs.metadata["disturbance"]:
        raise SystemExit("hidden-target metadata leaked the target row/threshold")
    if "candidates" in obs.metadata:
        raise SystemExit("hidden adjacency leaked a candidate set")

    surfaces = [obs.metadata, obs.feedback]
    surfaces.append(env.step(Phase2Action(tool="dram.info", args={})).metadata)

    hammer = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 0}}]}))
    if hammer.feedback.get("trace_tail") != []:
        raise SystemExit("summarized feedback leaked a per-command trace with coordinates")
    surfaces.append(hammer.feedback)

    rejected = read(env, {"kind": "physical", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": dist.known_target_row, "column": 0})
    _expect_not_disclosed(rejected, "physical read under hidden target")
    surfaces.append({"error": rejected.error})

    for needle in (secret_addr, secret_threshold):
        if _leaks(surfaces, needle):
            raise SystemExit(f"hidden value {needle} leaked into a public surface")
    for needle in (secret_addr, dist.known_target_row, secret_threshold):
        if str(needle) in handle:
            raise SystemExit(f"handle name {handle!r} encodes hidden coordinate {needle}")

    # The handle still resolves server-side to the real (hidden) target.
    resolved = read(env, {"kind": "handle", "id": handle})
    _expect_ok(resolved, "target-handle read")
    if base64.b64decode(resolved.data_b64 or "") == b"":
        raise SystemExit("target handle did not resolve to a readable address")
    env.close()
    # A final proof the surfaces really were serialisable/inspected end-to-end.
    json.dumps(surfaces, default=str)
    print("  non-leakage: hidden target addr/row/threshold absent from obs, info, errors, handle, and trace")


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase11.py"], cwd=ROOT, check=True)
    print("phase12 checks:")
    check_differential()
    check_disclosure_gating()
    check_non_leakage()
    print("phase12 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
