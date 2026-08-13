"""Per-episode address-mapper selection for discovery families (P24).

The DRAMA threat model (`spec/TIER2_DISCOVERY_PLAN.md` §4.1) requires that a
policy given a victim's *numeric* address cannot compute which candidate rows are
same-bank neighbours — physical adjacency must be reverse-engineered through the
bank-conflict timing channel, exactly as on real hardware where the controller's
bank-select function is undocumented.

Ramulator v2.1.0's stock mappers do **not** provide this: `RoBaRaCoCh` keeps the
bank a pure low-bit slice, and — contrary to the original plan — `MOP4CLXOR` XORs
*column* bits into the bank index (cache-line interleaving), so `addr + row_stride`
stays in the *same bank* under both. This module therefore selects, for discovery
families, the authored `RoBaRaCoChRowXOR` mapper
(`cpp/ramulator_extensions/row_xor_addr_mapper.cpp`), which XORs the *row* bits
into the bank/bankgroup index — a real, documented controller behaviour (Pessl et
al., DRAMA, USENIX Security 2016), not fabricated physics (SPEC §2). See
`docs/adr-0004-secret-address-mapping.md`.

The active mapper (and its seedable `xor_offset`) is a per-episode secret sourced
deterministically from `(task_id, seed)`; it is never disclosed to the policy.
Non-discovery families keep the public `RoBaRaCoCh` base config unchanged.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
from typing import Mapping

# The public default mapper: the base worker YAML ships it, non-discovery families
# use it unchanged, and it is the only mapper the Python projection in
# ``tools/addressing.py`` reproduces (so ``physical`` addressing is admitted only
# for it — see ``is_python_projectable``).
DEFAULT_MAPPER = "RoBaRaCoCh"

# Real, row->bank-scrambling mappers whose adjacency is not computable from the
# linear address. Every entry uses the authored ``RoBaRaCoChRowXOR`` (bit 0 of Row
# always folds into Bank, so ``victim ± row_stride`` lands in a *different* bank on
# every episode); the seedable ``xor_offset`` varies the BankGroup scramble as the
# per-episode secret. The set is intentionally all-scattering so a numeric-address
# control (``victim ± row_stride``) fails on every episode (P25).
SECRET_MAPPERS: tuple[tuple[str, dict[str, int]], ...] = tuple(
    ("RoBaRaCoChRowXOR", {"xor_offset": off}) for off in (0, 1, 2, 3, 5, 7)
)


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Publish ``text`` at ``path`` atomically, so a reader never sees a partial file.

    Derived worker YAMLs live at a *content-addressed* path, so concurrent resets that
    made the same choice race to write byte-identical content. Writing in place is
    still unsafe: ``write_text`` truncates first, and a worker launched by another
    session can open the empty/partial file in that window and die with a Ramulator
    configuration error. Writing to a unique temporary file in the same directory and
    ``os.replace``-ing it makes publication a single rename — readers see either the
    old complete file or the new complete file, never a torn one. Concurrent writers
    are safe precisely because the bytes are identical.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def is_python_projectable(mapper_impl: str) -> bool:
    """Whether ``tools/addressing.py``'s Python RoBaRaCoCh projection matches the worker.

    Only the public ``RoBaRaCoCh`` mapper is reproduced in Python; a secret mapper
    must be decoded through the worker ``DECODE`` op, and ``physical``-form
    addressing / ``_physical_target`` must fail closed for it (P24 task 4).
    """
    return mapper_impl == DEFAULT_MAPPER


def select_secret_mapper(task_id: str, seed: int) -> tuple[str, dict[str, int]]:
    """Deterministically pick a per-episode secret mapper from ``(task_id, seed)``.

    Returns a *fresh* params dict each call. ``SECRET_MAPPERS`` holds the admitted
    seed-derived choices; handing out its dict by reference would let an accidental
    trusted-side mutation (the environment exposes it as ``_active_mapper_params``)
    rewrite the admitted value for every later episode in the process, so a
    supposedly pure selection would depend on mutable state surviving reset.
    @spec:invariant-determinism
    """
    digest = hashlib.sha256(f"{task_id}:{seed}:mapper".encode()).digest()
    impl, params = SECRET_MAPPERS[int.from_bytes(digest[:8], "big") % len(SECRET_MAPPERS)]
    return impl, dict(params)


def worker_config_for_mapper(
    base_config_path: pathlib.Path,
    mapper_impl: str,
    params: Mapping[str, int] | None = None,
    *,
    out_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Generate (or select) a worker YAML whose controller uses ``mapper_impl``.

    The default public mapper with no params reuses the base config verbatim, so
    the non-discovery path is byte-identical to before. A secret mapper writes a
    derived YAML that swaps only the ``addr_mapper`` node — every other field
    (geometry, timings, plugins) is untouched, so the disclosed geometry (P21)
    stays honest and the disturbance model is unaffected. The derived path lives
    server-side only; its name is a hash of the choice, so the secret ``xor_offset``
    never appears in a policy-visible field.
    """
    params = dict(params or {})
    _validate_mapper_params(mapper_impl, params)
    if mapper_impl == DEFAULT_MAPPER and not params:
        return base_config_path

    import yaml

    out_dir = out_dir or base_config_path.parent / "mappers"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        repr((str(base_config_path), mapper_impl, sorted(params.items()))).encode()
    ).hexdigest()[:12]
    out_path = out_dir / f"{base_config_path.stem}.map.{digest}.yaml"

    node = {"impl": mapper_impl, **params}
    config = yaml.safe_load(base_config_path.read_text())
    for controller in config["memory_system"]["controllers"]:
        controller["addr_mapper"] = dict(node)
    atomic_write_text(out_path, yaml.safe_dump(config, sort_keys=False))
    return out_path


# Widest Row field any admitted standard publishes; ``xor_offset`` must stay inside it
# so the C++ shift is defined and the BankGroup scramble is not a silent no-op. The
# mapper re-validates against the *actual* Row width at init, where the geometry is
# known; this is the Python-side fail-closed check on the admitted parameter set.
MAX_XOR_OFFSET = 31


def _validate_mapper_params(mapper_impl: str, params: Mapping[str, int]) -> None:
    """Reject a mapper parameter set the worker cannot safely apply (fail closed).

    ``RoBaRaCoChRowXOR`` right-shifts a 32-bit Row value by ``xor_offset``; C++ leaves
    a shift at or beyond the operand width undefined, so an oversized offset would make
    the BankGroup decode vary by compiler/build instead of being rejected. The shipped
    ``SECRET_MAPPERS`` offsets are all well inside the bound, so this guards a
    hand-written or future config, not the discovery path.
    """
    if mapper_impl != "RoBaRaCoChRowXOR":
        if params:
            raise ValueError(f"BAD_SCHEMA:mapper {mapper_impl!r} takes no parameters")
        return
    unknown = set(params) - {"xor_offset"}
    if unknown:
        raise ValueError(f"BAD_SCHEMA:unknown mapper parameter(s) {sorted(unknown)}")
    offset = params.get("xor_offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("BAD_SCHEMA:xor_offset must be an integer")
    if not 0 <= offset <= MAX_XOR_OFFSET:
        raise ValueError(f"BAD_SCHEMA:xor_offset must be in [0,{MAX_XOR_OFFSET}], got {offset}")
