"""End-to-end profile build: ingest -> canonicalize -> fit -> validate -> sign -> write.

Reward-relevant simulator code never runs here; this only produces the signed,
held-out-validated profile package that a later phase loads. Every gate fails closed:
a hash mismatch, a count mismatch, a non-finite fit, or a failed held-out check aborts
the build instead of emitting a degraded profile.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from .. import PROFILE_ID
from ..canonicalize import build_tables, canonical_counts, source_row_counts, table_digest
from ..errors import ProfileBuildError, ProfileRejected
from ..fit.profile import AGGR_CLASSES, build_profile_fit
from ..manifest import load_source_pin
from ..model_cards import render_model_card
from ..paths import CONFIG, OUT_DIR, ROOT, TRUST_DIR
from ..validate.heldout import build_report
from . import canonical_bytes, public_key_hex, sign, verify

PUBKEY_PATH = TRUST_DIR / f"{PROFILE_ID}.pub"


def _load_config() -> tuple[dict[str, Any], str]:
    raw = CONFIG.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def _derive_domain(config: dict, fit: dict, tables) -> dict:
    temps = sorted({d.temp for d in tables.ds_ber})
    patterns = sorted({r.pattern for r in tables.rd["rd_hcf"]})
    classes = [c for c in AGGR_CLASSES if any(r.aggr_class == c for r in tables.rd["rd_hcf"])]
    rowpress = any(fam["rowpress"]["supported"] for fam in fit["families"].values())
    policy = config["domain"]
    return {
        "supported_temperatures_celsius": temps,
        "temperature_extrapolation": policy["temperature_extrapolation"],
        "timing": policy["timing"],
        "timing_extrapolation": policy["timing_extrapolation"],
        "aggressor_classes": classes,
        "rowpress": rowpress,
        "data_patterns": patterns,
    }


def _check_counts(tables) -> None:
    canon = canonical_counts(tables)
    raw = source_row_counts()
    if canon != raw:
        mismatched = sorted(k for k in set(canon) | set(raw) if canon.get(k) != raw.get(k))
        raise ProfileBuildError(f"canonical counts do not reproduce source rows: {mismatched}")


def build(write: bool = True) -> dict:
    pin = load_source_pin(verify_data=True)
    config, config_sha = _load_config()

    tables = build_tables()
    _check_counts(tables)
    table_sha = table_digest(tables)

    fit = build_profile_fit(
        tables,
        direction_bias_min_strength=config["fit"]["direction_bias_min_strength"],
    )
    report = build_report(fit, tables, **config["validation"])
    if not report["passed"]:
        raise ProfileBuildError(
            f"held-out validation failed: {report['n_failed']}/{report['n_checks']} checks"
        )

    profile = {
        "schema_version": "1",
        "profile_id": config["profile_id"],
        "standard": config["standard"],
        "labeling": config["labeling"],
        "source": {
            "manifest_id": pin.source_id,
            "url": pin.url,
            "commit": pin.commit,
            "paper": "https://arxiv.org/abs/2503.16749",
            "license": pin.license,
            "retrieved_utc": pin.retrieved_utc,
            "data_sha256": pin.sha256,
            "n_files": len(pin.files),
        },
        "build_basis": {"config_sha256": config_sha, "canonical_table_sha256": table_sha},
        "domain": _derive_domain(config, fit, tables),
        "fit": fit,
        "validation": {
            "gates": report["gates"],
            "holdout_chips": report["holdout_chips"],
            "n_checks": report["n_checks"],
            "n_gated": report["n_gated"],
            "n_failed": report["n_failed"],
            "passed": report["passed"],
        },
    }

    payload = canonical_bytes(profile)
    signature = sign(payload)
    pub = public_key_hex()
    verify(payload, signature, pub)  # fail closed if the freshly signed package is invalid

    sources = {
        "manifest_id": pin.source_id,
        "url": pin.url,
        "commit": pin.commit,
        "license": pin.license,
        "retrieved_utc": pin.retrieved_utc,
        "data_sha256": pin.sha256,
        "files": [{"name": name, "sha256": digest} for name, digest in pin.files],
    }
    card = render_model_card(profile, report)

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        TRUST_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "profile.json").write_bytes(payload)
        (OUT_DIR / "profile.sig").write_text(signature + "\n")
        (OUT_DIR / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        (OUT_DIR / "SOURCES.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
        (OUT_DIR / "model_card.md").write_text(card)
        PUBKEY_PATH.write_text(pub + "\n")

    return {
        "profile_id": profile["profile_id"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signature": signature,
        "public_key": pub,
        "validation_passed": report["passed"],
        "n_checks": report["n_checks"],
        "out_dir": str(OUT_DIR.relative_to(ROOT)),
    }


def read_package() -> tuple[bytes, str, str]:
    payload = (OUT_DIR / "profile.json").read_bytes()
    signature = (OUT_DIR / "profile.sig").read_text().strip()
    pub = PUBKEY_PATH.read_text().strip()
    return payload, signature, pub


def verify_package() -> dict:
    """Verify the committed package against the trust store; raise on tamper."""
    if not (OUT_DIR / "profile.json").is_file():
        raise ProfileRejected("profile package is not built")
    payload, signature, pub = read_package()
    verify(payload, signature, pub)
    return json.loads(payload)


def main() -> int:
    summary = build(write=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
