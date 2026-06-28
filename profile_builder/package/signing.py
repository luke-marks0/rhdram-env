"""Ed25519 signing and verification for profile packages.

The signing key is derived deterministically from a committed development trust-root
seed so the self-contained build reproduces a byte-identical signature. The verifier
only ever needs the committed public key from the trust store; tampering with the
signed bytes raises ``PROFILE_REJECTED``.

This development trust root binds integrity and provenance within this repository. A
production deployment must replace the seed with an externally managed (HSM/KMS) key,
re-sign, and publish the new public key in the trust store.
"""

from __future__ import annotations

import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..errors import ProfileRejected
from ..paths import SIGNING_SEED


def _seed_bytes() -> bytes:
    if not SIGNING_SEED.is_file():
        raise ProfileRejected(f"missing signing seed: {SIGNING_SEED}")
    # Only the first non-empty line is the canonical seed phrase, so trailing
    # newline edits do not change the derived key.
    for line in SIGNING_SEED.read_text().splitlines():
        if line.strip():
            return line.strip().encode("utf-8")
    raise ProfileRejected("signing seed is empty")


def _private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(_seed_bytes()).digest())


def public_key_hex() -> str:
    return _private_key().public_key().public_bytes_raw().hex()


def sign(data: bytes) -> str:
    return _private_key().sign(data).hex()


def verify(data: bytes, signature_hex: str, public_key_hex: str) -> None:
    """Verify a detached signature; raise ProfileRejected on any failure."""
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), data)
    except (InvalidSignature, ValueError) as exc:
        raise ProfileRejected(f"signature verification failed: {exc}") from exc
