from __future__ import annotations

from typing import Any

from profile_builder.package.build import verify_package


ADMITTED_PROFILES = {"ddr4_vts25_v1"}
DEFERRED_PROFILES = {"hbm2_read_disturbance", "hbm2_read_disturbance_v1"}


def load_profile(profile_id: str) -> dict[str, Any]:
    if profile_id not in ADMITTED_PROFILES:
        raise ValueError(f"UNAVAILABLE_CAPABILITY:{profile_id}")
    profile = verify_package()
    if profile.get("profile_id") != profile_id:
        raise ValueError("PROFILE_REJECTED:profile id mismatch")
    return profile

