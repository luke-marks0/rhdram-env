"""Stable error codes for the profile pipeline.

These mirror the environment-wide stable error code set so that failures surface
the same code whether they originate in the builder or in a consumer that loads a
packaged profile.
"""

from __future__ import annotations


class ProfileError(Exception):
    """Base class carrying a stable error code."""

    code = "INTERNAL_SIMULATOR_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class SourceUnavailable(ProfileError):
    """Required source/capability is absent; fail closed, do not substitute."""

    code = "UNAVAILABLE_CAPABILITY"


class ProfileRejected(ProfileError):
    """Profile failed signature or integrity verification."""

    code = "PROFILE_REJECTED"


class ProfileBuildError(ProfileError):
    """A build/validation gate did not pass."""

    code = "INTERNAL_SIMULATOR_ERROR"
