"""Sigstore verification for model checksum manifests."""

from __future__ import annotations

from dataclasses import dataclass

from opai_models.client import ModelDownloadError
from opai_models.errors import sanitize_error_detail


@dataclass(frozen=True)
class SigstoreIdentity:
    """Exact certificate identity and OIDC issuer trusted for model signatures."""

    identity: str
    issuer: str

    def __post_init__(self) -> None:
        if not self.identity.strip() or not self.issuer.strip():
            raise ValueError("Sigstore identity and issuer must not be empty")


def verify_sigstore_bundle(
    artifact: bytes,
    bundle_data: bytes,
    trusted_identity: SigstoreIdentity,
    *,
    offline: bool = False,
) -> None:
    """Verify a Sigstore bundle over the exact artifact bytes."""
    try:
        from sigstore.models import Bundle
        from sigstore.verify import Verifier
        from sigstore.verify.policy import Identity

        bundle = Bundle.from_json(bundle_data)
        policy = Identity(
            identity=trusted_identity.identity,
            issuer=trusted_identity.issuer,
        )
        Verifier.production(offline=offline).verify_artifact(artifact, bundle, policy)
    except Exception as exc:
        detail = sanitize_error_detail(exc)
        raise ModelDownloadError(f"model signature verification failed: {detail}") from None
