from unittest.mock import MagicMock, patch

import pytest

from opai_models.client import ModelDownloadError
from opai_models.signatures import SigstoreIdentity, verify_sigstore_bundle


def test_identity_requires_exact_nonempty_values() -> None:
    identity = SigstoreIdentity("https://github.com/onprem-ai/repo/workflow@ref", "https://issuer")
    assert identity.identity.startswith("https://github.com/")
    with pytest.raises(ValueError):
        SigstoreIdentity("", "https://issuer")


def test_verify_bundle_uses_exact_bytes_and_identity_policy() -> None:
    bundle = MagicMock()
    verifier = MagicMock()
    with (
        patch("sigstore.models.Bundle.from_json", return_value=bundle) as parse,
        patch("sigstore.verify.Verifier.production", return_value=verifier) as production,
        patch("sigstore.verify.policy.Identity", return_value="policy") as policy,
    ):
        verify_sigstore_bundle(
            b"exact-checksum-bytes\n",
            b'{"mediaType":"application/vnd.dev.sigstore.bundle.v0.3+json"}',
            SigstoreIdentity("identity", "https://issuer"),
            offline=True,
        )
    parse.assert_called_once()
    production.assert_called_once_with(offline=True)
    policy.assert_called_once_with(identity="identity", issuer="https://issuer")
    verifier.verify_artifact.assert_called_once_with(b"exact-checksum-bytes\n", bundle, "policy")


def test_verify_bundle_exposes_only_safe_error() -> None:
    with (
        patch("sigstore.models.Bundle.from_json", side_effect=ValueError("secret details")),
        pytest.raises(ModelDownloadError, match="model signature verification failed") as caught,
    ):
        verify_sigstore_bundle(b"checksums", b"invalid", SigstoreIdentity("identity", "issuer"))
    assert "secret details" not in str(caught.value)
