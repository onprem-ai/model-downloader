from unittest.mock import MagicMock, patch

import pytest

from opai_models.async_client import AsyncModelClient
from opai_models.client import ModelAccess, ModelDownloadError


def test_signature_verification_requires_complete_trust_policy() -> None:
    with pytest.raises(ValueError, match="sigstore_identity and sigstore_issuer"):
        AsyncModelClient("https://license.example", lambda: "license")
    with pytest.raises(ValueError, match="sigstore_identity and sigstore_issuer"):
        AsyncModelClient(
            "https://license.example",
            lambda: "license",
            sigstore_identity="identity",
        )


@pytest.mark.asyncio
async def test_async_client_reuses_configuration_and_fetches_credentials_per_operation() -> None:
    credentials = MagicMock(return_value="license")
    client = AsyncModelClient("https://license.example", credentials, verify_signatures=False)
    listing = {"models": ["a"], "next_cursor": None}
    access = ModelAccess("file", "https://s3/a", 1, "source", "later", {}, {})
    with (
        patch("opai_models.async_client.LicenseClient.list_models", return_value=listing),
        patch("opai_models.async_client.LicenseClient.access", return_value=access),
    ):
        async with client as entered:
            assert entered is client
            assert await client.list_models() == listing
            assert await client.get_model_file("a", "file") == access
    assert credentials.call_count == 2


@pytest.mark.asyncio
async def test_async_source_and_snapshot_delegate_without_persisting_credentials() -> None:
    client = AsyncModelClient("https://license.example", lambda: "license", verify_signatures=False)
    source = (
        b'{"schema_version":1,"source":{"provider":"huggingface",'
        b'"repository":"a/b","revision":null}}'
    )
    with (
        patch("opai_models.async_client.LicenseClient.read_small", return_value=source),
        patch("opai_models.async_client.snapshot_model", return_value="snapshot") as snapshot,
    ):
        assert (await client.get_source("a")).source.repository == "a/b"
        assert await client.snapshot_model("a") == "snapshot"
    assert snapshot.call_count == 1
    assert snapshot.call_args.args[1] == "a"
    assert snapshot.call_args.kwargs == {
        "verify_checksums": True,
        "verify_signatures": False,
        "trusted_identity": None,
        "sigstore_offline": False,
    }


@pytest.mark.asyncio
async def test_async_source_rejects_invalid_json() -> None:
    client = AsyncModelClient("https://license.example", lambda: "license", verify_signatures=False)
    with patch("opai_models.async_client.LicenseClient.read_small", return_value=b"secret-invalid"):
        with pytest.raises(ModelDownloadError, match="invalid .source.json") as caught:
            await client.get_source("a")
    assert "secret-invalid" not in str(caught.value)
