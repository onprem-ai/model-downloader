from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": ["a"], "next_cursor": None})
        return httpx.Response(
            200,
            json={
                "path": "file",
                "url": "https://s3.example/a",
                "size": 1,
                "source_id": "source",
                "expires_at": "later",
                "checksums": {},
                "required_headers": {},
            },
        )

    client = AsyncModelClient("https://license.example", credentials, verify_signatures=False)
    await client._client.http.aclose()
    client._client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client as entered:
        assert entered is client
        assert await client.list_models() == {"models": ["a"], "next_cursor": None}
        assert await client.get_model_file("a", "file") == ModelAccess(
            "file", "https://s3.example/a", 1, "source", "later", {}, {}
        )
    assert credentials.call_count == 2
    assert all(request.headers["Authorization"] == "Bearer license" for request in requests)


@pytest.mark.asyncio
async def test_async_source_and_snapshot_delegate_without_persisting_credentials() -> None:
    client = AsyncModelClient("https://license.example", lambda: "license", verify_signatures=False)
    source = (
        b'{"schema_version":1,"source":{"provider":"huggingface",'
        b'"repository":"a/b","revision":null}}'
    )
    with (
        patch("opai_models.async_client.LicenseClient.read_small", AsyncMock(return_value=source)),
        patch(
            "opai_models.async_client.snapshot_model", AsyncMock(return_value="snapshot")
        ) as snapshot,
    ):
        assert (await client.get_source("a")).source.repository == "a/b"
        assert await client.snapshot_model("a") == "snapshot"
    assert snapshot.await_count == 1
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
    with patch(
        "opai_models.async_client.LicenseClient.read_small",
        AsyncMock(return_value=b"secret-invalid"),
    ):
        with pytest.raises(ModelDownloadError, match="invalid .source.json") as caught:
            await client.get_source("a")
    assert "secret-invalid" not in str(caught.value)
