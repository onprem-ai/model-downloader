from unittest.mock import AsyncMock

import httpx
import pytest

from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError


def metadata_access(size: int, url: str = "https://s3.example/meta") -> ModelAccess:
    return ModelAccess(".source.json", url, size, "source", "later", {}, {})


def client(handler) -> LicenseClient:
    return LicenseClient(
        "https://license.example",
        lambda: "secret",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_read_small_fetches_bounded_metadata(monkeypatch) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=b"abc")

    instance = client(handler)
    access = AsyncMock(return_value=metadata_access(3))
    monkeypatch.setattr(instance, "access", access)
    assert await instance.read_small("a", ".source.json", maximum=3) == b"abc"
    access.assert_awaited_once_with("a", ".source.json")
    assert str(requests[0].url) == "https://s3.example/meta"
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_read_small_rejects_large_invalid_url_and_wrong_size(monkeypatch) -> None:
    instance = client(lambda request: httpx.Response(200, content=b"ab"))
    monkeypatch.setattr(instance, "access", AsyncMock(return_value=metadata_access(4)))
    with pytest.raises(ModelDownloadError, match="too large"):
        await instance.read_small("a", ".source.json", maximum=3)
    monkeypatch.setattr(
        instance, "access", AsyncMock(return_value=metadata_access(1, "file:///secret"))
    )
    with pytest.raises(ModelDownloadError, match="HTTPS"):
        await instance.read_small("a", ".source.json")
    monkeypatch.setattr(instance, "access", AsyncMock(return_value=metadata_access(3)))
    with pytest.raises(ModelDownloadError, match="size"):
        await instance.read_small("a", ".source.json")
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_read_small_sanitizes_network_errors(monkeypatch) -> None:
    def handler(request):
        raise httpx.ConnectError("token=secret", request=request)

    instance = client(handler)
    access = metadata_access(3, "https://s3.example/meta?token=secret")
    monkeypatch.setattr(instance, "access", AsyncMock(return_value=access))
    with pytest.raises(ModelDownloadError, match="cannot read model metadata") as caught:
        await instance.read_small("a", ".source.json")
    assert "token=secret" not in str(caught.value)
    await instance.http.aclose()
