import httpx
import pytest
from conftest import secret_key

from opai_models.client import (
    ModelDownloadError,
    TransientModelDownloadError,
    _AsyncLicenseTransport,
)


def client(handler, key=secret_key) -> _AsyncLicenseTransport:
    return _AsyncLicenseTransport(
        "https://license.example",
        key,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_restricts_model_dir_names_and_relative_paths() -> None:
    instance = client(lambda request: httpx.Response(500))
    for model_dir_name in ("", "namespace/example", "a/b", "..", "bad\n"):
        with pytest.raises(ModelDownloadError, match="model directory name"):
            await instance.access(model_dir_name, "file")
    for path in ("../secret", "a//file", "a\\file", "file\x01", "folder/"):
        with pytest.raises(ModelDownloadError, match="model"):
            await instance.access("example", path)
    await instance.http.aclose()


def test_relative_directory_prefix_validation() -> None:
    assert _AsyncLicenseTransport._relative_path("nested/", prefix=True) == "nested/"
    assert _AsyncLicenseTransport._relative_path("nested", prefix=True) == "nested/"
    for value in ("", "/nested/", "../nested/", "nested//child/"):
        with pytest.raises(ModelDownloadError, match="model-relative path"):
            _AsyncLicenseTransport._relative_path(value, prefix=True)


@pytest.mark.asyncio
async def test_access_parses_relative_metadata_without_logging_url() -> None:
    body = {
        "path": "a.bin",
        "url": "https://s3.example/a?signature=secret",
        "size": 12,
        "source_id": "a" * 64,
        "expires_at": "2099-01-01T00:00:00Z",
        "checksums": {"sha256": "b" * 64},
        "required_headers": {"If-Match": "etag"},
    }
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    async def credentials() -> str:
        return "license-secret"

    instance = client(handler, credentials)
    result = await instance.access("example", "a.bin")
    assert result.path == "a.bin"
    assert result.size == 12
    assert str(requests[0].url) == "https://license.example/v1/models/example/access/a.bin"
    assert requests[0].headers["Authorization"] == "Bearer license-secret"
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_list_models_follows_cursor_and_validates_ids() -> None:
    pages = iter(
        [
            {"models": ["b", "a"], "next_cursor": "cursor"},
            {"models": ["a", "c"], "next_cursor": None},
        ]
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=next(pages))

    instance = client(handler)
    result = await instance.list_models(limit=2)
    assert result == {"models": ["a", "b", "c"], "next_cursor": None}
    assert requests[1].url.params["cursor"] == "cursor"
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_list_models_rejects_bad_results_and_repeated_cursor() -> None:
    for body, message in (
        ({"models": {}, "next_cursor": None}, "invalid model listing"),
        ({"models": [], "next_cursor": "same"}, "repeated"),
        ({"models": ["namespace/model"], "next_cursor": None}, "model directory name"),
    ):
        instance = client(lambda request, body=body: httpx.Response(200, json=body))
        with pytest.raises(ModelDownloadError, match=message):
            await instance.list_models()
        await instance.http.aclose()


@pytest.mark.asyncio
async def test_list_page_uses_model_dir_name_and_relative_prefix() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={})

    instance = client(handler)
    await instance.list_page("example model", "nested files/", limit=5, cursor="a+b=")
    request = requests[0]
    assert request.url.path == "/v1/models/example model/files"
    assert request.url.params["prefix"] == "nested files/"
    assert request.url.params["cursor"] == "a+b="
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_list_all_follows_cursor_and_deduplicates_prefixes(monkeypatch) -> None:
    instance = client(lambda request: httpx.Response(500))
    pages = [
        {"objects": [{"key": "a", "size": 1}], "prefixes": ["sub/"], "next_cursor": "x"},
        {"objects": [{"key": "b", "size": 2}], "prefixes": ["sub/"], "next_cursor": None},
    ]
    from unittest.mock import AsyncMock

    mocked = AsyncMock(side_effect=pages)
    monkeypatch.setattr(instance, "list_page", mocked)
    result = await instance.list_all("example")
    assert [item["key"] for item in result["objects"]] == ["a", "b"]
    assert result["prefixes"] == ["sub/"]
    assert result["model_dir_name"] == "example"
    assert mocked.await_count == 2
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_list_normalizes_null_collections_and_rejects_bad_pages(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    instance = client(lambda request: httpx.Response(500))
    monkeypatch.setattr(
        instance,
        "list_page",
        AsyncMock(return_value={"objects": None, "prefixes": None, "next_cursor": None}),
    )
    assert await instance.list_all("example") == {
        "model_dir_name": "example",
        "prefix": "",
        "objects": [],
        "prefixes": [],
        "next_cursor": None,
    }
    monkeypatch.setattr(
        instance,
        "list_page",
        AsyncMock(return_value={"objects": [], "prefixes": [], "next_cursor": "same"}),
    )
    with pytest.raises(ModelDownloadError, match="repeated"):
        await instance.list_all("example")
    monkeypatch.setattr(
        instance,
        "list_page",
        AsyncMock(return_value={"objects": {}, "prefixes": []}),
    )
    with pytest.raises(ModelDownloadError, match="invalid listing"):
        await instance.list_all("example")
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_access_rejects_wrong_path_negative_size_and_bad_maps() -> None:
    base = {
        "path": "a",
        "url": "https://s3.example/a",
        "size": 1,
        "source_id": "source",
        "expires_at": "later",
        "checksums": {},
        "required_headers": {},
    }
    for change in (
        {"path": "b"},
        {"size": -1},
        {"size": True},
        {"checksums": {"sha256": 1}},
        {"required_headers": []},
    ):
        instance = client(lambda request, body={**base, **change}: httpx.Response(200, json=body))
        with pytest.raises(ModelDownloadError, match="invalid model metadata"):
            await instance.access("example", "a")
        await instance.http.aclose()


@pytest.mark.asyncio
async def test_failures_are_classified_with_sanitized_details() -> None:
    for status, error_type in ((503, TransientModelDownloadError), (403, ModelDownloadError)):
        instance = client(
            lambda request, status=status: httpx.Response(
                status,
                json={
                    "detail": (
                        "model is not licensed; inspect "
                        "https://license.example/account?token=secret-value"
                    )
                },
            )
        )
        with pytest.raises(error_type, match=f"HTTP {status}") as caught:
            await instance.access("example", "a")
        message = str(caught.value)
        assert "model is not licensed" in message
        assert "license-secret" not in message
        assert "secret-value" not in message
        assert "[URL REDACTED]" in message
        await instance.http.aclose()

    def offline(request):
        raise httpx.ConnectError("DNS lookup failed for license.internal", request=request)

    instance = client(offline)
    with pytest.raises(TransientModelDownloadError, match="ConnectError") as caught:
        await instance.access("example", "a")
    assert "DNS lookup failed for license.internal" in str(caught.value)
    await instance.http.aclose()


@pytest.mark.asyncio
async def test_invalid_json_and_response_shape() -> None:
    for response, message in (
        (httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (httpx.Response(200, json=[]), "invalid response"),
    ):
        instance = client(lambda request, response=response: response)
        with pytest.raises(ModelDownloadError, match=message):
            await instance.access("example", "a")
        await instance.http.aclose()


@pytest.mark.asyncio
async def test_supports_async_license_provider() -> None:
    async def key() -> str:
        return "async-secret"

    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"models": [], "next_cursor": None})

    instance = client(handler, key)
    await instance.list_models()
    assert seen == ["Bearer async-secret"]
    await instance.http.aclose()
