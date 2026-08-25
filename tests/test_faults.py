import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import secret_key

from opai_models.cli import _human_size, _license_key, _progress, main
from opai_models.client import ModelAccess, ModelDownloadError, _AsyncLicenseTransport
from opai_models.download import (
    AccessProvider,
    PermanentDownloadError,
    RetryableDownloadError,
    _backoff_delay,
    _download_range,
    _expected_sha256,
    _permanent_os_error,
    _retry_after,
)


def model_access(
    *,
    url: str = "https://s3.example/model?secret=query",
    source_id: str = "source",
) -> ModelAccess:
    return ModelAccess(
        path="test.bin",
        url=url,
        size=3,
        source_id=source_id,
        expires_at="2099-01-01T00:00:00Z",
        checksums={},
        required_headers={"If-Match": "etag"},
    )


def async_client(handler, key=secret_key) -> _AsyncLicenseTransport:
    return _AsyncLicenseTransport(
        "https://example.com",
        key,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_client_rejects_unsafe_urls_and_provider() -> None:
    for url in (
        "file:///tmp/api",
        "http://example.com",
        "https://user:password@example.com",
        "missing-scheme.example",
    ):
        with pytest.raises(ModelDownloadError, match="API URL"):
            _AsyncLicenseTransport(url, secret_key)
    with pytest.raises(TypeError, match="callable"):
        _AsyncLicenseTransport("https://example.com", "secret")  # type: ignore[arg-type]
    _AsyncLicenseTransport("http://127.0.0.1:8000", secret_key)


@pytest.mark.asyncio
async def test_client_rejects_empty_license_and_unsafe_model_paths() -> None:
    async def empty_key() -> str:
        return ""

    client = async_client(lambda request: httpx.Response(500), empty_key)
    with pytest.raises(ModelDownloadError, match="must not be empty"):
        await client.list_models()
    for path in ("../secret", "a//file", "a\\file", "file\x01"):
        with pytest.raises(ModelDownloadError, match="invalid"):
            await client.access("example", path)
    with pytest.raises(ModelDownloadError, match="model ID"):
        await client.access("namespace/example", "file")
    await client.http.aclose()


@pytest.mark.asyncio
async def test_access_provider_refresh_and_change_detection() -> None:
    client = MagicMock()
    client.access = AsyncMock(
        side_effect=[model_access(source_id="one"), model_access(source_id="one")]
    )
    provider = AccessProvider(client, "example", "test.bin")
    assert (await provider.get()).source_id == "one"
    assert (await provider.get()).source_id == "one"
    assert (await provider.get(refresh=True)).source_id == "one"
    assert provider.refreshes == 2

    client.access = AsyncMock(
        side_effect=[model_access(source_id="one"), model_access(source_id="two")]
    )
    provider = AccessProvider(client, "example", "test.bin")
    await provider.get()
    with pytest.raises(ModelDownloadError, match="changed"):
        await provider.get(refresh=True)

    changed_size = ModelAccess(**{**model_access().__dict__, "size": 4})
    client.access = AsyncMock(side_effect=[model_access(), changed_size])
    provider = AccessProvider(client, "example", "test.bin")
    await provider.get()
    with pytest.raises(ModelDownloadError, match="changed"):
        await provider.get(refresh=True)


def download_client(responses, access: ModelAccess | None = None):
    sequence = iter(responses)

    def handler(request):
        value = next(sequence)
        if isinstance(value, Exception):
            raise value
        return value

    client = MagicMock()
    client.access = AsyncMock(return_value=access or model_access())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def range_response(
    body: bytes,
    *,
    status: int = 206,
    content_range: str = "bytes 0-2/3",
    content_length: str | None = "3",
) -> httpx.Response:
    headers = {"Content-Range": content_range}
    if content_length is not None:
        headers["Content-Length"] = content_length
    return httpx.Response(status, headers=headers, content=body)


@pytest.mark.asyncio
async def test_download_range_success_and_request_headers(tmp_path: Path) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return range_response(b"abc")

    client = MagicMock()
    client.access = AsyncMock(return_value=model_access())
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert (
            await _download_range(
                AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
            )
            == 3
        )
    finally:
        os.close(descriptor)
        await client.http.aclose()
    assert requests[0].headers["Range"] == "bytes=0-2"
    assert requests[0].headers["If-Match"] == "etag"
    assert (tmp_path / "partial").read_bytes() == b"abc"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (range_response(b"abc", status=200), "HTTP 200"),
        (range_response(b"abc", content_range="bytes 1-2/3"), "Content-Range"),
        (range_response(b"abc", content_length="2"), "Content-Length"),
        (range_response(b"ab"), "truncated"),
        (range_response(b"abcd"), "exceeded"),
    ],
)
@pytest.mark.asyncio
async def test_download_range_validates_response(tmp_path: Path, response, message: str) -> None:
    client = download_client([response])
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(ModelDownloadError, match=message):
            await _download_range(
                AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
            )
    finally:
        os.close(descriptor)
        await client.http.aclose()


@pytest.mark.parametrize(
    ("status", "message"), [(412, "changed"), (416, "byte range"), (404, "HTTP 404")]
)
@pytest.mark.asyncio
async def test_download_range_maps_terminal_http_errors(
    tmp_path: Path, status: int, message: str
) -> None:
    client = download_client([httpx.Response(status)])
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(ModelDownloadError, match=message):
            await _download_range(
                AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
            )
    finally:
        os.close(descriptor)
        await client.http.aclose()


def test_retry_after_backoff_and_permanent_disk_errors(monkeypatch) -> None:
    monkeypatch.setattr("opai_models.download.time.time", lambda: 0)
    monkeypatch.setattr("opai_models.download.secrets.randbelow", lambda maximum: maximum - 1)
    assert _retry_after("12") == 12
    assert _retry_after("Thu, 01 Jan 1970 00:00:20 GMT") == 20
    assert _retry_after("invalid") is None
    assert _backoff_delay(2, 0.5, 60, None) < 2.001
    assert _backoff_delay(2, 0.5, 60, 10) == 10
    assert _permanent_os_error(OSError(28, "full"))
    assert not _permanent_os_error(OSError(104, "reset"))


@pytest.mark.asyncio
async def test_download_range_records_retry_after_and_retries(tmp_path: Path, monkeypatch) -> None:
    client = download_client(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            range_response(b"abc"),
        ]
    )
    errors = []

    async def record_error(event) -> None:
        errors.append(event)

    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    monkeypatch.setattr("opai_models.download.secrets.randbelow", lambda maximum: 0)
    sleep = AsyncMock()
    monkeypatch.setattr("opai_models.download.asyncio.sleep", sleep)
    try:
        assert (
            await _download_range(
                AccessProvider(client, "example", "test.bin"),
                descriptor,
                0,
                2,
                1,
                1,
                chunk_index=4,
                on_error=record_error,
            )
            == 3
        )
    finally:
        os.close(descriptor)
        await client.http.aclose()
    assert errors[0]["http_status"] == 429
    assert errors[0]["backoff_seconds"] == 7
    sleep.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_download_range_treats_disk_full_as_permanent(tmp_path: Path) -> None:
    client = download_client([OSError(28, "disk full")])
    # MockTransport surfaces handler OSError directly, exercising disk/network classification.
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    errors = []

    async def record_error(event) -> None:
        errors.append(event)

    try:
        with pytest.raises(PermanentDownloadError):
            await _download_range(
                AccessProvider(client, "example", "test.bin"),
                descriptor,
                0,
                2,
                8,
                1,
                chunk_index=2,
                on_error=record_error,
            )
    finally:
        os.close(descriptor)
        await client.http.aclose()
    assert errors[0]["retryable"] is False
    assert errors[0]["message"] == "OSError: [Errno 28] disk full"


@pytest.mark.asyncio
async def test_download_range_preserves_network_error_detail(tmp_path: Path) -> None:
    client = download_client([httpx.ConnectError("DNS lookup failed for storage.internal")])
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    errors = []

    async def record_error(event) -> None:
        errors.append(event)

    try:
        with pytest.raises(RetryableDownloadError) as caught:
            await _download_range(
                AccessProvider(client, "example", "test.bin"),
                descriptor,
                0,
                2,
                0,
                1,
                on_error=record_error,
            )
    finally:
        os.close(descriptor)
        await client.http.aclose()
    assert "DNS lookup failed for storage.internal" in str(caught.value)
    assert errors[0]["message"] == str(caught.value)


@pytest.mark.asyncio
async def test_download_range_redacts_signed_url_from_network_error(tmp_path: Path) -> None:
    client = download_client(
        [httpx.ConnectError("request failed at https://storage.example/file?token=secret")]
    )
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(RetryableDownloadError) as caught:
            await _download_range(
                AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
            )
    finally:
        os.close(descriptor)
        await client.http.aclose()
    message = str(caught.value)
    assert "secret" not in message
    assert "[URL REDACTED]" in message


@pytest.mark.asyncio
async def test_download_range_refreshes_after_403_then_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    client = download_client([httpx.Response(403), range_response(b"abc")])
    provider = AccessProvider(client, "example", "test.bin")
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    monkeypatch.setattr("opai_models.download.asyncio.sleep", AsyncMock())
    monkeypatch.setattr("opai_models.download.secrets.randbelow", lambda maximum: 0)
    try:
        assert await _download_range(provider, descriptor, 0, 2, 1, 1) == 3
    finally:
        os.close(descriptor)
        await client.http.aclose()
    assert provider.refreshes >= 2


@pytest.mark.asyncio
async def test_download_range_rejects_unsafe_signed_url(tmp_path: Path) -> None:
    client = download_client([], model_access(url="file:///tmp/model"))
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(ModelDownloadError, match="HTTPS"):
            await _download_range(
                AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
            )
    finally:
        os.close(descriptor)
        await client.http.aclose()


def test_sha256_missing_and_wrong_length_base64() -> None:
    assert _expected_sha256({}) is None
    assert _expected_sha256({"sha256": "YQ=="}) is None


def test_cli_human_output_progress_prompt_and_json_error(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    listing = {"models": ["a", "sub"], "next_cursor": None}
    with patch(
        "opai_models.cli._AsyncLicenseTransport.list_models", AsyncMock(return_value=listing)
    ):
        assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert output.splitlines() == ["a", "sub"]

    monkeypatch.delenv("OPAI_LICENSE_KEY")
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("getpass.getpass", return_value="prompted"),
    ):
        assert _license_key("OPAI_LICENSE_KEY", prompt=True) == "prompted"

    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    with patch(
        "opai_models.cli._AsyncLicenseTransport.list_models",
        AsyncMock(side_effect=ModelDownloadError("safe")),
    ):
        assert main(["--json", "list"]) == 1
    assert json.loads(capsys.readouterr().err) == {"event": "error", "error": "safe"}

    __import__("asyncio").run(
        _progress(
            {
                "event": "chunk_complete",
                "completed_bytes": 512,
                "total_bytes": 1024,
                "bytes_per_second": 256,
            }
        )
    )
    __import__("asyncio").run(
        _progress({"event": "complete", "path": "a", "destination": "output/a"})
    )
    stderr = capsys.readouterr().err
    assert "50.00%" in stderr and "Downloaded a" in stderr
    assert _human_size(2 * 1024**4) == "2.00 TiB"
