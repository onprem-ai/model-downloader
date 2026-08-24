import io
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opai_models.cli import _human_size, _license_key, _progress, main
from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError
from opai_models.download import (
    AccessProvider,
    PermanentDownloadError,
    _backoff_delay,
    _download_range,
    _expected_sha256,
    _permanent_os_error,
    _retry_after,
)


class RangeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 206,
        content_range: str = "bytes 0-2/3",
        content_length: str | None = "3",
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


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


def test_client_rejects_unsafe_urls_and_empty_license() -> None:
    for url in (
        "file:///tmp/api",
        "http://example.com",
        "https://user:password@example.com",
        "missing-scheme.example",
    ):
        with pytest.raises(ModelDownloadError, match="API URL"):
            LicenseClient(url, "secret")
    with pytest.raises(ModelDownloadError, match="must not be empty"):
        LicenseClient("https://example.com", "")
    LicenseClient("http://127.0.0.1:8000", "secret")


def test_client_rejects_unsafe_model_paths() -> None:
    client = LicenseClient("https://example.com", "secret")
    for path in ("../secret", "a//file", "a\\file", "file\x01"):
        with pytest.raises(ModelDownloadError, match="invalid"):
            client.access("example", path)
    with pytest.raises(ModelDownloadError, match="model ID"):
        client.access("namespace/example", "file")


def test_client_maps_bad_json_and_invalid_metadata() -> None:
    client = LicenseClient("https://example.com", "secret")
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"[]")):
        with pytest.raises(ModelDownloadError, match="invalid response"):
            client.access("example", "a")
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"not-json")):
        with pytest.raises(ModelDownloadError, match="invalid JSON"):
            client.access("example", "a")
    with patch.object(client, "_json", return_value={"path": "wrong"}):
        with pytest.raises(ModelDownloadError, match="invalid model metadata"):
            client.access("example", "a")


def test_http_error_with_invalid_body_is_generic() -> None:
    client = LicenseClient("https://example.com", "secret")
    error = urllib.error.HTTPError("https://example.com", 500, "error", {}, io.BytesIO(b"not-json"))
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ModelDownloadError, match="HTTP 500"):
            client.access("example", "a")


def test_list_page_encodes_cursor() -> None:
    client = LicenseClient("https://example.com", "secret")
    with patch.object(client, "_json", return_value={}) as request:
        client.list_page("example", "a folder/", limit=5, cursor="a+b=")
    path = request.call_args.args[0]
    assert "/v1/models/example/files?" in path
    assert "prefix=a+folder%2F" in path
    assert "limit=5" in path
    assert "cursor=a%2Bb%3D" in path


def test_access_provider_refresh_and_change_detection() -> None:
    client = MagicMock()
    client.access.side_effect = [model_access(source_id="one"), model_access(source_id="one")]
    provider = AccessProvider(client, "example", "test.bin")
    assert provider.get().source_id == "one"
    assert provider.get().source_id == "one"
    assert provider.get(refresh=True).source_id == "one"
    assert provider.refreshes == 2

    client.access.side_effect = [model_access(source_id="one"), model_access(source_id="two")]
    provider = AccessProvider(client, "example", "test.bin")
    provider.get()
    with pytest.raises(ModelDownloadError, match="changed"):
        provider.get(refresh=True)

    changed_size = ModelAccess(**{**model_access().__dict__, "size": 4})
    client.access.side_effect = [model_access(), changed_size]
    provider = AccessProvider(client, "example", "test.bin")
    provider.get()
    with pytest.raises(ModelDownloadError, match="changed"):
        provider.get(refresh=True)


def test_download_range_success_and_request_headers(tmp_path: Path) -> None:
    client = MagicMock()
    client.access.return_value = model_access()
    provider = AccessProvider(client, "example", "test.bin")
    destination = tmp_path / "partial"
    descriptor = os.open(destination, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with patch("urllib.request.urlopen", return_value=RangeResponse(b"abc")) as urlopen:
            assert _download_range(provider, descriptor, 0, 2, 0, 1) == 3
        request = urlopen.call_args.args[0]
        assert request.headers["Range"] == "bytes=0-2"
        assert request.headers["If-match"] == "etag"
    finally:
        os.close(descriptor)
    assert destination.read_bytes() == b"abc"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (RangeResponse(b"abc", status=200), "expected 206"),
        (RangeResponse(b"abc", content_range="bytes 1-2/3"), "Content-Range"),
        (RangeResponse(b"abc", content_length="2"), "Content-Length"),
        (RangeResponse(b"ab"), "truncated"),
        (RangeResponse(b"abcd"), "exceeded"),
    ],
)
def test_download_range_validates_response(
    tmp_path: Path, response: RangeResponse, message: str
) -> None:
    client = MagicMock()
    client.access.return_value = model_access()
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with patch("urllib.request.urlopen", return_value=response):
            with pytest.raises(ModelDownloadError, match=message):
                _download_range(
                    AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
                )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("status", "message"),
    [(412, "changed"), (416, "byte range"), (404, "HTTP 404")],
)
def test_download_range_maps_terminal_http_errors(
    tmp_path: Path, status: int, message: str
) -> None:
    client = MagicMock()
    client.access.return_value = model_access()
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    error = urllib.error.HTTPError("https://s3.example", status, "error", {}, None)
    try:
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(ModelDownloadError, match=message):
                _download_range(
                    AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1
                )
    finally:
        os.close(descriptor)


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


def test_download_range_records_retry_after_and_retries(tmp_path: Path) -> None:
    client = MagicMock()
    client.access.return_value = model_access()
    provider = AccessProvider(client, "example", "test.bin")
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    error = urllib.error.HTTPError("https://s3.example", 429, "busy", {"Retry-After": "7"}, None)
    errors = []
    try:
        with (
            patch("urllib.request.urlopen", side_effect=[error, RangeResponse(b"abc")]),
            patch("time.sleep") as sleep,
            patch("secrets.randbelow", return_value=0),
        ):
            assert (
                _download_range(
                    provider,
                    descriptor,
                    0,
                    2,
                    1,
                    1,
                    chunk_index=4,
                    on_error=errors.append,
                )
                == 3
            )
    finally:
        os.close(descriptor)
    assert errors == [
        {
            "chunk": 4,
            "attempt": 1,
            "error_type": "RetryableDownloadError",
            "message": "storage temporarily returned HTTP 429",
            "retryable": True,
            "http_status": 429,
            "backoff_seconds": 7,
        }
    ]
    sleep.assert_called_once_with(7)


def test_download_range_treats_disk_full_as_permanent(tmp_path: Path) -> None:
    client = MagicMock()
    client.access.return_value = model_access()
    provider = AccessProvider(client, "example", "test.bin")
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    errors = []
    try:
        with patch("urllib.request.urlopen", side_effect=OSError(28, "disk full")):
            with pytest.raises(PermanentDownloadError):
                _download_range(
                    provider,
                    descriptor,
                    0,
                    2,
                    8,
                    1,
                    chunk_index=2,
                    on_error=errors.append,
                )
    finally:
        os.close(descriptor)
    assert errors[0]["chunk"] == 2
    assert errors[0]["retryable"] is False


def test_download_range_refreshes_after_403_then_succeeds(tmp_path: Path) -> None:
    client = MagicMock()
    client.access.return_value = model_access()
    provider = AccessProvider(client, "example", "test.bin")
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    error = urllib.error.HTTPError("https://s3.example", 403, "expired", {}, None)
    try:
        with (
            patch("urllib.request.urlopen", side_effect=[error, RangeResponse(b"abc")]),
            patch("time.sleep"),
            patch("secrets.randbelow", return_value=0),
        ):
            assert _download_range(provider, descriptor, 0, 2, 1, 1) == 3
    finally:
        os.close(descriptor)
    assert provider.refreshes >= 2


def test_download_range_rejects_unsafe_signed_url(tmp_path: Path) -> None:
    client = MagicMock()
    client.access.return_value = model_access(url="file:///tmp/model")
    descriptor = os.open(tmp_path / "partial", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(ModelDownloadError, match="HTTPS"):
            _download_range(AccessProvider(client, "example", "test.bin"), descriptor, 0, 2, 0, 1)
    finally:
        os.close(descriptor)


def test_sha256_missing_and_wrong_length_base64() -> None:
    assert _expected_sha256({}) is None
    assert _expected_sha256({"sha256": "YQ=="}) is None


def test_cli_human_output_progress_prompt_and_json_error(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    listing = {"models": ["a", "sub"], "next_cursor": None}
    with patch("opai_models.cli.LicenseClient.list_models", return_value=listing):
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
    with patch("opai_models.cli.LicenseClient.list_models", side_effect=ModelDownloadError("safe")):
        assert main(["--json", "list"]) == 1
    assert json.loads(capsys.readouterr().err) == {"event": "error", "error": "safe"}

    _progress(
        {
            "event": "chunk_complete",
            "completed_bytes": 512,
            "total_bytes": 1024,
            "bytes_per_second": 256,
        }
    )
    _progress({"event": "complete", "path": "a", "destination": "output/a"})
    stderr = capsys.readouterr().err
    assert "50.00%" in stderr and "Downloaded a" in stderr
    assert _human_size(2 * 1024**4) == "2.00 TiB"
