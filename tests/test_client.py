import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from opai_models.client import LicenseClient, ModelDownloadError


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def response(value: object) -> Response:
    return Response(json.dumps(value).encode())


def test_restricts_model_ids_and_relative_paths() -> None:
    client = LicenseClient("https://license.example", "secret")
    for model_id in ("", "namespace/example", "a/b", "..", "bad\n"):
        with pytest.raises(ModelDownloadError, match="model ID"):
            client.access(model_id, "file")
    for path in ("../secret", "a//file", "a\\file", "file\x01", "folder/"):
        with pytest.raises(ModelDownloadError, match="model"):
            client.access("example", path)


def test_relative_directory_prefix_validation() -> None:
    assert LicenseClient._relative_path("nested/", prefix=True) == "nested/"
    assert LicenseClient._relative_path("nested", prefix=True) == "nested/"
    for value in ("", "/nested/", "../nested/", "nested//child/"):
        with pytest.raises(ModelDownloadError, match="model-relative path"):
            LicenseClient._relative_path(value, prefix=True)


def test_access_parses_relative_metadata_without_logging_url() -> None:
    body = {
        "path": "a.bin",
        "url": "https://s3.example/a?signature=secret",
        "size": 12,
        "source_id": "a" * 64,
        "expires_at": "2099-01-01T00:00:00Z",
        "checksums": {"sha256": "b" * 64},
        "required_headers": {"If-Match": "etag"},
    }
    client = LicenseClient("https://license.example/", "license-secret")
    with patch("urllib.request.urlopen", return_value=response(body)) as urlopen:
        result = client.access("example", "a.bin")
    assert result.path == "a.bin"
    assert result.size == 12
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://license.example/v1/models/example/access/a.bin"
    assert request.headers["Authorization"] == "Bearer license-secret"


def test_list_models_follows_cursor_and_validates_ids() -> None:
    client = LicenseClient("https://license.example", "secret")
    pages = [
        {"models": ["b", "a"], "next_cursor": "cursor"},
        {"models": ["a", "c"], "next_cursor": None},
    ]
    with patch.object(client, "_json", side_effect=pages) as request:
        result = client.list_models(limit=2)
    assert result == {"models": ["a", "b", "c"], "next_cursor": None}
    assert "cursor=cursor" in request.call_args_list[1].args[0]


def test_list_models_rejects_bad_results_and_repeated_cursor() -> None:
    client = LicenseClient("https://license.example", "secret")
    with patch.object(client, "_json", return_value={"models": {}, "next_cursor": None}):
        with pytest.raises(ModelDownloadError, match="invalid model listing"):
            client.list_models()
    with patch.object(
        client,
        "_json",
        return_value={"models": [], "next_cursor": "same"},
    ):
        with pytest.raises(ModelDownloadError, match="repeated"):
            client.list_models()
    with patch.object(
        client,
        "_json",
        return_value={"models": ["namespace/model"], "next_cursor": None},
    ):
        with pytest.raises(ModelDownloadError, match="model ID"):
            client.list_models()


def test_list_page_uses_model_id_and_relative_prefix() -> None:
    client = LicenseClient("https://license.example", "secret")
    with patch.object(client, "_json", return_value={}) as request:
        client.list_page("example model", "nested files/", limit=5, cursor="a+b=")
    url = request.call_args.args[0]
    assert url.startswith("/v1/models/example%20model/files?")
    assert "prefix=nested+files%2F" in url
    assert "cursor=a%2Bb%3D" in url


def test_list_all_follows_cursor_and_deduplicates_prefixes() -> None:
    client = LicenseClient("https://license.example", "secret")
    pages = [
        {
            "objects": [{"key": "a", "size": 1}],
            "prefixes": ["sub/"],
            "next_cursor": "cursor",
        },
        {
            "objects": [{"key": "b", "size": 2}],
            "prefixes": ["sub/"],
            "next_cursor": None,
        },
    ]
    with patch.object(client, "list_page", side_effect=pages) as list_page:
        result = client.list_all("example")
    assert [item["key"] for item in result["objects"]] == ["a", "b"]
    assert result["prefixes"] == ["sub/"]
    assert result["model_id"] == "example"
    assert list_page.call_count == 2


def test_list_normalizes_null_collections() -> None:
    client = LicenseClient("https://license.example", "secret")
    with patch.object(
        client,
        "list_page",
        return_value={"objects": None, "prefixes": None, "next_cursor": None},
    ):
        assert client.list_all("example") == {
            "model_id": "example",
            "prefix": "",
            "objects": [],
            "prefixes": [],
            "next_cursor": None,
        }


def test_list_rejects_repeated_cursor_and_malformed_collections() -> None:
    client = LicenseClient("https://license.example", "secret")
    with patch.object(
        client,
        "list_page",
        return_value={"objects": [], "prefixes": [], "next_cursor": "same"},
    ):
        with pytest.raises(ModelDownloadError, match="repeated"):
            client.list_all("example")
    with patch.object(client, "list_page", return_value={"objects": {}, "prefixes": []}):
        with pytest.raises(ModelDownloadError, match="invalid listing"):
            client.list_all("example")


def test_access_rejects_wrong_path_negative_size_and_bad_maps() -> None:
    client = LicenseClient("https://license.example", "secret")
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
        with patch.object(client, "_json", return_value={**base, **change}):
            with pytest.raises(ModelDownloadError, match="invalid model metadata"):
                client.access("example", "a")


def test_http_error_is_sanitized() -> None:
    client = LicenseClient("https://license.example", "license-secret")
    error = urllib.error.HTTPError(
        "https://license.example/path",
        403,
        "forbidden",
        {},
        io.BytesIO(b'{"detail":"not allowed"}'),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ModelDownloadError, match="HTTP 403") as caught:
            client.access("example", "a")
    assert "license-secret" not in str(caught.value)
