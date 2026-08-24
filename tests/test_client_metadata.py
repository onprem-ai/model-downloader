import io
import urllib.error
from unittest.mock import patch

import pytest

from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def metadata_access(size: int, url: str = "https://s3.example/meta") -> ModelAccess:
    return ModelAccess(".source.json", url, size, "source", "later", {}, {})


def test_read_small_fetches_bounded_metadata() -> None:
    client = LicenseClient("https://license.example", "secret")
    with (
        patch.object(client, "access", return_value=metadata_access(3)) as access,
        patch("urllib.request.urlopen", return_value=Response(b"abc")) as opened,
    ):
        assert client.read_small("a", ".source.json", maximum=3) == b"abc"
    access.assert_called_once_with("a", ".source.json")
    assert opened.call_args.kwargs["timeout"] == 30


def test_read_small_rejects_large_invalid_url_and_wrong_size() -> None:
    client = LicenseClient("https://license.example", "secret")
    with patch.object(client, "access", return_value=metadata_access(4)):
        with pytest.raises(ModelDownloadError, match="too large"):
            client.read_small("a", ".source.json", maximum=3)
    with patch.object(client, "access", return_value=metadata_access(1, "file:///secret")):
        with pytest.raises(ModelDownloadError, match="HTTPS"):
            client.read_small("a", ".source.json")
    with (
        patch.object(client, "access", return_value=metadata_access(3)),
        patch("urllib.request.urlopen", return_value=Response(b"ab")),
    ):
        with pytest.raises(ModelDownloadError, match="size"):
            client.read_small("a", ".source.json")


def test_read_small_sanitizes_network_errors() -> None:
    client = LicenseClient("https://license.example", "secret")
    access = metadata_access(3, "https://s3.example/meta?token=secret")
    with (
        patch.object(client, "access", return_value=access),
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("token=secret")),
    ):
        with pytest.raises(ModelDownloadError, match="cannot read model metadata") as caught:
            client.read_small("a", ".source.json")
    assert "token=secret" not in str(caught.value)
