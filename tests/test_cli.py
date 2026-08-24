import json
from unittest.mock import patch

from opai_models.cli import main
from opai_models.client import ModelAccess


def test_list_json_uses_configured_license(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    listing = {"models": ["a", "b"], "next_cursor": None}
    with patch("opai_models.cli.LicenseClient.list_models", return_value=listing):
        assert main(["--json", "list"]) == 0
    assert json.loads(capsys.readouterr().out) == listing


def test_info_does_not_print_signed_url(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    access = ModelAccess(
        path="config.json",
        url="https://s3.example/test?signature=secret",
        size=10,
        source_id="a" * 64,
        expires_at="2099-01-01T00:00:00Z",
        checksums={},
        required_headers={},
    )
    with patch("opai_models.cli.LicenseClient.access", return_value=access) as fetch:
        assert main(["info", "example", "config.json"]) == 0
    fetch.assert_called_once_with("example", "config.json")
    output = capsys.readouterr().out
    assert "config.json" in output
    assert "signature=secret" not in output


def test_missing_noninteractive_license_is_safe(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPAI_LICENSE_KEY", raising=False)
    assert main(["--no-prompt", "list"]) == 1
    error = capsys.readouterr().err
    assert "OPAI_LICENSE_KEY" in error
