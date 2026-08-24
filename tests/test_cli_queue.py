from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from opai_models.cli import main
from opai_models.manager import DownloadJob


def job(state: str = "completed") -> DownloadJob:
    return DownloadJob(
        "id",
        "example",
        "/downloads/example",
        state,
        10,
        10,
        2,
        2,
        None,
        1,
        3,
        "sha256:" + "a" * 64,
        None,
        None,
        "now",
        "now",
        "now",
        "now" if state == "completed" else None,
        None,
        None,
        None,
    )


def test_cli_pull_uses_directory_manager(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    manager = MagicMock()
    manager.start = AsyncMock()
    manager.close = AsyncMock()
    manager.enqueue = AsyncMock(return_value=job("queued"))
    manager.wait = AsyncMock(return_value=job())
    with patch("opai_models.cli.DownloadManager", return_value=manager):
        assert (
            main(
                [
                    "pull",
                    "example",
                    "example",
                    "--download-directory",
                    str(tmp_path),
                    "--database",
                    str(tmp_path / "queue.sqlite"),
                    "--skip-signature-verification",
                ]
            )
            == 0
        )
    manager.enqueue.assert_awaited_once_with("example", Path("example"))
    manager.close.assert_awaited_once()
    assert manager.wait.call_args.kwargs["on_update"] is not None
    assert "Downloaded 2 files" in capsys.readouterr().err


def test_cli_pull_forwards_verification_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    manager = MagicMock()
    manager.start = AsyncMock()
    manager.close = AsyncMock()
    manager.enqueue = AsyncMock(return_value=job("queued"))
    manager.wait = AsyncMock(return_value=job())
    with (
        patch("opai_models.cli.AsyncModelClient") as client_type,
        patch("opai_models.cli.DownloadManager", return_value=manager),
    ):
        assert (
            main(
                [
                    "pull",
                    "example",
                    "--download-directory",
                    str(tmp_path),
                    "--database",
                    str(tmp_path / "queue.sqlite"),
                    "--sigstore-identity",
                    "trusted@example.com",
                    "--sigstore-issuer",
                    "https://issuer.example",
                    "--sigstore-offline",
                    "--skip-checksum-verification",
                ]
            )
            == 0
        )
    kwargs = client_type.call_args.kwargs
    assert kwargs["verify_checksums"] is False
    assert kwargs["verify_signatures"] is True
    assert kwargs["sigstore_identity"] == "trusted@example.com"
    assert kwargs["sigstore_issuer"] == "https://issuer.example"
    assert kwargs["sigstore_offline"] is True


def test_cli_pull_can_explicitly_skip_signature_verification(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    manager = MagicMock()
    manager.start = AsyncMock()
    manager.close = AsyncMock()
    manager.enqueue = AsyncMock(return_value=job("queued"))
    manager.wait = AsyncMock(return_value=job())
    with (
        patch("opai_models.cli.AsyncModelClient") as client_type,
        patch("opai_models.cli.DownloadManager", return_value=manager),
    ):
        assert (
            main(
                [
                    "pull",
                    "example",
                    "--download-directory",
                    str(tmp_path),
                    "--database",
                    str(tmp_path / "queue.sqlite"),
                    "--skip-signature-verification",
                ]
            )
            == 0
        )
    kwargs = client_type.call_args.kwargs
    assert kwargs["verify_checksums"] is True
    assert kwargs["verify_signatures"] is False


def test_cli_pull_failure_is_sanitized(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    failed = job("failed")
    failed = type(failed)(**{**failed.__dict__, "error_message": "Download failed (RuntimeError)"})
    manager = MagicMock()
    manager.start = AsyncMock()
    manager.close = AsyncMock()
    manager.enqueue = AsyncMock(return_value=job("queued"))
    manager.wait = AsyncMock(return_value=failed)
    with patch("opai_models.cli.DownloadManager", return_value=manager):
        assert (
            main(
                [
                    "--json",
                    "pull",
                    "example",
                    "example",
                    "--download-directory",
                    str(tmp_path),
                    "--database",
                    str(tmp_path / "q.sqlite"),
                    "--skip-signature-verification",
                ]
            )
            == 1
        )
    assert "RuntimeError" in capsys.readouterr().err
