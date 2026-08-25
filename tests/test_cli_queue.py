from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from opai_models.cli import main
from opai_models.manager import DownloadJob


def job(state: str = "completed") -> DownloadJob:
    return DownloadJob(
        id="id",
        model_id="example",
        destination="/downloads/example",
        state=state,
        completed_bytes=10,
        total_bytes=10,
        completed_files=2,
        total_files=2,
        bytes_per_second=None,
        run_count=1,
        consecutive_failures=0,
        next_retry_at=None,
        last_progress_at="now",
        snapshot_sha256="sha256:" + "a" * 64,
        error_code=None,
        error_message=None,
        created_at="now",
        updated_at="now",
        started_at="now",
        completed_at="now" if state == "completed" else None,
        worker_id=None,
        lease_expires_at=None,
        heartbeat_at=None,
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
    client = MagicMock()
    client.aclose = AsyncMock()
    with (
        patch("opai_models.cli.AsyncModelClient", return_value=client) as client_type,
        patch("opai_models.cli.DownloadManager", return_value=manager) as manager_type,
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
                    "--request-retries",
                    "12",
                    "--integrity-retries",
                    "4",
                    "--initial-backoff",
                    "1.5",
                    "--max-backoff",
                    "90",
                    "--no-progress-timeout",
                    "7200",
                    "--overall-timeout",
                    "14400",
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
    manager_kwargs = manager_type.call_args.kwargs
    assert manager_kwargs["request_retries"] == 12
    assert manager_kwargs["integrity_retries"] == 4
    assert manager_kwargs["initial_backoff"] == 1.5
    assert manager_kwargs["max_backoff"] == 90
    assert manager_kwargs["no_progress_timeout"] == 7200
    assert manager_kwargs["overall_timeout"] == 14400


def test_cli_pull_can_explicitly_skip_signature_verification(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    manager = MagicMock()
    manager.start = AsyncMock()
    manager.close = AsyncMock()
    manager.enqueue = AsyncMock(return_value=job("queued"))
    manager.wait = AsyncMock(return_value=job())
    client = MagicMock()
    client.aclose = AsyncMock()
    with (
        patch("opai_models.cli.AsyncModelClient", return_value=client) as client_type,
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


def test_cli_sync_forwards_policy(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("OPAI_LICENSE_KEY", "secret")
    result = MagicMock(
        model_id="example",
        destination=tmp_path / "example",
        files=2,
        bytes=10,
        reused_files=1,
        downloaded_files=1,
        deleted_files=0,
        rehashed_files=1,
    )
    client = MagicMock()
    client.sync_model = AsyncMock(return_value=result)
    client.aclose = AsyncMock()
    with patch("opai_models.cli.AsyncModelClient", return_value=client) as client_type:
        assert (
            main(
                [
                    "sync",
                    "example",
                    str(tmp_path / "example"),
                    "--rehash",
                    "--delete",
                    "--request-retries",
                    "12",
                    "--skip-signature-verification",
                ]
            )
            == 0
        )
    assert client_type.call_args.kwargs["verify_checksums"] is True
    assert client_type.call_args.kwargs["verify_signatures"] is False
    assert client.sync_model.call_args.kwargs["rehash"] is True
    assert client.sync_model.call_args.kwargs["delete"] is True
    assert client.sync_model.call_args.kwargs["request_retries"] == 12
    assert "1 reused, 1 downloaded" in capsys.readouterr().err


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
