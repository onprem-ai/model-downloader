import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from opai_models.manager import JobConflictError, SQLiteQueueStore
from opai_models.metadata import SourceDocument
from opai_models.snapshot import ModelFile, ModelSnapshot


def snapshot() -> ModelSnapshot:
    source = SourceDocument.from_dict(
        {
            "schema_version": 1,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": "a" * 40,
            },
        }
    )
    return ModelSnapshot.create(
        "example",
        [
            ModelFile("a", "a", 3, "source-a", "a" * 64),
            ModelFile("sub/b", "sub/b", 5, "source-b", "b" * 64),
        ],
        source,
    )


def test_schema_is_normalized_wal_and_contains_no_secrets(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    assert store.database_path.stat().st_mode & 0o777 == 0o600
    job = store.enqueue("example", str(tmp_path / "model"))
    store.save_snapshot(job.id, "token", snapshot()) if False else None
    with closing(sqlite3.connect(store.database_path)) as connection, connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "download_jobs",
            "download_files",
            "download_chunks",
            "download_errors",
        } <= tables
        columns = " ".join(
            row[1]
            for table in tables
            for row in connection.execute(f"PRAGMA table_info({table})")  # noqa: S608
        )
    for forbidden in ("license", "signed_url", "authorization", "credential"):
        assert forbidden not in columns
    with closing(sqlite3.connect(store.database_path)) as connection, connection:
        foreign_keys = connection.execute("PRAGMA foreign_key_list(download_errors)").fetchall()
    assert any(row[2:5] == ("download_chunks", "file_id", "file_id") for row in foreign_keys)
    assert any(
        row[2:5] == ("download_chunks", "chunk_index", "chunk_index") for row in foreign_keys
    )


def test_incompatible_database_must_be_deleted(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE download_jobs(id TEXT PRIMARY KEY, old_column TEXT)")
    with pytest.raises(RuntimeError, match="delete it and restart"):
        SQLiteQueueStore(path)


def test_atomic_claim_fencing_snapshot_and_progress(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    queued = store.enqueue("example", str(tmp_path / "model"))
    first = store.claim("one", 60)
    assert first is not None
    job, token = first
    assert job.state == "snapshotting" and job.run_count == 1
    assert store.claim("two", 60) is None
    assert store.save_snapshot(job.id, "wrong", snapshot()) is False
    assert store.save_snapshot(job.id, token, snapshot()) is True
    current = store.get(job.id)
    assert current.state == "downloading"
    assert current.total_files == 2 and current.total_bytes == 8
    files = store.files(job.id)
    assert [item.relative_path for item in files] == ["a", "sub/b"]
    assert store.file(files[0].id) == files[0]
    with pytest.raises(KeyError):
        store.file("missing")
    assert store.prepare_chunks(files[0].id, token, 3, 2)
    assert store.record_progress(
        files[0].id,
        token,
        {
            "event": "chunk_complete",
            "chunk": 0,
            "completed_bytes": 2,
            "bytes_per_second": 100,
        },
    )
    assert store.completed_chunks(files[0].id) == {0}
    progressed = store.get(job.id)
    assert progressed.bytes_per_second == 100
    assert progressed.consecutive_failures == 0
    assert store.record_error(
        files[0].id,
        token,
        {
            "chunk": 1,
            "attempt": 2,
            "error_type": "RetryableDownloadError",
            "message": (
                "temporary https://signed.example/object?token=secret "
                "Bearer hidden " + "ONPRM-" + "AAAAA-BBBBB-CCCCC-DDDDD-EEEEE"
            ),
            "retryable": True,
            "http_status": 503,
            "backoff_seconds": 2.5,
        },
    )
    error = store.errors(job.id)[0]
    assert error.file_id == files[0].id and error.chunk_index == 1
    assert error.retryable and error.http_status == 503
    assert error.request_attempt == 2 and error.backoff_seconds == 2.5
    assert "signed.example" not in error.message
    assert "Bearer hidden" not in error.message
    assert "ONPRM-" not in error.message
    with pytest.raises(KeyError):
        store.errors("missing")
    with pytest.raises(JobConflictError, match="chunk size"):
        store.prepare_chunks(files[0].id, token, 3, 3)
    assert store.update_file(
        files[0].id, token, completed_bytes=3, state="completed", sha256="a" * 64
    )
    assert store.get(job.id).completed_bytes == 3
    assert not store.record_progress(
        files[0].id, token, {"event": "chunk_complete", "completed_bytes": 4}
    )
    assert store.reset_file(files[0].id, token)
    assert store.get(job.id).completed_bytes == 0
    assert store.update_file(
        files[0].id, token, completed_bytes=3, state="completed", sha256="a" * 64
    )
    assert store.finish(job.id, "wrong", "completed") is False
    assert store.finish(job.id, token, "completed") is False
    assert store.update_file(
        files[1].id, token, completed_bytes=5, state="completed", sha256="b" * 64
    )
    assert store.mark_verifying(job.id, token)
    assert store.finish(job.id, token, "completed") is True
    assert store.get(job.id).state == "completed"
    assert queued.id == job.id


def test_destination_conflict_cancel_and_manual_retry(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "same"))
    assert store.enqueue("a", str(tmp_path / "same")).id == job.id
    with pytest.raises(JobConflictError):
        store.enqueue("b", str(tmp_path / "same"))
    assert store.request_cancel(job.id).state == "cancelled"
    with pytest.raises(JobConflictError, match="not retryable"):
        store.retry(job.id)
    replacement = store.enqueue("a", str(tmp_path / "same"))
    assert replacement.id != job.id
    claimed = store.claim("worker", 60)
    assert claimed and claimed[0].id == replacement.id
    assert store.finish(
        replacement.id,
        claimed[1],
        "failed",
        error_code="failed",
        error_message="safe",
    )
    assert store.retry(replacement.id).state == "queued"


def test_job_error_is_sanitized_and_fenced(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"))
    claimed = store.claim("worker", 60)
    assert claimed
    assert not store.record_job_error(job.id, "wrong", {"message": "ignored"})
    assert store.record_job_error(
        job.id,
        claimed[1],
        {
            "error_type": "Temporary",
            "message": "https://signed.example/path?secret=x Bearer abc",
            "retryable": True,
            "backoff_seconds": 3,
        },
    )
    error = store.errors(job.id)[0]
    assert error.file_id is None and error.chunk_index is None
    assert error.message == "[URL REDACTED] Bearer [REDACTED]"


def test_retry_wait_is_durable_and_claimed_when_due(tmp_path: Path, monkeypatch) -> None:
    import opai_models.manager as manager

    current = [manager.datetime(2026, 1, 1, tzinfo=manager.UTC)]
    monkeypatch.setattr(manager, "_now", lambda: current[0])
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"))
    claimed = store.claim("worker", 60)
    assert claimed
    assert store.schedule_retry(
        job.id,
        claimed[1],
        delay=30,
        error_code="RetryableDownloadError",
        error_message="temporary",
    )
    waiting = store.get(job.id)
    assert waiting.state == "retry_wait"
    assert waiting.consecutive_failures == 1
    assert store.claim("early", 60) is None
    current[0] = manager.datetime(2026, 1, 1, 0, 0, 31, tzinfo=manager.UTC)
    resumed = store.claim("later", 60)
    assert resumed and resumed[0].id == job.id and resumed[0].run_count == 2


def test_enqueue_resumes_retryable_failed_job(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"))
    claimed = store.claim("worker", 60)
    assert claimed and store.finish(job.id, claimed[1], "failed")
    resumed = store.enqueue("a", str(tmp_path / "a"))
    assert resumed.id == job.id
    assert resumed.state == "queued"
    assert resumed.consecutive_failures == 0


def test_expired_lease_reclaims_with_new_fencing_token(tmp_path: Path, monkeypatch) -> None:
    import opai_models.manager as manager

    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"))
    times = iter(
        [
            manager.datetime(2026, 1, 1, tzinfo=manager.UTC),
            manager.datetime(2026, 1, 1, 0, 2, tzinfo=manager.UTC),
        ]
    )
    latest = [manager.datetime(2026, 1, 1, tzinfo=manager.UTC)]

    def clock():
        latest[0] = next(times, latest[0])
        return latest[0]

    monkeypatch.setattr(manager, "_now", clock)
    first = store.claim("one", 30)
    second = store.claim("two", 30)
    assert first and second
    assert first[1] != second[1]
    assert store.heartbeat(job.id, first[1], 30) is False
    assert store.finish(job.id, first[1], "failed") is False


def test_cancel_active_job_is_cooperative(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"))
    claimed = store.claim("worker", 60)
    assert claimed
    assert store.request_cancel(job.id).state == "cancel_requested"
    assert store.cancellation_requested(job.id, claimed[1])
    assert store.finish(job.id, claimed[1], "cancelled")


def test_persistence_and_filtered_list(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    job = SQLiteQueueStore(path).enqueue("a", str(tmp_path / "a"))
    reopened = SQLiteQueueStore(path)
    assert reopened.get(job.id) == job
    assert reopened.list(state="queued") == [job]
    with pytest.raises(KeyError):
        reopened.get("missing")
    with pytest.raises(ValueError):
        reopened.list(limit=0)
