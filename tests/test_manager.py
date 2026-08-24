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
    job = store.enqueue("example", str(tmp_path / "model"), 3)
    store.save_snapshot(job.id, "token", snapshot()) if False else None
    with closing(sqlite3.connect(store.database_path)) as connection, connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"download_jobs", "download_files", "download_chunks"} <= tables
        columns = " ".join(
            row[1]
            for table in tables
            for row in connection.execute(f"PRAGMA table_info({table})")  # noqa: S608
        )
    for forbidden in ("license", "signed_url", "authorization", "credential"):
        assert forbidden not in columns


def test_atomic_claim_fencing_snapshot_and_progress(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    queued = store.enqueue("example", str(tmp_path / "model"), 3)
    first = store.claim("one", 60)
    assert first is not None
    job, token = first
    assert job.state == "snapshotting" and job.attempt == 1
    assert store.claim("two", 60) is None
    assert store.save_snapshot(job.id, "wrong", snapshot()) is False
    assert store.save_snapshot(job.id, token, snapshot()) is True
    current = store.get(job.id)
    assert current.state == "downloading"
    assert current.total_files == 2 and current.total_bytes == 8
    files = store.files(job.id)
    assert [item.relative_path for item in files] == ["a", "sub/b"]
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
    assert store.get(job.id).bytes_per_second == 100
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


def test_destination_conflict_cancel_retry_and_attempt_limit(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "same"), 1)
    assert store.enqueue("a", str(tmp_path / "same"), 1).id == job.id
    with pytest.raises(JobConflictError):
        store.enqueue("b", str(tmp_path / "same"), 2)
    assert store.request_cancel(job.id).state == "cancelled"
    assert store.retry(job.id).state == "queued"
    claimed = store.claim("worker", 60)
    assert claimed
    assert store.finish(job.id, claimed[1], "failed", error_code="failed", error_message="safe")
    with pytest.raises(JobConflictError, match="not retryable"):
        store.retry(job.id)
    replacement = store.enqueue("a", str(tmp_path / "same"), 2)
    assert replacement.id != job.id


def test_enqueue_resumes_retryable_failed_job(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"), 2)
    claimed = store.claim("worker", 60)
    assert claimed and store.finish(job.id, claimed[1], "failed")
    resumed = store.enqueue("a", str(tmp_path / "a"), 9)
    assert resumed.id == job.id
    assert resumed.state == "queued"
    assert resumed.max_attempts == 2


def test_expired_lease_reclaims_with_new_fencing_token(tmp_path: Path, monkeypatch) -> None:
    import opai_models.manager as manager

    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("a", str(tmp_path / "a"), 3)
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
    job = store.enqueue("a", str(tmp_path / "a"), 2)
    claimed = store.claim("worker", 60)
    assert claimed
    assert store.request_cancel(job.id).state == "cancel_requested"
    assert store.cancellation_requested(job.id, claimed[1])
    assert store.finish(job.id, claimed[1], "cancelled")


def test_persistence_and_filtered_list(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    job = SQLiteQueueStore(path).enqueue("a", str(tmp_path / "a"), 2)
    reopened = SQLiteQueueStore(path)
    assert reopened.get(job.id) == job
    assert reopened.list(state="queued") == [job]
    with pytest.raises(KeyError):
        reopened.get("missing")
    with pytest.raises(ValueError):
        reopened.list(limit=0)
