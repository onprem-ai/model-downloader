"""Durable, fenced model-directory download queue."""

from __future__ import annotations

import asyncio
import inspect
import json
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from opai_models.async_client import AsyncModelClient
from opai_models.client import (
    ModelDownloadError,
    TransientModelDownloadError,
    _AsyncLicenseTransport,
)
from opai_models.download import (
    ChecksumMismatchError,
    DownloadCancelled,
    RetryableDownloadError,
)
from opai_models.errors import sanitize_error_detail
from opai_models.metadata import SourceDocument
from opai_models.snapshot import ModelSnapshot

JobState = Literal[
    "queued",
    "snapshotting",
    "downloading",
    "retry_wait",
    "verifying",
    "completed",
    "failed",
    "cancel_requested",
    "cancelled",
]
FileState = Literal["queued", "downloading", "verifying", "completed", "failed"]
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_RUNNING = frozenset({"snapshotting", "downloading", "retry_wait", "verifying"})
_JOB_COLUMNS = """id, model_id, destination, state, completed_bytes, total_bytes,
completed_files, total_files, bytes_per_second, run_count, consecutive_failures,
next_retry_at, last_progress_at, snapshot_sha256, error_code, error_message,
created_at, updated_at, started_at, completed_at, worker_id, lease_expires_at,
heartbeat_at"""
_FILE_COLUMNS = """id, job_id, object_path, relative_path, state, expected_size,
completed_bytes, source_id, expected_sha256, etag, version_id, computed_sha256,
integrity_failures, error_code, error_message, created_at, updated_at, completed_at"""


class JobNotFoundError(KeyError):
    """A requested job does not exist."""


class JobConflictError(RuntimeError):
    """A requested queue transition conflicts with durable state."""


@dataclass(frozen=True)
class DownloadJob:
    id: str
    model_id: str
    destination: str
    state: JobState
    completed_bytes: int
    total_bytes: int | None
    completed_files: int
    total_files: int | None
    bytes_per_second: int | None
    run_count: int
    consecutive_failures: int
    next_retry_at: str | None
    last_progress_at: str
    snapshot_sha256: str | None
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    worker_id: str | None
    lease_expires_at: str | None
    heartbeat_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DownloadFile:
    id: str
    job_id: str
    object_path: str
    relative_path: str
    state: FileState
    expected_size: int
    completed_bytes: int
    source_id: str
    expected_sha256: str | None
    etag: str | None
    version_id: str | None
    computed_sha256: str | None
    integrity_failures: int
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class DownloadError:
    id: int
    job_id: str
    file_id: str | None
    chunk_index: int | None
    occurred_at: str
    error_type: str
    message: str
    retryable: bool
    http_status: int | None
    request_attempt: int | None
    backoff_seconds: float | None


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _safe_error(value: object, maximum: int = 500) -> str:
    return sanitize_error_detail(value, maximum)


class SQLiteQueueStore:
    """Single-host SQLite WAL queue with expiring leases and fencing tokens."""

    _EXPECTED_COLUMNS = {
        "download_jobs": {
            "id",
            "model_id",
            "destination",
            "state",
            "completed_bytes",
            "total_bytes",
            "completed_files",
            "total_files",
            "bytes_per_second",
            "run_count",
            "consecutive_failures",
            "next_retry_at",
            "last_progress_at",
            "snapshot_sha256",
            "source_json",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
            "worker_id",
            "claim_token",
            "lease_expires_at",
            "heartbeat_at",
        },
        "download_files": {
            "id",
            "job_id",
            "object_path",
            "relative_path",
            "state",
            "expected_size",
            "completed_bytes",
            "source_id",
            "expected_sha256",
            "etag",
            "version_id",
            "computed_sha256",
            "integrity_failures",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
            "completed_at",
        },
        "download_chunks": {
            "file_id",
            "chunk_index",
            "start_byte",
            "end_byte",
            "completed",
            "completed_at",
        },
        "download_errors": {
            "id",
            "job_id",
            "file_id",
            "chunk_index",
            "occurred_at",
            "error_type",
            "message",
            "retryable",
            "http_status",
            "request_attempt",
            "backoff_seconds",
        },
    }

    def __init__(self, database_path: Path, *, busy_timeout_ms: int = 5000) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize()
        self.database_path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path, timeout=self.busy_timeout_ms / 1000, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        # SQLite does not accept bound parameters in PRAGMA statements. The
        # value is normalized to int at construction and therefore cannot
        # inject SQL.
        connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")  # noqa: S608
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            if str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
                raise RuntimeError("SQLite WAL mode is required")
            connection.execute("PRAGMA synchronous=FULL")
            existing = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if existing:
                if existing != set(self._EXPECTED_COLUMNS):
                    raise RuntimeError("incompatible downloader database; delete it and restart")
                actual = {
                    table: {
                        str(row[1])
                        for row in connection.execute(f"PRAGMA table_info({table})")  # noqa: S608
                    }
                    for table in self._EXPECTED_COLUMNS
                }
                if actual != self._EXPECTED_COLUMNS:
                    raise RuntimeError("incompatible downloader database; delete it and restart")
                return
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS download_jobs (
                  id TEXT PRIMARY KEY, model_id TEXT NOT NULL, destination TEXT NOT NULL,
                  state TEXT NOT NULL CHECK(state IN ('queued','snapshotting','downloading',
                    'retry_wait','verifying','completed','failed','cancel_requested','cancelled')),
                  completed_bytes INTEGER NOT NULL DEFAULT 0 CHECK(completed_bytes >= 0),
                  total_bytes INTEGER CHECK(total_bytes IS NULL OR total_bytes >= 0),
                  completed_files INTEGER NOT NULL DEFAULT 0 CHECK(completed_files >= 0),
                  total_files INTEGER CHECK(total_files IS NULL OR total_files >= 0),
                  bytes_per_second INTEGER CHECK(bytes_per_second IS NULL OR bytes_per_second >= 0),
                  run_count INTEGER NOT NULL DEFAULT 0 CHECK(run_count >= 0),
                  consecutive_failures INTEGER NOT NULL DEFAULT 0 CHECK(consecutive_failures >= 0),
                  next_retry_at TEXT, last_progress_at TEXT NOT NULL, snapshot_sha256 TEXT,
                  source_json TEXT, error_code TEXT, error_message TEXT, created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
                  worker_id TEXT, claim_token TEXT, lease_expires_at TEXT, heartbeat_at TEXT
                );
                CREATE TABLE IF NOT EXISTS download_files (
                  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES download_jobs(id) ON DELETE CASCADE,
                  object_path TEXT NOT NULL, relative_path TEXT NOT NULL,
                  state TEXT NOT NULL CHECK(state IN ('queued','downloading','verifying','completed','failed')),
                  expected_size INTEGER NOT NULL CHECK(expected_size >= 0),
                  completed_bytes INTEGER NOT NULL DEFAULT 0 CHECK(completed_bytes >= 0),
                  source_id TEXT NOT NULL, expected_sha256 TEXT, etag TEXT, version_id TEXT,
                  computed_sha256 TEXT,
                  integrity_failures INTEGER NOT NULL DEFAULT 0 CHECK(integrity_failures >= 0),
                  error_code TEXT, error_message TEXT,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT,
                  UNIQUE(job_id, relative_path)
                );
                CREATE TABLE IF NOT EXISTS download_chunks (
                  file_id TEXT NOT NULL REFERENCES download_files(id) ON DELETE CASCADE,
                  chunk_index INTEGER NOT NULL CHECK(chunk_index >= 0),
                  start_byte INTEGER NOT NULL CHECK(start_byte >= 0), end_byte INTEGER NOT NULL,
                  completed INTEGER NOT NULL DEFAULT 0 CHECK(completed IN (0,1)), completed_at TEXT,
                  PRIMARY KEY(file_id, chunk_index), CHECK(end_byte >= start_byte)
                );
                CREATE TABLE IF NOT EXISTS download_errors (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL REFERENCES download_jobs(id) ON DELETE CASCADE,
                  file_id TEXT REFERENCES download_files(id) ON DELETE CASCADE,
                  chunk_index INTEGER, occurred_at TEXT NOT NULL,
                  error_type TEXT NOT NULL, message TEXT NOT NULL,
                  retryable INTEGER NOT NULL CHECK(retryable IN (0,1)),
                  http_status INTEGER, request_attempt INTEGER,
                  backoff_seconds REAL CHECK(backoff_seconds IS NULL OR backoff_seconds >= 0),
                  FOREIGN KEY(file_id, chunk_index)
                    REFERENCES download_chunks(file_id, chunk_index),
                  CHECK((file_id IS NULL AND chunk_index IS NULL) OR file_id IS NOT NULL)
                );
                CREATE INDEX IF NOT EXISTS download_jobs_claim ON download_jobs(state, created_at);
                CREATE INDEX IF NOT EXISTS download_jobs_lease ON download_jobs(state, lease_expires_at);
                CREATE INDEX IF NOT EXISTS download_jobs_destination ON download_jobs(destination, state);
                CREATE INDEX IF NOT EXISTS download_files_job ON download_files(job_id, relative_path);
                CREATE INDEX IF NOT EXISTS download_errors_job ON download_errors(job_id, occurred_at);
                CREATE INDEX IF NOT EXISTS download_errors_chunk
                  ON download_errors(file_id, chunk_index, occurred_at);
                """
            )

    @staticmethod
    def _job(row: sqlite3.Row) -> DownloadJob:
        return DownloadJob(**{name: row[name] for name in DownloadJob.__dataclass_fields__})

    @staticmethod
    def _file(row: sqlite3.Row) -> DownloadFile:
        return DownloadFile(**{name: row[name] for name in DownloadFile.__dataclass_fields__})

    @staticmethod
    def _error(row: sqlite3.Row) -> DownloadError:
        values = {name: row[name] for name in DownloadError.__dataclass_fields__}
        values["retryable"] = bool(values["retryable"])
        return DownloadError(**values)

    def enqueue(self, model_id: str, destination: str) -> DownloadJob:
        now, job_id = _timestamp(), str(uuid.uuid4())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            conflict = connection.execute(
                "SELECT id,model_id,state FROM download_jobs "
                "WHERE destination=? ORDER BY created_at DESC LIMIT 1",
                (destination,),
            ).fetchone()
            if conflict and conflict["model_id"] == model_id:
                if conflict["state"] in _RUNNING or conflict["state"] == "queued":
                    connection.commit()
                    return self.get(str(conflict["id"]))
                if conflict["state"] == "failed":
                    connection.execute(
                        "UPDATE download_jobs SET state='queued',consecutive_failures=0,"
                        "next_retry_at=NULL,error_code=NULL,error_message=NULL,"
                        "completed_at=NULL,updated_at=? WHERE id=?",
                        (now, conflict["id"]),
                    )
                    connection.commit()
                    return self.get(str(conflict["id"]))
            if conflict and conflict["state"] not in _TERMINAL:
                raise JobConflictError(
                    f"an active download already targets this destination: {conflict['id']}"
                )
            connection.execute(
                "INSERT INTO download_jobs(id,model_id,destination,state,last_progress_at,"
                "created_at,updated_at) VALUES(?,?,?,'queued',?,?,?)",
                (job_id, model_id, destination, now, now, now),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get(job_id)

    def get(self, job_id: str) -> DownloadJob:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                f"SELECT {_JOB_COLUMNS} FROM download_jobs WHERE id=?", (job_id,)
            ).fetchone()  # noqa: S608
        if row is None:
            raise JobNotFoundError(job_id)
        return self._job(row)

    def list(self, *, limit: int = 100, state: JobState | None = None) -> list[DownloadJob]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        if state is None:
            sql, values = (
                f"SELECT {_JOB_COLUMNS} FROM download_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )  # noqa: S608
        else:
            sql, values = (
                f"SELECT {_JOB_COLUMNS} FROM download_jobs WHERE state=? ORDER BY created_at DESC LIMIT ?",
                (state, limit),
            )  # noqa: S608
        with closing(self._connect()) as connection, connection:
            return [self._job(row) for row in connection.execute(sql, values)]

    def file(self, file_id: str) -> DownloadFile:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                f"SELECT {_FILE_COLUMNS} FROM download_files WHERE id=?", (file_id,)
            ).fetchone()  # noqa: S608
        if row is None:
            raise JobNotFoundError(file_id)
        return self._file(row)

    def files(self, job_id: str) -> list[DownloadFile]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                f"SELECT {_FILE_COLUMNS} FROM download_files WHERE job_id=? ORDER BY relative_path",
                (job_id,),
            ).fetchall()  # noqa: S608
        return [self._file(row) for row in rows]

    def claim(self, worker_id: str, lease_seconds: int) -> tuple[DownloadJob, str] | None:
        now_value = _now()
        now = _timestamp(now_value)
        expires = _timestamp(now_value + timedelta(seconds=lease_seconds))
        token = uuid.uuid4().hex
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE download_jobs SET state='cancelled',completed_at=?,updated_at=?,worker_id=NULL,claim_token=NULL,lease_expires_at=NULL WHERE state='cancel_requested' AND (worker_id IS NULL OR lease_expires_at<?)",
                (now, now, now),
            )
            row = connection.execute(
                "SELECT id FROM download_jobs WHERE state='queued' "
                "OR (state='retry_wait' AND next_retry_at<=?) "
                "OR (state IN ('snapshotting','downloading','verifying') AND lease_expires_at<?) "
                "ORDER BY created_at,id LIMIT 1",
                (now, now),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            claimed = connection.execute(
                f"UPDATE download_jobs SET state=CASE WHEN snapshot_sha256 IS NULL THEN 'snapshotting' ELSE 'downloading' END,worker_id=?,claim_token=?,lease_expires_at=?,heartbeat_at=?,run_count=run_count+1,started_at=COALESCE(started_at,?),updated_at=?,next_retry_at=NULL,error_code=NULL,error_message=NULL WHERE id=? RETURNING {_JOB_COLUMNS}",  # noqa: S608
                (worker_id, token, expires, now, now, now, row["id"]),
            ).fetchone()
            connection.commit()
            if claimed is None:  # Defensive: UPDATE ... RETURNING must return the selected row.
                raise RuntimeError("claimed queue row disappeared")
            return self._job(claimed), token
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat(self, job_id: str, token: str, lease_seconds: int) -> bool:
        now_value = _now()
        now = _timestamp(now_value)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE download_jobs SET lease_expires_at=?,heartbeat_at=?,updated_at=? WHERE id=? AND claim_token=? AND state IN ('snapshotting','downloading','verifying')",
                (_timestamp(now_value + timedelta(seconds=lease_seconds)), now, now, job_id, token),
            )
        return cursor.rowcount == 1

    def save_snapshot(self, job_id: str, token: str, snapshot: ModelSnapshot) -> bool:
        now = _timestamp()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                "SELECT id FROM download_jobs WHERE id=? AND claim_token=? AND state='snapshotting'",
                (job_id, token),
            ).fetchone()
            if owned is None:
                connection.rollback()
                return False
            connection.execute("DELETE FROM download_files WHERE job_id=?", (job_id,))
            for item in snapshot.files:
                connection.execute(
                    "INSERT INTO download_files(id,job_id,object_path,relative_path,state,expected_size,source_id,expected_sha256,etag,version_id,created_at,updated_at) VALUES(?,?,?,?,'queued',?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        job_id,
                        item.object_path,
                        item.relative_path,
                        item.size,
                        item.source_id,
                        item.sha256,
                        item.etag,
                        item.version_id,
                        now,
                        now,
                    ),
                )
            if snapshot.source is None:
                raise ValueError("snapshot source provenance is required")
            source_json = json.dumps(
                snapshot.source.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            cursor = connection.execute(
                "UPDATE download_jobs SET state='downloading',total_bytes=?,total_files=?,"
                "snapshot_sha256=?,source_json=?,last_progress_at=?,updated_at=? "
                "WHERE id=? AND claim_token=? AND state='snapshotting'",
                (
                    snapshot.total_bytes,
                    snapshot.file_count,
                    snapshot.snapshot_sha256,
                    source_json,
                    now,
                    now,
                    job_id,
                    token,
                ),
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def source(self, job_id: str) -> SourceDocument:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT source_json FROM download_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        if not row["source_json"]:
            raise JobConflictError("job has no source provenance")
        try:
            return SourceDocument.from_dict(json.loads(row["source_json"]))
        except (json.JSONDecodeError, ModelDownloadError):
            raise JobConflictError("job has invalid source provenance") from None

    def prepare_chunks(self, file_id: str, token: str, size: int, chunk_size: int) -> bool:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                "SELECT 1 FROM download_files f JOIN download_jobs j ON j.id=f.job_id "
                "WHERE f.id=? AND j.claim_token=? AND j.state IN ('downloading','verifying')",
                (file_id, token),
            ).fetchone()
            if owned is None:
                connection.rollback()
                return False
            expected = [
                (index, start, min(start + chunk_size, size) - 1)
                for index, start in enumerate(range(0, size, chunk_size))
            ]
            existing = [
                tuple(row)
                for row in connection.execute(
                    "SELECT chunk_index,start_byte,end_byte FROM download_chunks "
                    "WHERE file_id=? ORDER BY chunk_index",
                    (file_id,),
                ).fetchall()
            ]
            if existing and existing != expected:
                raise JobConflictError("chunk size differs from persisted download state")
            if not existing:
                connection.executemany(
                    "INSERT INTO download_chunks(file_id,chunk_index,start_byte,end_byte) "
                    "VALUES(?,?,?,?)",
                    [(file_id, *chunk) for chunk in expected],
                )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def completed_chunks(self, file_id: str) -> set[int]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT chunk_index FROM download_chunks "
                "WHERE file_id=? AND completed=1 ORDER BY chunk_index",
                (file_id,),
            ).fetchall()
        return {int(row["chunk_index"]) for row in rows}

    def record_progress(self, file_id: str, token: str, event: dict[str, Any]) -> bool:
        completed = int(event.get("completed_bytes", 0))
        rate = event.get("bytes_per_second")
        bytes_per_second = int(rate) if rate is not None else None
        chunk = event.get("chunk")
        now = _timestamp()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE download_files SET state='downloading',completed_bytes=?,updated_at=? "
                "WHERE id=? AND ? BETWEEN 0 AND expected_size "
                "AND job_id IN (SELECT id FROM download_jobs WHERE claim_token=?)",
                (completed, now, file_id, completed, token),
            )
            if cursor.rowcount and chunk is not None:
                connection.execute(
                    "UPDATE download_chunks SET completed=1,completed_at=? WHERE file_id=? AND chunk_index=?",
                    (now, file_id, int(chunk)),
                )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE download_jobs SET completed_bytes=(SELECT COALESCE(SUM(completed_bytes),0) FROM download_files WHERE job_id=download_jobs.id),bytes_per_second=?,consecutive_failures=0,last_progress_at=?,updated_at=? WHERE id=(SELECT job_id FROM download_files WHERE id=?) AND claim_token=?",
                    (bytes_per_second, now, now, file_id, token),
                )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_error(self, file_id: str, token: str, event: dict[str, Any]) -> bool:
        now = _timestamp()
        message = _safe_error(event.get("message") or type(event).__name__)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT f.job_id FROM download_files f JOIN download_jobs j ON j.id=f.job_id "
                "WHERE f.id=? AND j.claim_token=? AND j.state='downloading'",
                (file_id, token),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            connection.execute(
                "INSERT INTO download_errors(job_id,file_id,chunk_index,occurred_at,"
                "error_type,message,retryable,http_status,request_attempt,backoff_seconds) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    row["job_id"],
                    file_id,
                    event.get("chunk"),
                    now,
                    str(event.get("error_type") or "DownloadError")[:100],
                    message,
                    int(bool(event.get("retryable"))),
                    event.get("http_status"),
                    event.get("attempt"),
                    event.get("backoff_seconds"),
                ),
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_job_error(self, job_id: str, token: str, event: dict[str, Any]) -> bool:
        now = _timestamp()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT INTO download_errors(job_id,file_id,chunk_index,occurred_at,"
                "error_type,message,retryable,http_status,request_attempt,backoff_seconds) "
                "SELECT id,NULL,NULL,?,?,?,?,?,?,? FROM download_jobs "
                "WHERE id=? AND claim_token=? AND state IN "
                "('snapshotting','downloading','verifying')",
                (
                    now,
                    str(event.get("error_type") or "DownloadError")[:100],
                    _safe_error(event.get("message")),
                    int(bool(event.get("retryable"))),
                    event.get("http_status"),
                    event.get("attempt"),
                    event.get("backoff_seconds"),
                    job_id,
                    token,
                ),
            )
        return cursor.rowcount == 1

    def errors(self, job_id: str) -> list[DownloadError]:
        self.get(job_id)
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT id,job_id,file_id,chunk_index,occurred_at,error_type,message,"
                "retryable,http_status,request_attempt,backoff_seconds "
                "FROM download_errors WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [self._error(row) for row in rows]

    def schedule_retry(
        self,
        job_id: str,
        token: str,
        *,
        delay: float,
        error_code: str,
        error_message: str,
    ) -> bool:
        now_value = _now()
        now = _timestamp(now_value)
        retry_at = _timestamp(now_value + timedelta(seconds=max(0.0, delay)))
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE download_jobs SET state='retry_wait',consecutive_failures="
                "consecutive_failures+1,next_retry_at=?,error_code=?,error_message=?,"
                "updated_at=?,bytes_per_second=NULL,worker_id=NULL,claim_token=NULL,"
                "lease_expires_at=NULL,heartbeat_at=NULL WHERE id=? AND claim_token=? "
                "AND state IN ('snapshotting','downloading','verifying')",
                (retry_at, error_code[:100], _safe_error(error_message), now, job_id, token),
            )
        return cursor.rowcount == 1

    def release(self, job_id: str, token: str) -> bool:
        now = _timestamp()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE download_jobs SET state='queued',next_retry_at=NULL,updated_at=?,"
                "bytes_per_second=NULL,worker_id=NULL,claim_token=NULL,lease_expires_at=NULL,"
                "heartbeat_at=NULL WHERE id=? AND claim_token=? "
                "AND state IN ('snapshotting','downloading','verifying')",
                (now, job_id, token),
            )
        return cursor.rowcount == 1

    def reset_file(self, file_id: str, token: str, *, integrity_failure: bool = False) -> bool:
        now = _timestamp()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE download_files SET state='failed',completed_bytes=0,computed_sha256=NULL,"
                "integrity_failures=integrity_failures+?,error_code=?,error_message=?,"
                "updated_at=? WHERE id=? AND job_id IN "
                "(SELECT id FROM download_jobs WHERE claim_token=?)",
                (
                    int(integrity_failure),
                    "checksum_mismatch" if integrity_failure else None,
                    "Final SHA-256 verification failed" if integrity_failure else None,
                    now,
                    file_id,
                    token,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE download_chunks SET completed=0,completed_at=NULL WHERE file_id=?",
                    (file_id,),
                )
                connection.execute(
                    "UPDATE download_jobs SET completed_bytes=(SELECT COALESCE(SUM(completed_bytes),0) FROM download_files WHERE job_id=download_jobs.id),completed_files=(SELECT COUNT(*) FROM download_files WHERE job_id=download_jobs.id AND state='completed'),updated_at=? WHERE id=(SELECT job_id FROM download_files WHERE id=?) AND claim_token=?",
                    (now, file_id, token),
                )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def update_file(
        self,
        file_id: str,
        token: str,
        *,
        completed_bytes: int,
        state: FileState,
        sha256: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        now = _timestamp()
        completed_at = now if state == "completed" else None
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE download_files SET state=?,completed_bytes=?,computed_sha256=?,error_code=?,error_message=?,updated_at=?,completed_at=? WHERE id=? AND job_id IN (SELECT id FROM download_jobs WHERE claim_token=? AND state IN ('downloading','verifying'))",
                (
                    state,
                    completed_bytes,
                    sha256,
                    error_code[:100] if error_code else None,
                    _safe_error(error_message, 1000) if error_message else None,
                    now,
                    completed_at,
                    file_id,
                    token,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE download_jobs SET completed_bytes=(SELECT COALESCE(SUM(completed_bytes),0) FROM download_files WHERE job_id=download_jobs.id),completed_files=(SELECT COUNT(*) FROM download_files WHERE job_id=download_jobs.id AND state='completed'),updated_at=? WHERE id=(SELECT job_id FROM download_files WHERE id=?) AND claim_token=?",
                    (now, file_id, token),
                )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_verifying(self, job_id: str, token: str) -> bool:
        now = _timestamp()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE download_jobs SET state='verifying',updated_at=? "
                "WHERE id=? AND claim_token=? AND state='downloading' "
                "AND total_files IS NOT NULL AND completed_files=total_files "
                "AND total_bytes=completed_bytes",
                (now, job_id, token),
            )
        return cursor.rowcount == 1

    def finish(self, job_id: str, token: str, state: JobState, **fields: Any) -> bool:
        if state not in _TERMINAL:
            raise ValueError("finish state must be terminal")
        now = _timestamp()
        with closing(self._connect()) as connection, connection:
            completion_guard = (
                "AND state='verifying' AND total_files IS NOT NULL "
                "AND completed_files=total_files AND total_bytes=completed_bytes"
                if state == "completed"
                else "AND state IN ('snapshotting','downloading','verifying','cancel_requested')"
            )
            cursor = connection.execute(
                "UPDATE download_jobs SET state=?,error_code=?,error_message=?,completed_at=?,"
                "updated_at=?,bytes_per_second=NULL,worker_id=NULL,claim_token=NULL,"
                f"lease_expires_at=NULL WHERE id=? AND claim_token=? {completion_guard}",  # noqa: S608
                (
                    state,
                    fields.get("error_code"),
                    _safe_error(fields["error_message"], 1000)
                    if fields.get("error_message")
                    else None,
                    now,
                    now,
                    job_id,
                    token,
                ),
            )
        return cursor.rowcount == 1

    def request_cancel(self, job_id: str) -> DownloadJob:
        now = _timestamp()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM download_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if row["state"] == "queued":
                connection.execute(
                    "UPDATE download_jobs SET state='cancelled',completed_at=?,updated_at=? WHERE id=?",
                    (now, now, job_id),
                )
            elif row["state"] not in _TERMINAL:
                connection.execute(
                    "UPDATE download_jobs SET state='cancel_requested',updated_at=? WHERE id=?",
                    (now, job_id),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get(job_id)

    def retry(self, job_id: str) -> DownloadJob:
        now = _timestamp()
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE download_jobs SET state='queued',consecutive_failures=0,"
                "next_retry_at=NULL,error_code=NULL,error_message=NULL,completed_at=NULL,"
                "worker_id=NULL,claim_token=NULL,lease_expires_at=NULL,heartbeat_at=NULL,"
                "updated_at=? WHERE id=? AND state='failed'",
                (now, job_id),
            )
        if cursor.rowcount != 1:
            self.get(job_id)
            raise JobConflictError("job is not retryable")
        return self.get(job_id)

    def cancellation_requested(self, job_id: str, token: str) -> bool:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT state,claim_token FROM download_jobs WHERE id=?", (job_id,)
            ).fetchone()
        return row is None or row["claim_token"] != token or row["state"] == "cancel_requested"


class DownloadManager:
    """Framework-neutral async owner of queued model downloads."""

    def __init__(
        self,
        database_path: Path,
        download_directory: Path,
        client: AsyncModelClient,
        *,
        max_concurrent_downloads: int = 2,
        lease_seconds: int = 120,
        poll_interval: float = 0.5,
        chunk_size: int = 64 * 1024 * 1024,
        range_workers: int = 4,
        request_retries: int = 8,
        integrity_retries: int = 2,
        initial_backoff: float = 0.5,
        max_backoff: float = 60.0,
        no_progress_timeout: float = 3600.0,
        overall_timeout: float = 0.0,
    ) -> None:
        if max_concurrent_downloads < 1:
            raise ValueError("max_concurrent_downloads must be positive")
        if lease_seconds < 30:
            raise ValueError("lease_seconds must be at least 30")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if chunk_size < 1024 * 1024 or not 1 <= range_workers <= 32:
            raise ValueError("invalid chunk size or range worker count")
        if request_retries < 0 or integrity_retries < 0:
            raise ValueError("retry counts must not be negative")
        if initial_backoff <= 0 or max_backoff < initial_backoff:
            raise ValueError("invalid retry backoff")
        if no_progress_timeout <= 0:
            raise ValueError("no_progress_timeout must be positive")
        if overall_timeout < 0:
            raise ValueError("overall_timeout must not be negative")
        self.download_directory = Path(download_directory).expanduser().resolve()
        self.download_directory.mkdir(parents=True, exist_ok=True)
        self.client, self.max_concurrent_downloads, self.lease_seconds = (
            client,
            max_concurrent_downloads,
            lease_seconds,
        )
        self.poll_interval, self.chunk_size, self.range_workers = (
            poll_interval,
            chunk_size,
            range_workers,
        )
        self.request_retries = request_retries
        self.integrity_retries = integrity_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.no_progress_timeout = no_progress_timeout
        self.overall_timeout = overall_timeout
        self.store = SQLiteQueueStore(Path(database_path))
        self.worker_id = f"worker-{uuid.uuid4().hex}"
        self._stop, self._wake = asyncio.Event(), asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._cancel_events: dict[str, threading.Event] = {}
        self._shutdown_events: dict[str, threading.Event] = {}

    async def start(self) -> None:
        if not self._tasks:
            self._stop.clear()
            self._tasks = [
                asyncio.create_task(self._worker(i)) for i in range(self.max_concurrent_downloads)
            ]

    async def close(self) -> None:
        self._stop.set()
        self._wake.set()
        for event in self._shutdown_events.values():
            event.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def __aenter__(self) -> DownloadManager:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _destination(self, model_id: str, destination: str | Path | None) -> Path:
        relative = Path(destination) if destination is not None else Path(model_id.rstrip("/")).name
        relative = Path(relative)
        resolved = (
            relative.resolve()
            if relative.is_absolute()
            else (self.download_directory / relative).resolve()
        )
        if not resolved.is_relative_to(self.download_directory):
            raise ValueError("destination must be inside download_directory")
        return resolved

    async def enqueue(
        self,
        model_id: str,
        destination: str | Path | None = None,
    ) -> DownloadJob:
        model = _AsyncLicenseTransport._model_id(model_id)
        job = await asyncio.to_thread(
            self.store.enqueue, model, str(self._destination(model, destination))
        )
        self._wake.set()
        return job

    async def get(self, job_id: str) -> DownloadJob:
        return await asyncio.to_thread(self.store.get, job_id)

    async def errors(self, job_id: str) -> list[DownloadError]:
        return await asyncio.to_thread(self.store.errors, job_id)

    async def list(self, *, limit: int = 100, state: JobState | None = None) -> list[DownloadJob]:
        return await asyncio.to_thread(self.store.list, limit=limit, state=state)

    async def cancel(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.request_cancel, job_id)
        if event := self._cancel_events.get(job_id):
            event.set()
        self._wake.set()
        return job

    async def retry(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.retry, job_id)
        self._wake.set()
        return job

    async def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = 0.25,
        on_update: Callable[[DownloadJob], Awaitable[None]] | None = None,
    ) -> DownloadJob:
        if on_update is not None and not inspect.iscoroutinefunction(on_update):
            raise TypeError("on_update must be an async callable")
        previous: DownloadJob | None = None
        while True:
            job = await self.get(job_id)
            if on_update is not None and job != previous:
                await on_update(job)
            if job.state in _TERMINAL:
                return job
            previous = job
            await asyncio.sleep(poll_interval)

    async def _worker(self, index: int) -> None:
        worker = f"{self.worker_id}-{index}"
        while not self._stop.is_set():
            claimed = await asyncio.to_thread(self.store.claim, worker, self.lease_seconds)
            if claimed is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), self.poll_interval)
                except TimeoutError:
                    pass
                continue
            await self._execute(*claimed)

    async def _execute(self, job: DownloadJob, token: str) -> None:
        cancel = threading.Event()
        shutdown = threading.Event()
        deadline_exceeded = threading.Event()
        deadline = (
            datetime.fromisoformat(job.started_at) + timedelta(seconds=self.overall_timeout)
            if self.overall_timeout and job.started_at
            else None
        )

        def should_stop() -> bool:
            if cancel.is_set() or shutdown.is_set():
                return True
            if deadline is not None and _now() >= deadline:
                deadline_exceeded.set()
                return True
            return False

        self._cancel_events[job.id] = cancel
        self._shutdown_events[job.id] = shutdown
        heartbeat = asyncio.create_task(self._heartbeat(job.id, token, cancel, shutdown))
        try:
            if job.snapshot_sha256 is None:
                snapshot = await self.client.snapshot_model(job.model_id)
                if not await asyncio.to_thread(self.store.save_snapshot, job.id, token, snapshot):
                    return
            # Directory transfer is implemented by AsyncModelClient against the persisted snapshot.
            await self.client._pull_job(
                job.id,
                Path(job.destination),
                self.store,
                token,
                chunk_size=self.chunk_size,
                workers=self.range_workers,
                request_retries=self.request_retries,
                initial_backoff=self.initial_backoff,
                max_backoff=self.max_backoff,
                should_cancel=should_stop,
            )
            if not await asyncio.to_thread(self.store.finish, job.id, token, "completed"):
                raise DownloadCancelled("download lease was lost")
        except DownloadCancelled:
            current = await asyncio.to_thread(self.store.get, job.id)
            if shutdown.is_set():
                await asyncio.to_thread(self.store.release, job.id, token)
            elif deadline_exceeded.is_set():
                await asyncio.to_thread(
                    self.store.record_job_error,
                    job.id,
                    token,
                    {
                        "error_type": "overall_timeout",
                        "message": "Download exceeded the configured overall timeout",
                        "retryable": False,
                    },
                )
                await asyncio.to_thread(
                    self.store.finish,
                    job.id,
                    token,
                    "failed",
                    error_code="overall_timeout",
                    error_message="Download exceeded the configured overall timeout",
                )
            elif current.state == "cancel_requested":
                await asyncio.to_thread(
                    self.store.finish,
                    job.id,
                    token,
                    "cancelled",
                    error_code="cancelled",
                    error_message="Download cancelled",
                )
            else:
                await asyncio.to_thread(self.store.release, job.id, token)
                self._wake.set()
        except Exception as exc:
            current = await asyncio.to_thread(self.store.get, job.id)
            cancelled = current.state == "cancel_requested"
            if cancelled:
                await asyncio.to_thread(
                    self.store.finish,
                    job.id,
                    token,
                    "cancelled",
                    error_code="cancelled",
                    error_message="Download cancelled",
                )
            else:
                now = _now()
                last_progress = datetime.fromisoformat(current.last_progress_at)
                # No-progress timeout is meaningful only for retryable transfer
                # failures. Permanent metadata errors must retain their real type.
                is_retryable_failure = isinstance(
                    exc,
                    (
                        RetryableDownloadError,
                        TransientModelDownloadError,
                        ChecksumMismatchError,
                    ),
                )
                stalled = (
                    is_retryable_failure
                    and (now - last_progress).total_seconds() >= self.no_progress_timeout
                )
                integrity_exhausted = False
                if isinstance(exc, ChecksumMismatchError):
                    failed_files = [
                        item
                        for item in await asyncio.to_thread(self.store.files, job.id)
                        if item.integrity_failures > self.integrity_retries
                    ]
                    integrity_exhausted = bool(failed_files)
                retryable = is_retryable_failure and not stalled and not integrity_exhausted
                if retryable:
                    cap = min(
                        self.max_backoff,
                        self.initial_backoff * (2 ** min(current.consecutive_failures, 16)),
                    )
                    delay = secrets.randbelow(max(1, int(cap * 1000) + 1)) / 1000
                    # Preserve the dependency's operator-safe exception detail.
                    # The store applies URL, bearer-token, and license redaction.
                    message = f"Temporary download failure ({type(exc).__name__}): {exc}"
                    await asyncio.to_thread(
                        self.store.record_job_error,
                        job.id,
                        token,
                        {
                            "error_type": type(exc).__name__,
                            "message": message,
                            "retryable": True,
                            "backoff_seconds": delay,
                        },
                    )
                    await asyncio.to_thread(
                        self.store.schedule_retry,
                        job.id,
                        token,
                        delay=delay,
                        error_code=type(exc).__name__,
                        error_message=message,
                    )
                    self._wake.set()
                else:
                    error_code = (
                        "no_progress_timeout"
                        if stalled
                        else "integrity_retries_exhausted"
                        if integrity_exhausted
                        else type(exc).__name__
                    )
                    message = (
                        f"Download made no progress before the configured timeout: {exc}"
                        if stalled
                        else f"Download failed repeated integrity verification: {exc}"
                        if integrity_exhausted
                        else f"Download failed ({type(exc).__name__}): {exc}"
                    )
                    await asyncio.to_thread(
                        self.store.record_job_error,
                        job.id,
                        token,
                        {
                            "error_type": error_code,
                            "message": message,
                            "retryable": False,
                        },
                    )
                    await asyncio.to_thread(
                        self.store.finish,
                        job.id,
                        token,
                        "failed",
                        error_code=error_code,
                        error_message=message,
                    )
        finally:
            cancel.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._cancel_events.pop(job.id, None)
            self._shutdown_events.pop(job.id, None)

    async def _heartbeat(
        self,
        job_id: str,
        token: str,
        cancel: threading.Event,
        shutdown: threading.Event | None = None,
    ) -> None:
        while (
            not cancel.is_set() and not (shutdown and shutdown.is_set()) and not self._stop.is_set()
        ):
            await asyncio.sleep(max(1, min(5, self.lease_seconds // 3)))
            if await asyncio.to_thread(
                self.store.cancellation_requested, job_id, token
            ) or not await asyncio.to_thread(
                self.store.heartbeat, job_id, token, self.lease_seconds
            ):
                cancel.set()
                return
