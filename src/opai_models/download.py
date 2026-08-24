"""Persistent, resumable ranged model downloads."""

import base64
import concurrent.futures
import email.utils
import errno
import hashlib
import os
import queue
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from opai_models.client import (
    LicenseClient,
    ModelAccess,
    ModelDownloadError,
    TransientModelDownloadError,
)

_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
ErrorCallback = Callable[[dict[str, Any]], None]


class DownloadCancelled(ModelDownloadError):
    """Raised when a caller cooperatively cancels a transfer."""


class ChecksumMismatchError(ModelDownloadError):
    """Raised when downloaded bytes do not match the immutable inventory."""


class RetryableDownloadError(ModelDownloadError):
    """A temporary transfer failure that may succeed later."""


class PermanentDownloadError(ModelDownloadError):
    """A transfer failure that retries cannot repair."""


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            return max(0.0, parsed.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def _backoff_delay(
    attempt: int, initial: float, maximum: float, retry_after: float | None
) -> float:
    cap = min(maximum, initial * (2**attempt))
    jitter = secrets.randbelow(max(1, int(cap * 1000) + 1)) / 1000
    return min(maximum, max(jitter, retry_after or 0.0))


def _permanent_os_error(error: OSError) -> bool:
    return error.errno in {errno.EACCES, errno.EDQUOT, errno.ENOSPC, errno.EROFS}


class AccessProvider:
    def __init__(self, client: LicenseClient, model_id: str, object_path: str) -> None:
        self.client = client
        self.model_id = model_id
        self.object_path = object_path
        self._lock = threading.Lock()
        self._access: ModelAccess | None = None
        self.refreshes = 0

    def get(self, *, refresh: bool = False) -> ModelAccess:
        with self._lock:
            if refresh or self._access is None:
                new_access = self.client.access(self.model_id, self.object_path)
                if self._access and (
                    new_access.source_id != self._access.source_id
                    or new_access.size != self._access.size
                    or (
                        _expected_sha256(new_access.checksums) is not None
                        and _expected_sha256(self._access.checksums) is not None
                        and _expected_sha256(new_access.checksums)
                        != _expected_sha256(self._access.checksums)
                    )
                ):
                    raise PermanentDownloadError("source model changed while downloading")
                self._access = new_access
                self.refreshes += 1
            return self._access


def _ranges(size: int, chunk_size: int) -> list[tuple[int, int, int]]:
    return [
        (index, start, min(start + chunk_size, size) - 1)
        for index, start in enumerate(range(0, size, chunk_size))
    ]


def _download_range(
    access_provider: AccessProvider,
    descriptor: int,
    start: int,
    end: int,
    request_retries: int,
    timeout: float,
    should_cancel: CancelCallback | None = None,
    *,
    chunk_index: int = 0,
    initial_backoff: float = 0.5,
    max_backoff: float = 60.0,
    on_error: ErrorCallback | None = None,
) -> int:
    expected_length = end - start + 1
    for attempt in range(request_retries + 1):
        if should_cancel and should_cancel():
            raise DownloadCancelled("download cancelled")
        try:
            access = access_provider.get(refresh=attempt > 0 and attempt % 2 == 0)
            parsed = urllib.parse.urlparse(access.url)
            loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or (parsed.scheme != "https" and not loopback)
            ):
                raise PermanentDownloadError("download URL must use an HTTPS origin")
            headers = dict(access.required_headers)
            headers["Range"] = f"bytes={start}-{end}"
            request = urllib.request.Request(  # noqa: S310 -- scheme validated above
                access.url, headers=headers
            )
            with urllib.request.urlopen(  # noqa: S310 -- scheme validated above
                request, timeout=timeout
            ) as response:
                if response.status != 206:
                    raise PermanentDownloadError(
                        f"range request returned HTTP {response.status}, expected 206"
                    )
                match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (start, end, access.size):
                    raise PermanentDownloadError("invalid Content-Range response")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != expected_length:
                    raise PermanentDownloadError("invalid ranged Content-Length")
                position = start
                remaining = expected_length
                while remaining:
                    if should_cancel and should_cancel():
                        raise DownloadCancelled("download cancelled")
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RetryableDownloadError("truncated ranged response")
                    os.pwrite(descriptor, block, position)
                    position += len(block)
                    remaining -= len(block)
                if response.read(1):
                    raise PermanentDownloadError("range response exceeded expected length")
                return expected_length
        except urllib.error.HTTPError as exc:
            status = exc.code
            retryable = status in {401, 403, 408, 429, 500, 502, 503, 504}
            retry_after = _retry_after(exc.headers.get("Retry-After")) if exc.headers else None
            exc.close()
            if status == 412:
                error = PermanentDownloadError("source model changed (If-Match failed)")
            elif status == 416:
                error = PermanentDownloadError("server rejected byte range")
            elif not retryable:
                error = PermanentDownloadError(f"storage returned HTTP {status}")
            else:
                error = RetryableDownloadError(f"storage temporarily returned HTTP {status}")
            if not retryable:
                if on_error:
                    on_error(
                        {
                            "chunk": chunk_index,
                            "attempt": attempt + 1,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "retryable": False,
                            "http_status": status,
                            "backoff_seconds": None,
                        }
                    )
                raise error from None
            if status in {401, 403}:
                access_provider.get(refresh=True)
        except PermanentDownloadError as error:
            if on_error:
                on_error(
                    {
                        "chunk": chunk_index,
                        "attempt": attempt + 1,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "retryable": False,
                        "http_status": None,
                        "backoff_seconds": None,
                    }
                )
            raise
        except DownloadCancelled:
            raise
        except (
            urllib.error.URLError,
            TimeoutError,
            RetryableDownloadError,
            TransientModelDownloadError,
        ) as caught:
            status = None
            retry_after = None
            message = (
                str(caught) if isinstance(caught, RetryableDownloadError) else type(caught).__name__
            )
            error = RetryableDownloadError(message)
        except OSError as caught:
            if _permanent_os_error(caught):
                error = PermanentDownloadError(type(caught).__name__)
                if on_error:
                    on_error(
                        {
                            "chunk": chunk_index,
                            "attempt": attempt + 1,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "retryable": False,
                            "http_status": None,
                            "backoff_seconds": None,
                        }
                    )
                raise error from None
            status = None
            retry_after = None
            error = RetryableDownloadError(type(caught).__name__)
        delay = _backoff_delay(attempt, initial_backoff, max_backoff, retry_after)
        if on_error:
            on_error(
                {
                    "chunk": chunk_index,
                    "attempt": attempt + 1,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "retryable": True,
                    "http_status": status,
                    "backoff_seconds": delay if attempt < request_retries else None,
                }
            )
        if attempt >= request_retries:
            raise error
        time.sleep(delay)
    raise ModelDownloadError("range retries exhausted")


def _expected_sha256(checksums: dict[str, str]) -> bytes | None:
    value = checksums.get("sha256")
    if not value:
        return None
    if re.fullmatch(r"[A-Fa-f0-9]{64}", value):
        return bytes.fromhex(value)
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError:
        return None
    return decoded if len(decoded) == 32 else None


def pull_file_with_state(
    client: LicenseClient,
    model_id: str,
    object_path: str,
    partial: Path,
    *,
    expected_source_id: str,
    expected_size: int,
    expected_sha256: str | None,
    completed_chunks: set[int],
    mark_complete: Callable[[int, int, int], None],
    chunk_size: int = 64 * 1024 * 1024,
    workers: int = 4,
    request_retries: int = 8,
    timeout: float = 60,
    initial_backoff: float = 0.5,
    max_backoff: float = 60.0,
    should_cancel: CancelCallback | None = None,
    on_error: ErrorCallback | None = None,
    verify_checksum: bool = True,
) -> Path:
    """Download one file while an external durable store owns chunk state."""
    provider = AccessProvider(client, model_id, object_path)
    access = provider.get()
    if access.source_id != expected_source_id or access.size != expected_size:
        raise ModelDownloadError("source model changed while downloading")
    supplied = _expected_sha256({"sha256": expected_sha256 or ""})
    current = _expected_sha256(access.checksums)
    if verify_checksum and supplied is None:
        raise ModelDownloadError("invalid expected model checksum")
    if verify_checksum and current is not None and current != supplied:
        raise ModelDownloadError("source model checksum changed while downloading")
    chunks = _ranges(expected_size, chunk_size)
    if not completed_chunks <= {chunk[0] for chunk in chunks}:
        raise ModelDownloadError("invalid persisted chunk state")
    partial.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
    os.fchmod(descriptor, 0o600)
    lock = threading.Lock()
    started = time.monotonic()
    try:
        os.ftruncate(descriptor, expected_size)

        pending: queue.Queue[tuple[int, int, int]] = queue.Queue()
        for chunk in chunks:
            if chunk[0] not in completed_chunks:
                pending.put(chunk)
        stop = threading.Event()

        def transfer() -> None:
            while not stop.is_set():
                try:
                    index, start, end = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    _download_range(
                        provider,
                        descriptor,
                        start,
                        end,
                        request_retries,
                        timeout,
                        lambda: stop.is_set() or bool(should_cancel and should_cancel()),
                        chunk_index=index,
                        initial_backoff=initial_backoff,
                        max_backoff=max_backoff,
                        on_error=on_error,
                    )
                    with lock:
                        os.fsync(descriptor)
                        completed_chunks.add(index)
                        completed_bytes = sum(
                            chunks[i][2] - chunks[i][1] + 1 for i in completed_chunks
                        )
                        elapsed = max(time.monotonic() - started, 0.001)
                        mark_complete(index, completed_bytes, int(completed_bytes / elapsed))
                except Exception:
                    stop.set()
                    raise
                finally:
                    pending.task_done()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(transfer) for _ in range(min(workers, pending.qsize()))]
            failures: list[BaseException] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    stop.set()
                    failures.append(exc)
            if failures:
                substantive = next(
                    (error for error in failures if not isinstance(error, DownloadCancelled)),
                    failures[0],
                )
                raise substantive
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if partial.stat().st_size != expected_size:
        raise ModelDownloadError("final size verification failed")
    if verify_checksum:
        if supplied is None:
            raise ModelDownloadError("invalid expected model checksum")
        digest = hashlib.sha256()
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.digest() != supplied:
            raise ChecksumMismatchError("final SHA-256 verification failed")
    return partial
