"""Persistent, resumable ranged model downloads."""

import base64
import concurrent.futures
import hashlib
import os
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

from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError

_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]


class DownloadCancelled(ModelDownloadError):
    """Raised when a caller cooperatively cancels a transfer."""


class ChecksumMismatchError(ModelDownloadError):
    """Raised when downloaded bytes do not match the immutable inventory."""


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
                    raise ModelDownloadError("source model changed while downloading")
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
    retries: int,
    timeout: float,
    should_cancel: CancelCallback | None = None,
) -> int:
    expected_length = end - start + 1
    for attempt in range(retries + 1):
        if should_cancel and should_cancel():
            raise DownloadCancelled("download cancelled")
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
            raise ModelDownloadError("download URL must use an HTTPS origin")
        headers = dict(access.required_headers)
        headers["Range"] = f"bytes={start}-{end}"
        request = urllib.request.Request(  # noqa: S310 -- scheme validated above
            access.url, headers=headers
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 -- scheme validated above
                request, timeout=timeout
            ) as response:
                if response.status != 206:
                    raise ModelDownloadError(
                        f"range request returned HTTP {response.status}, expected 206"
                    )
                match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (start, end, access.size):
                    raise ModelDownloadError("invalid Content-Range response")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != expected_length:
                    raise ModelDownloadError("invalid ranged Content-Length")
                position = start
                remaining = expected_length
                while remaining:
                    if should_cancel and should_cancel():
                        raise DownloadCancelled("download cancelled")
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ModelDownloadError("truncated ranged response")
                    os.pwrite(descriptor, block, position)
                    position += len(block)
                    remaining -= len(block)
                if response.read(1):
                    raise ModelDownloadError("range response exceeded expected length")
                return expected_length
        except urllib.error.HTTPError as exc:
            try:
                if exc.code in {401, 403}:
                    access_provider.get(refresh=True)
                elif exc.code == 412:
                    raise ModelDownloadError("source model changed (If-Match failed)") from None
                elif exc.code == 416:
                    raise ModelDownloadError("server rejected byte range") from None
                elif exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise ModelDownloadError(f"storage returned HTTP {exc.code}") from None
            finally:
                exc.close()
        except (OSError, urllib.error.URLError, TimeoutError, ModelDownloadError):
            if attempt >= retries:
                raise
        jitter = secrets.randbelow(250) / 1000
        time.sleep(min(30.0, 0.5 * (2**attempt)) + jitter)
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
    expected_sha256: str,
    completed_chunks: set[int],
    mark_complete: Callable[[int, int, int], None],
    chunk_size: int = 64 * 1024 * 1024,
    workers: int = 4,
    retries: int = 5,
    timeout: float = 60,
    should_cancel: CancelCallback | None = None,
    verify_checksum: bool = True,
) -> Path:
    """Download one file while an external durable store owns chunk state."""
    provider = AccessProvider(client, model_id, object_path)
    access = provider.get()
    if access.source_id != expected_source_id or access.size != expected_size:
        raise ModelDownloadError("source model changed while downloading")
    supplied = _expected_sha256({"sha256": expected_sha256})
    current = _expected_sha256(access.checksums)
    if supplied is None:
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

        def transfer(chunk: tuple[int, int, int]) -> None:
            index, start, end = chunk
            _download_range(provider, descriptor, start, end, retries, timeout, should_cancel)
            with lock:
                os.fsync(descriptor)
                completed_chunks.add(index)
                completed_bytes = sum(chunks[i][2] - chunks[i][1] + 1 for i in completed_chunks)
                elapsed = max(time.monotonic() - started, 0.001)
                mark_complete(index, completed_bytes, int(completed_bytes / elapsed))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(transfer, chunk)
                for chunk in chunks
                if chunk[0] not in completed_chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if partial.stat().st_size != expected_size:
        raise ModelDownloadError("final size verification failed")
    if verify_checksum:
        digest = hashlib.sha256()
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.digest() != supplied:
            raise ChecksumMismatchError("final SHA-256 verification failed")
    return partial
